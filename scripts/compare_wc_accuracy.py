"""
Shared 3-class World Cup accuracy comparison for ELO, LR, and Dixon-Coles.

All models are scored on the same labelled match set (LR training rows: WC
2006–2022, 320 matches after dropping 2002 for missing market value). Stratified
5-fold CV uses random_state=42 so fold partitions match across models.

LR: proper cross-validation (re-fit each fold).
ELO / DC: fixed per-match predictions; folds only vary which indices are averaged.
DC: argmax(home_win, draw, away_win) from win probabilities — not scoreline MAE.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
OUTCOMES = ('away_win', 'draw', 'home_win')
WC_GROUP_MATCHES = 48

WinProbFn = Callable[[str, str, bool, bool], tuple[float, float, float]]

RESULTS_TO_TRAIN: dict[str, str] = {'China': 'China PR'}

ELO_ALIASES: dict[str, str] = {
    'USA': 'United States',
    'Czech Republic': 'Czechia',
    'Ivory Coast': 'Ivory Coast',
    'IR Iran': 'Iran',
    'China': 'China PR',
}


@dataclass
class ModelAccuracy:
    name: str
    n_scored: int
    n_total: int
    accuracy: float
    std: float
    fold_accs: list[float]
    note: str = ''


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _normalise_team(name: str, year: int) -> str:
    if name == 'Serbia' and year == 2006:
        return 'Serbia and Montenegro'
    return RESULTS_TO_TRAIN.get(name, name)


def _norm_to_elo(name: str, year: int) -> str:
    if name == 'Serbia' and year == 2006:
        return 'Serbia and Montenegro'
    if name == 'China':
        return 'China PR'
    return name


def _eval_canon(name: str) -> str:
    return ELO_ALIASES.get(name, name)


def argmax_outcome(p_home: float, p_draw: float, p_away: float) -> str:
    probs = {'home_win': p_home, 'draw': p_draw, 'away_win': p_away}
    return max(probs, key=probs.get)


def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, list[float]]:
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    fold_accs: list[float] = []
    for _, test_idx in CV.split(np.zeros(len(y_true)), y_true):
        mask = y_pred[test_idx] != ''
        if mask.sum() == 0:
            fold_accs.append(float('nan'))
            continue
        idx = test_idx[mask]
        fold_accs.append(float((y_pred[idx] == y_true[idx]).mean()))
    valid = [f for f in fold_accs if not np.isnan(f)]
    if not valid:
        return float('nan'), float('nan'), fold_accs
    return float(np.mean(valid)), float(np.std(valid)), fold_accs


def build_wc_eval_df(root: Optional[Path] = None) -> pd.DataFrame:
    """LR-equivalent WC eval set: 320 matches, 2006–2022, shootout-resolved labels."""
    root = root or repo_root()
    data = root / 'data' / 'historical'
    tournament = root / 'data' / 'tournament'

    results = pd.read_csv(data / 'results.csv')
    shootouts = pd.read_csv(data / 'shootouts.csv')
    train_df = pd.read_csv(tournament / 'train.csv')

    results['year'] = pd.to_datetime(results['date']).dt.year
    shootouts['year'] = pd.to_datetime(shootouts['date']).dt.year

    wc = results[
        (results['tournament'] == 'FIFA World Cup')
        & results['year'].between(2002, 2022)
    ].copy().reset_index(drop=True)

    wc_shootouts = shootouts[shootouts['year'].between(2002, 2022)].copy()

    wc['home_team_norm'] = wc.apply(lambda r: _normalise_team(r['home_team'], r['year']), axis=1)
    wc['away_team_norm'] = wc.apply(lambda r: _normalise_team(r['away_team'], r['year']), axis=1)

    train_indexed = train_df.set_index(['version', 'team'])
    feature_cols = [
        'goals_scored_last_4y', 'goals_received_last_4y',
        'wins_last_4y', 'losses_last_4y', 'draws_last_4y',
        'world_cup_titles_before', 'squad_total_market_value_eur',
        'fifa_rank_pre_tournament',
    ]

    def _get_features(team: str, year: int) -> pd.Series | None:
        try:
            return train_indexed.loc[(year, team), feature_cols]
        except KeyError:
            return None

    rows: list[pd.Series] = []
    for _, match in wc.iterrows():
        hf = _get_features(match['home_team_norm'], match['year'])
        af = _get_features(match['away_team_norm'], match['year'])
        if hf is None or af is None:
            continue
        row = match.copy()
        for col, val in hf.items():
            row[f'home_{col}'] = val
        for col, val in af.items():
            row[f'away_{col}'] = val
        rows.append(row)

    df = pd.DataFrame(rows).reset_index(drop=True)

    shootout_lookup: dict[tuple, str] = {}
    for _, row in wc_shootouts.iterrows():
        shootout_lookup[(row['date'], row['home_team'], row['away_team'])] = row['winner']

    def _outcome(row: pd.Series) -> str:
        hs, as_ = row['home_score'], row['away_score']
        if hs > as_:
            return 'home_win'
        if as_ > hs:
            return 'away_win'
        winner = shootout_lookup.get((row['date'], row['home_team'], row['away_team']))
        if winner is not None:
            return 'home_win' if winner == row['home_team'] else 'away_win'
        return 'draw'

    df['outcome'] = df.apply(_outcome, axis=1)
    df = df.sort_values(['year', 'date']).reset_index(drop=True)
    df['is_group_stage'] = df.groupby('year').cumcount() < WC_GROUP_MATCHES
    df['is_neutral'] = (df['neutral'] == 'TRUE').astype(bool)
    df['match_date'] = pd.to_datetime(df['date'])

    # Drop 2002 (missing market value) — matches lr_model.ipynb
    home_mv = df['home_squad_total_market_value_eur'].clip(lower=1e6)
    away_mv = df['away_squad_total_market_value_eur'].clip(lower=1e6)
    df['rank_diff'] = df['home_fifa_rank_pre_tournament'] - df['away_fifa_rank_pre_tournament']
    df['goals_scored_diff'] = df['home_goals_scored_last_4y'] - df['away_goals_scored_last_4y']
    df['goals_conceded_diff'] = df['home_goals_received_last_4y'] - df['away_goals_received_last_4y']
    df['win_rate_diff'] = (
        df['home_wins_last_4y'] / (df['home_wins_last_4y'] + df['home_losses_last_4y'] + df['home_draws_last_4y']).clip(lower=1)
        - df['away_wins_last_4y'] / (df['away_wins_last_4y'] + df['away_losses_last_4y'] + df['away_draws_last_4y']).clip(lower=1)
    )
    df['market_value_ratio'] = np.log(home_mv / away_mv)
    df['titles_diff'] = df['home_world_cup_titles_before'] - df['away_world_cup_titles_before']
    df['is_neutral_int'] = df['is_neutral'].astype(int)

    features = [
        'rank_diff', 'goals_scored_diff', 'goals_conceded_diff',
        'win_rate_diff', 'market_value_ratio', 'titles_diff', 'is_neutral_int',
    ]
    df = df.dropna(subset=features).reset_index(drop=True)
    df.attrs['feature_cols'] = features
    return df


def evaluate_lr(df: pd.DataFrame) -> ModelAccuracy:
    features: list[str] = df.attrs['feature_cols']
    x = df[features].values
    y = np.asarray(df['outcome'].tolist(), dtype=object)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)),
    ])
    cv_res = cross_validate(
        pipeline, x, y, cv=CV, scoring='accuracy', return_train_score=False,
    )
    fold_accs = list(cv_res.get('test_accuracy', cv_res['test_score']))
    return ModelAccuracy(
        name='Logistic Regression',
        n_scored=len(df),
        n_total=len(df),
        accuracy=float(np.mean(fold_accs)),
        std=float(np.std(fold_accs)),
        fold_accs=fold_accs,
        note='5-fold CV with re-fit each fold',
    )


def _load_elo_hist(root: Path) -> pd.DataFrame:
    elo = pd.read_csv(root / 'data' / 'tournament' / 'elo_ratings_wc2026.csv')
    elo['snapshot_date'] = pd.to_datetime(elo['snapshot_date'])
    return elo.sort_values(['country', 'snapshot_date']).reset_index(drop=True)


def _load_hist_ranks(root: Path) -> dict[tuple, float]:
    train = pd.read_csv(root / 'data' / 'tournament' / 'train.csv')
    ranks: dict[tuple, float] = {}
    for _, row in train.iterrows():
        ranks[(int(row['version']), row['team'])] = row['fifa_rank_pre_tournament']
    return ranks


def _get_elo_at(elo_hist: pd.DataFrame, team: str, match_date: pd.Timestamp) -> float | None:
    for name in (_eval_canon(team), team):
        mask = (elo_hist['country'] == name) & (elo_hist['snapshot_date'] <= match_date)
        rows = elo_hist[mask]
        if not rows.empty:
            return float(rows.iloc[-1]['rating'])
    return None


def _get_hist_rank(ranks: dict[tuple, float], team: str, year: int) -> float:
    for name in (team, _norm_to_elo(team, year), _eval_canon(team)):
        v = ranks.get((year, name))
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 999.0


def evaluate_elo(df: pd.DataFrame, root: Optional[Path] = None) -> ModelAccuracy:
    root = root or repo_root()
    elo_hist = _load_elo_hist(root)
    hist_ranks = _load_hist_ranks(root)
    elo_countries = set(elo_hist['country'].unique())

    def _has_elo(team: str) -> bool:
        c = _eval_canon(team)
        return c in elo_countries or team in elo_countries

    y_true = np.asarray(df['outcome'].tolist(), dtype=object)
    y_pred = np.empty(len(df), dtype=object)

    for i, row in df.iterrows():
        home = _norm_to_elo(row['home_team_norm'], int(row['year']))
        away = _norm_to_elo(row['away_team_norm'], int(row['year']))
        if not (_has_elo(home) and _has_elo(away)):
            y_pred[i] = ''
            continue
        md = row['match_date']
        yr = int(row['year'])
        h_elo = _get_elo_at(elo_hist, home, md)
        a_elo = _get_elo_at(elo_hist, away, md)
        if h_elo is None or a_elo is None:
            y_pred[i] = ''
            continue
        if h_elo > a_elo:
            y_pred[i] = 'home_win'
        elif a_elo > h_elo:
            y_pred[i] = 'away_win'
        else:
            h_rank = _get_hist_rank(hist_ranks, home, yr)
            a_rank = _get_hist_rank(hist_ranks, away, yr)
            y_pred[i] = 'home_win' if h_rank <= a_rank else 'away_win'

    scored = y_pred != ''
    n_scored = int(scored.sum())
    acc, std, fold_accs = _fold_metrics(y_true, y_pred)
    return ModelAccuracy(
        name='ELO',
        n_scored=n_scored,
        n_total=len(df),
        accuracy=acc,
        std=std,
        fold_accs=fold_accs,
        note='era-matched ratings; never predicts draw',
    )


def evaluate_dc(
    df: pd.DataFrame,
    predict_probs: WinProbFn,
) -> ModelAccuracy:
    """Score DC on 3-class argmax of win probabilities (not scoreline)."""
    y_true = np.asarray(df['outcome'].tolist(), dtype=object)
    y_pred = np.empty(len(df), dtype=object)

    for i, row in df.iterrows():
        try:
            p_hw, p_d, p_aw = predict_probs(
                row['home_team'],
                row['away_team'],
                bool(row['is_neutral']),
                bool(row['is_group_stage']),
            )
            y_pred[i] = argmax_outcome(p_hw, p_d, p_aw)
        except Exception:
            y_pred[i] = ''

    scored = y_pred != ''
    n_scored = int(scored.sum())
    acc, std, fold_accs = _fold_metrics(y_true, y_pred)
    return ModelAccuracy(
        name='Dixon-Coles',
        n_scored=n_scored,
        n_total=len(df),
        accuracy=acc,
        std=std,
        fold_accs=fold_accs,
        note='argmax(home/draw/away) from DC win probabilities',
    )


def elo_scorable_mask(df: pd.DataFrame, root: Optional[Path] = None) -> np.ndarray:
    """True where both teams have era-matched ELO in the ratings file."""
    root = root or repo_root()
    elo_hist = _load_elo_hist(root)
    elo_countries = set(elo_hist['country'].unique())

    def _has_elo(team: str) -> bool:
        c = _eval_canon(team)
        return c in elo_countries or team in elo_countries

    mask = np.ones(len(df), dtype=bool)
    for i, row in df.iterrows():
        home = _norm_to_elo(row['home_team_norm'], int(row['year']))
        away = _norm_to_elo(row['away_team_norm'], int(row['year']))
        if not (_has_elo(home) and _has_elo(away)):
            mask[i] = False
            continue
        md = row['match_date']
        if _get_elo_at(elo_hist, home, md) is None or _get_elo_at(elo_hist, away, md) is None:
            mask[i] = False
    return mask


def compare_all(
    df: pd.DataFrame,
    dc_predict_fn: Optional[WinProbFn] = None,
    root: Optional[Path] = None,
) -> list[ModelAccuracy]:
    root = root or repo_root()
    results = [
        evaluate_lr(df),
        evaluate_elo(df, root),
    ]
    if dc_predict_fn is not None:
        results.append(evaluate_dc(df, dc_predict_fn))
    return results


def compare_elo_scorable_subset(
    df: pd.DataFrame,
    dc_predict_fn: Optional[WinProbFn] = None,
    root: Optional[Path] = None,
) -> list[ModelAccuracy]:
    """Same as compare_all but only on matches where ELO can score both teams."""
    mask = elo_scorable_mask(df, root)
    sub = df.loc[mask].reset_index(drop=True)
    sub.attrs['feature_cols'] = df.attrs['feature_cols']
    return compare_all(sub, dc_predict_fn=dc_predict_fn, root=root)


def format_comparison(results: list[ModelAccuracy], n_matches: int) -> str:
    lines = [
        'World Cup 3-class accuracy comparison (shared eval set)',
        '=' * 58,
        f'  Eval set     : {n_matches} WC matches (2006-2022, same as lr_model.ipynb)',
        f'  CV folds     : 5-fold StratifiedKFold, random_state=42',
        f'  Random guess : 0.333',
        '',
        f'{"Model":<24} {"Accuracy":>10} {"+/-":>5} {"Scored":>12}  Notes',
        '-' * 58,
    ]
    for r in results:
        scored = f'{r.n_scored}/{r.n_total}'
        lines.append(
            f'{r.name:<24} {r.accuracy:>10.3f} {r.std:>4.3f} {scored:>12}  {r.note}'
        )
    lines.append('')
    lines.append('Per-fold accuracy (same test indices for every model):')
    for r in results:
        fold_str = '  '.join(f'{a:.3f}' for a in r.fold_accs)
        lines.append(f'  {r.name:<22} {fold_str}')
    return '\n'.join(lines)


def print_wc_accuracy_comparison(
    dc_predict_fn: Optional[WinProbFn] = None,
    root: Optional[Path] = None,
) -> list[ModelAccuracy]:
    root = root or repo_root()
    df = build_wc_eval_df(root)
    results = compare_all(df, dc_predict_fn=dc_predict_fn, root=root)
    print(format_comparison(results, len(df)))

    mask = elo_scorable_mask(df, root)
    n_sub = int(mask.sum())
    if n_sub < len(df):
        sub_results = compare_elo_scorable_subset(df, dc_predict_fn=dc_predict_fn, root=root)
        print()
        print(f'Strict subset where ELO scores both teams (n={n_sub}):')
        print('=' * 58)
        lines = format_comparison(sub_results, n_sub).split('\n')
        # Skip header lines already printed above (title through random-guess)
        print('\n'.join(lines[6:]))

    return results


if __name__ == '__main__':
    print_wc_accuracy_comparison()
