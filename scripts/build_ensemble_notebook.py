"""One-off script to generate ensemble_model.ipynb."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "ensemble_model.ipynb"

def md(source: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source.splitlines(keepends=True)}

def code(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {"execution": {"iopub.execute_input": "2026-06-17T00:00:00Z"}},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }

cells = [
    md(
        "# World Cup Predictor: Episode 3 - Ensemble Model (LR + LightGBM)\n"
        "By Karl Estampador :)\n"
        "\n"
        "## What this notebook does\n"
        "\n"
        "Episode 2 trained a single Logistic Regression classifier. Episode 3 builds a **soft-voting ensemble** of Logistic Regression + LightGBM on the same seven pre-tournament difference features, then exposes the same prediction API as `lr_model.ipynb`.\n"
        "\n"
        "## How to use in `predictions.ipynb`\n"
        "\n"
        "```python\n"
        "%run ensemble_model.ipynb\n"
        "```\n"
        "\n"
        "`predict_winner`, `predict_score`, `get_elo`, and `get_lr_proba` (aliased to the ensemble) work unchanged.\n",
        "e3-intro",
    ),
    code(
        '''import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import log_loss, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier

# --- Config flags (only places this notebook may diverge from lr_model.ipynb) ---
USE_FORMER_NAMES_TABLE   = True
USE_REAL_NEUTRAL_FLAG    = False
INCLUDE_GOALS_REGRESSION = False

if INCLUDE_GOALS_REGRESSION:
    from sklearn.linear_model import PoissonRegressor

warnings.filterwarnings('ignore')
DATA = Path('data')
OUTPUTS = Path('outputs')
OUTPUTS.mkdir(exist_ok=True)

print('Imports OK')
print(f'  USE_FORMER_NAMES_TABLE   = {USE_FORMER_NAMES_TABLE}')
print(f'  USE_REAL_NEUTRAL_FLAG    = {USE_REAL_NEUTRAL_FLAG}')
print(f'  INCLUDE_GOALS_REGRESSION = {INCLUDE_GOALS_REGRESSION}')''',
        "e3-step0",
    ),
    md(
        "---\n"
        "## Step 1 - Data Cleaning & Join\n"
        "\n"
        "| File | What it contains |\n"
        "|---|---|\n"
        "| `data/historical_results/results.csv` | Match results |\n"
        "| `data/historical_results/shootouts.csv` | Penalty shootout winners |\n"
        "| `data/historical_results/former_names.csv` | Historical team-name renames (optional) |\n"
        "| `data/train.csv` | Pre-tournament team stats 2002-2022 |\n",
        "e3-s1-md",
    ),
    code(
        '''results_all = pd.read_csv(DATA / 'historical_results' / 'results.csv')
shootouts_all = pd.read_csv(DATA / 'historical_results' / 'shootouts.csv')
train_df = pd.read_csv(DATA / 'train.csv')
test_df  = pd.read_csv(DATA / 'test.csv')

results_all['year'] = pd.to_datetime(results_all['date']).dt.year
wc_mask = (
    (results_all['tournament'] == 'FIFA World Cup') &
    (results_all['year'].between(2002, 2022))
)
wc = results_all[wc_mask].copy().reset_index(drop=True)

shootouts_all['year'] = pd.to_datetime(shootouts_all['date']).dt.year
wc_shootouts = shootouts_all[shootouts_all['year'].between(2002, 2022)].copy()

print(f'WC matches 2002 - 2022: {len(wc)}')
print('Matches per tournament:')
print(wc['year'].value_counts().sort_index())
print(f'\\nShootout records in same period: {len(wc_shootouts)}')
print(f'\\ntrain.csv rows: {len(train_df)} | test.csv rows: {len(test_df)}')''',
        "e3-s1-load",
    ),
    md(
        "### 1b) Team name normalisation\n"
        "\n"
        "Always-on FIFA-label overrides, plus optional date-aware renames from `former_names.csv`.\n",
        "e3-s1b-md",
    ),
    code(
        '''RESULTS_TO_TRAIN: dict[str, str] = {
    'China': 'China PR',
}

def _build_former_name_rules() -> list[tuple[str, str, pd.Timestamp, pd.Timestamp]]:
    """former -> current, active between start_date and end_date (inclusive)."""
    if not USE_FORMER_NAMES_TABLE:
        return []
    fn = pd.read_csv(DATA / 'historical_results' / 'former_names.csv')
    rules = []
    for _, row in fn.iterrows():
        rules.append((
            row['former'],
            row['current'],
            pd.Timestamp(row['start_date']),
            pd.Timestamp(row['end_date']),
        ))
    return rules

FORMER_NAME_RULES = _build_former_name_rules()

def _normalise(name: str, year: int, match_date: str) -> str:
    """Map a results.csv team name to its train.csv equivalent."""
    if name == 'Serbia' and year == 2006:
        return 'Serbia and Montenegro'
    name = RESULTS_TO_TRAIN.get(name, name)
    if USE_FORMER_NAMES_TABLE:
        d = pd.Timestamp(match_date)
        for former, current, start, end in FORMER_NAME_RULES:
            if name == former and start <= d <= end:
                name = current
                break
    return name

wc['home_team_norm'] = wc.apply(
    lambda r: _normalise(r['home_team'], r['year'], r['date']), axis=1
)
wc['away_team_norm'] = wc.apply(
    lambda r: _normalise(r['away_team'], r['year'], r['date']), axis=1
)

# Audit remaps
remap_rows = []
for _, r in wc.iterrows():
    if r['home_team'] != r['home_team_norm']:
        remap_rows.append((r['home_team'], r['home_team_norm'], 'home'))
    if r['away_team'] != r['away_team_norm']:
        remap_rows.append((r['away_team'], r['away_team_norm'], 'away'))

if remap_rows:
    remap_df = pd.DataFrame(remap_rows, columns=['original', 'normalised', 'side'])
    summary = remap_df.groupby(['original', 'normalised']).size().reset_index(name='cells')
    print(f'Team names normalised: {len(remap_rows)} cells remapped')
    for _, row in summary.iterrows():
        print(f"  {row['original']} -> {row['normalised']}: {row['cells']} cells")
    print('\\nSample rows (home):')
    print(wc[wc['home_team_norm'] != wc['home_team']][['date', 'home_team', 'home_team_norm']].head().to_string())
    print('\\nSample rows (away):')
    print(wc[wc['away_team_norm'] != wc['away_team']][['date', 'away_team', 'away_team_norm']].head().to_string())
else:
    print('Team names normalised: 0 cells remapped')''',
        "e3-s1b",
    ),
    md(
        "### 1c - Join train.csv features for both teams\n",
        "e3-s1c-md",
    ),
    code(
        '''train_indexed = train_df.set_index(['version', 'team'])

FEATURE_COLS = [
    'goals_scored_last_4y', 'goals_received_last_4y',
    'wins_last_4y', 'losses_last_4y', 'draws_last_4y',
    'world_cup_titles_before', 'squad_total_market_value_eur',
    'fifa_rank_pre_tournament',
]

def _get_features(team: str, year: int) -> pd.Series | None:
    try:
        return train_indexed.loc[(year, team), FEATURE_COLS]
    except KeyError:
        return None

rows_before = len(wc)
home_features, away_features, keep_idx = [], [], []

for i, match in wc.iterrows():
    hf = _get_features(match['home_team_norm'], match['year'])
    af = _get_features(match['away_team_norm'], match['year'])
    if hf is None or af is None:
        if hf is None:
            print(f'  DROP (home not found): {match["year"]} {match["home_team"]} vs {match["away_team"]}')
        if af is None:
            print(f'  DROP (away not found): {match["year"]} {match["home_team"]} vs {match["away_team"]}')
        continue
    home_features.append(hf.add_prefix('home_'))
    away_features.append(af.add_prefix('away_'))
    keep_idx.append(i)

wc_joined = wc.loc[keep_idx].copy().reset_index(drop=True)
home_df = pd.DataFrame(home_features).reset_index(drop=True)
away_df = pd.DataFrame(away_features).reset_index(drop=True)
wc_joined = pd.concat([wc_joined, home_df, away_df], axis=1)

rows_dropped = rows_before - len(wc_joined)
print(f'\\nRows before join: {rows_before}')
print(f'Rows dropped (team not in train.csv): {rows_dropped}')
print(f'Rows remaining: {len(wc_joined)}')''',
        "e3-s1c",
    ),
    md("### 1d - Target variable + knockout-round draw handling\n", "e3-s1d-md"),
    code(
        '''shootout_lookup = {}
for _, row in wc_shootouts.iterrows():
    key = (row['date'], row['home_team'], row['away_team'])
    shootout_lookup[key] = row['winner']

def _outcome(row: pd.Series) -> str:
    hs, as_ = row['home_score'], row['away_score']
    if hs > as_:
        return 'home_win'
    if as_ > hs:
        return 'away_win'
    key = (row['date'], row['home_team'], row['away_team'])
    winner = shootout_lookup.get(key)
    if winner is not None:
        return 'home_win' if winner == row['home_team'] else 'away_win'
    return 'draw'

wc_joined['outcome'] = wc_joined.apply(_outcome, axis=1)

print('Class distribution (before feature drop):')
vc = wc_joined['outcome'].value_counts()
for cls in ['home_win', 'draw', 'away_win']:
    if cls in vc.index:
        print(f'  {cls}: {vc[cls]} ({vc[cls] / len(wc_joined):.1%})')
print(f'\\nTotal training matches: {len(wc_joined)}')''',
        "e3-s1d",
    ),
    md("### 1e - Feature engineering (difference features)\n", "e3-s1e-md"),
    code(
        '''df = wc_joined.copy()

df['home_gp'] = df['home_wins_last_4y'] + df['home_losses_last_4y'] + df['home_draws_last_4y']
df['away_gp'] = df['away_wins_last_4y'] + df['away_losses_last_4y'] + df['away_draws_last_4y']
df['home_gp'] = df['home_gp'].clip(lower=1)
df['away_gp'] = df['away_gp'].clip(lower=1)

df['rank_diff']           = df['home_fifa_rank_pre_tournament'] - df['away_fifa_rank_pre_tournament']
df['goals_scored_diff']   = df['home_goals_scored_last_4y']    - df['away_goals_scored_last_4y']
df['goals_conceded_diff'] = df['home_goals_received_last_4y']  - df['away_goals_received_last_4y']
df['win_rate_diff']       = (df['home_wins_last_4y'] / df['home_gp']) - \\
                            (df['away_wins_last_4y'] / df['away_gp'])

home_mv = df['home_squad_total_market_value_eur'].clip(lower=1e6)
away_mv = df['away_squad_total_market_value_eur'].clip(lower=1e6)
df['market_value_ratio'] = np.log(home_mv / away_mv)
df['titles_diff'] = df['home_world_cup_titles_before'] - df['away_world_cup_titles_before']

if USE_REAL_NEUTRAL_FLAG:
    df['is_neutral'] = (df['neutral'] == 'TRUE').astype(int)
else:
    # Match lr_model.ipynb training behaviour
    df['is_neutral'] = (df['neutral'] == 'TRUE').astype(int)

FEATURES = [
    'rank_diff', 'goals_scored_diff', 'goals_conceded_diff',
    'win_rate_diff', 'market_value_ratio', 'titles_diff', 'is_neutral',
]
TARGET = 'outcome'

n_before = len(df)
df = df.dropna(subset=FEATURES).reset_index(drop=True)
n_dropped = n_before - len(df)
print(f'Rows dropped for NaN features: {n_dropped}')
if n_dropped > 0:
    print('  (squad_total_market_value_eur was not recorded for 2002 - all 64 matches from that year are excluded)')

print(f'\\nFinal training set shape: {df[FEATURES].shape}')
print('Class distribution (training set):')
vc = df[TARGET].value_counts()
for cls in ['home_win', 'draw', 'away_win']:
    if cls in vc.index:
        print(f'  {cls}: {vc[cls]} ({vc[cls] / len(df):.1%})')
print()
df[FEATURES + [TARGET]].head()''',
        "e3-s1e",
    ),
    md(
        "---\n"
        "## Step 2 - Train the Ensemble Model\n"
        "\n"
        "Soft-voting ensemble of standardised Logistic Regression + LightGBM, validated with 5-fold stratified CV. Blend weights are tuned on CV log-loss before the final fit.\n",
        "e3-s2-md",
    ),
    code(
        '''X = df[FEATURES].values
y = df[TARGET].values

lr = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
lgbm = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    num_leaves=15,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

lr_pipe = Pipeline([('scaler', StandardScaler()), ('lr', lr)])
lgbm_pipe = Pipeline([('lgbm', lgbm)])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def _cv_metrics(estimator) -> tuple[float, float, float, float]:
    res = cross_validate(
        estimator, X, y, cv=cv,
        scoring=['accuracy', 'neg_log_loss'],
        return_train_score=False,
    )
    acc = res['test_accuracy'].mean()
    acc_std = res['test_accuracy'].std()
    ll = -res['test_neg_log_loss'].mean()
    ll_std = res['test_neg_log_loss'].std()
    return acc, acc_std, ll, ll_std

def _cv_summary(name: str, estimator) -> tuple[float, float]:
    acc, acc_std, ll, ll_std = _cv_metrics(estimator)
    print(f'\\n{name}')
    print('=' * 45)
    print(f'  Accuracy : {acc:.3f}  +/-{acc_std:.3f}')
    print(f'  Log-loss : {ll:.3f}  +/-{ll_std:.3f}')
    return acc, ll

print('5-fold stratified cross-validation results')
lr_acc, lr_ll = _cv_summary('Logistic Regression alone', lr_pipe)
lgbm_acc, lgbm_ll = _cv_summary('LightGBM alone', lgbm_pipe)

ensemble_eq = VotingClassifier(
    estimators=[('lr', lr_pipe), ('lgbm', lgbm_pipe)],
    voting='soft',
    weights=[1, 1],
)
ens_eq_acc, ens_eq_ll = _cv_summary('Ensemble (equal weights 50/50)', ensemble_eq)''',
        "e3-s2c",
    ),
    md("### 2g) Blend weight search\n", "e3-s2g-md"),
    code(
        '''print('Blend weight search (minimise CV log-loss)')
print('Fixed LightGBM weight = 1.0; searching Logistic Regression weight.')
print('=' * 55)

W_LGBM = 1.0
weight_grid = np.linspace(0.0, 2.0, 41)
search_rows = []
best_w_lr = 1.0
best_acc, best_ll = ens_eq_acc, ens_eq_ll

for w_lr in weight_grid:
    candidate = VotingClassifier(
        estimators=[('lr', lr_pipe), ('lgbm', lgbm_pipe)],
        voting='soft',
        weights=[w_lr, W_LGBM],
    )
    acc, _, ll, _ = _cv_metrics(candidate)
    search_rows.append({'w_lr': w_lr, 'accuracy': acc, 'log_loss': ll})

    if ll < best_ll - 1e-9 or (abs(ll - best_ll) <= 1e-9 and acc > best_acc):
        best_w_lr, best_acc, best_ll = w_lr, acc, ll

search_df = pd.DataFrame(search_rows)
pct_lr = 100 * best_w_lr / (best_w_lr + W_LGBM) if (best_w_lr + W_LGBM) > 0 else 0.0
pct_lgbm = 100 * W_LGBM / (best_w_lr + W_LGBM) if (best_w_lr + W_LGBM) > 0 else 100.0

print(f'Best weights: LR={best_w_lr:.2f}, LGBM={W_LGBM:.2f}')
print(f'  Implied blend: {pct_lr:.0f}% Logistic Regression / {pct_lgbm:.0f}% LightGBM')
print(f'  CV accuracy : {best_acc:.3f}')
print(f'  CV log-loss : {best_ll:.3f}')

# Show a few neighbouring weights for narration
near_best = search_df.iloc[(search_df['log_loss'] - best_ll).abs().argsort()[:5]]
print('\\nTop 5 weight settings by log-loss:')
print(f'{"w_lr":>6} {"LR %":>6} {"LGBM %":>8} {"Accuracy":>10} {"Log-loss":>10}')
print('-' * 46)
for _, row in near_best.sort_values('log_loss').iterrows():
    w = row['w_lr']
    lr_pct = 100 * w / (w + W_LGBM) if (w + W_LGBM) > 0 else 0
    lgbm_pct = 100 - lr_pct
    print(f'{w:6.2f} {lr_pct:6.0f} {lgbm_pct:8.0f} {row["accuracy"]:10.3f} {row["log_loss"]:10.3f}')

BEST_WEIGHTS = [best_w_lr, W_LGBM]
ensemble_pipe = VotingClassifier(
    estimators=[('lr', lr_pipe), ('lgbm', lgbm_pipe)],
    voting='soft',
    weights=BEST_WEIGHTS,
)
ens_acc, ens_ll = best_acc, best_ll

print('\\nModel comparison (CV means)')
print('-' * 58)
print(f'{"Model":<32} {"Accuracy":>10} {"Log-loss":>10}')
print('-' * 58)
for label, acc, ll in [
    ('Logistic Regression', lr_acc, lr_ll),
    ('LightGBM', lgbm_acc, lgbm_ll),
    ('Ensemble (50/50)', ens_eq_acc, ens_eq_ll),
    (f'Ensemble (tuned {pct_lr:.0f}/{pct_lgbm:.0f})', ens_acc, ens_ll),
]:
    print(f'{label:<32} {acc:>10.3f} {ll:>10.3f}')
print()
print('Note: random-guess baseline accuracy for 3 classes: 0.33')
print(f'Training set size: {len(X)} matches')''',
        "e3-s2g",
    ),
    code(
        '''ensemble_pipe.fit(X, y)
CLASS_ORDER = list(ensemble_pipe.classes_)
print('Class order:', CLASS_ORDER)

y_pred = ensemble_pipe.predict(X)
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y, y_pred,
    display_labels=CLASS_ORDER,
    ax=ax,
    colorbar=False,
)
ax.set_title('Confusion matrix (in-sample)')
plt.tight_layout()
plt.show()''',
        "e3-s2e",
    ),
    code(
        '''CLASS_LABELS = {
    'home_win': 'Home Win',
    'draw': 'Draw',
    'away_win': 'Away Win',
}

lr_oof_proba = cross_val_predict(lr_pipe, X, y, cv=cv, method='predict_proba')
lgbm_oof_proba = cross_val_predict(lgbm_pipe, X, y, cv=cv, method='predict_proba')

lr_oof_cls = np.array(CLASS_ORDER)[np.argmax(lr_oof_proba, axis=1)]
lgbm_oof_cls = np.array(CLASS_ORDER)[np.argmax(lgbm_oof_proba, axis=1)]
agree = lr_oof_cls == lgbm_oof_cls
disagree_rate = 1.0 - agree.mean()

print('Model agreement (out-of-fold CV predictions)')
print('=' * 45)
print(f'  Same predicted class : {agree.mean():.1%}')
print(f'  Disagreement rate    : {disagree_rate:.1%}')

# Largest gap between each model's confidence in its own top class
lr_top_conf = lr_oof_proba.max(axis=1)
lgbm_top_conf = lgbm_oof_proba.max(axis=1)
conf_gap = np.abs(lr_top_conf - lgbm_top_conf)

disagree_idx = np.where(~agree)[0]
if len(disagree_idx) == 0:
    print('\\nNo disagreements in CV - models always picked the same class.')
else:
    ranked = sorted(disagree_idx, key=lambda i: conf_gap[i], reverse=True)[:5]
    print(f'\\nTop {min(5, len(ranked))} disagreements (largest confidence gap):')
    for i in ranked:
        row = df.iloc[i]
        lr_cls = lr_oof_cls[i]
        lgbm_cls = lgbm_oof_cls[i]
        print(f"\\n{row['date']}  {row['home_team']} vs {row['away_team']}")
        print(f"  Logistic Regression: {CLASS_LABELS[lr_cls]} ({lr_top_conf[i]:.0%})")
        print(f"  LightGBM:            {CLASS_LABELS[lgbm_cls]} ({lgbm_top_conf[i]:.0%})")
        print(f"  Actual outcome:      {CLASS_LABELS.get(row['outcome'], row['outcome'])}")''',
        "e3-s2f",
    ),
    md("---\n## Step 3 - Feature Importance Visualisation\n", "e3-s3-md"),
    code(
        '''feature_labels = [
    'FIFA rank diff\\n(home - away)',
    'Goals scored diff\\n(last 4y)',
    'Goals conceded diff\\n(last 4y)',
    'Win rate diff',
    'Market value ratio\\n(log)',
    'WC titles diff',
    'Neutral venue',
]

lr_fitted = ensemble_pipe.named_estimators_['lr'].named_steps['lr']
lgbm_fitted = ensemble_pipe.named_estimators_['lgbm'].named_steps['lgbm']

coef_df = pd.DataFrame(lr_fitted.coef_, index=CLASS_ORDER, columns=feature_labels)
lgbm_imp = pd.Series(lgbm_fitted.feature_importances_, index=feature_labels)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Ensemble feature importance: LR coefficients vs LightGBM gain', fontsize=13, fontweight='bold', y=1.02)

class_titles = {'home_win': 'Home Win', 'draw': 'Draw', 'away_win': 'Away Win'}
ax_lr = axes[0]
y_pos = np.arange(len(feature_labels))
width = 0.25
for j, cls in enumerate(['home_win', 'draw', 'away_win']):
    if cls not in coef_df.index:
        continue
    offset = (j - 1) * width
    vals = coef_df.loc[cls]
    colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in vals]
    ax_lr.barh(y_pos + offset, vals, height=width * 0.9, label=class_titles[cls], color=colors, alpha=0.85)
ax_lr.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax_lr.set_yticks(y_pos)
ax_lr.set_yticklabels(feature_labels)
ax_lr.set_xlabel('Coefficient (standardised)')
ax_lr.set_title('Logistic Regression')
ax_lr.legend(loc='lower right', fontsize=8)
ax_lr.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

ax_lgbm = axes[1]
colors_lgbm = plt.cm.Blues(np.linspace(0.4, 0.9, len(feature_labels)))[::-1]
ax_lgbm.barh(feature_labels, lgbm_imp.values, color=colors_lgbm, edgecolor='white', height=0.6)
ax_lgbm.set_xlabel('Gain importance')
ax_lgbm.set_title('LightGBM')
ax_lgbm.invert_yaxis()

plt.tight_layout()
out_path = OUTPUTS / 'ensemble_feature_importance.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Chart saved to {out_path}')
plt.show()''',
        "e3-s3",
    ),
    md(
        "---\n"
        "## Step 4 - Prediction Functions\n"
        "\n"
        "Same public API as `lr_model.ipynb` for `%run` swap in `predictions.ipynb`.\n",
        "e3-s4-md",
    ),
    code(
        '''elo_raw = pd.read_csv(DATA / 'elo_ratings_wc2026.csv')
elo_2026 = (
    elo_raw[elo_raw['snapshot_date'] == '2026-05-27']
    .copy()
    .reset_index(drop=True)
)

ELO_ALIASES: dict[str, str] = {
    'USA':              'United States',
    'IR Iran':          'Iran',
    "Côte d'Ivoire":    'Ivory Coast',
    'Cabo Verde':       'Cape Verde',
    'DR Congo':         'DR Congo',
    'United States':    'United States',
    'Iran':             'Iran',
    'Ivory Coast':      'Ivory Coast',
    'Cape Verde':       'Cape Verde',
    'Czech Republic':   'Czechia',
    'Cura\\u00e7ao':     'Cura\\u00e7ao',
}

_elo_lookup: dict[str, float] = dict(zip(elo_2026['country'], elo_2026['rating']))

def _canonical_elo(name: str) -> str:
    return ELO_ALIASES.get(name, name)

def get_elo(team_name: str) -> float:
    """Pre-tournament ELO rating. Used by predictions.ipynb for standings only."""
    canon = _canonical_elo(team_name)
    if canon in _elo_lookup:
        return _elo_lookup[canon]
    if team_name in _elo_lookup:
        return _elo_lookup[team_name]
    raise KeyError(f'No Elo rating found for "{team_name}" (canonical: "{canon}")')

def _predict_winner_elo(home: str, away: str) -> tuple[str, str, float, float]:
    """ELO fallback when features are unavailable for a team."""
    home_elo = get_elo(home)
    away_elo = get_elo(away)
    if home_elo >= away_elo:
        return home, away, home_elo, away_elo
    return away, home, away_elo, home_elo

print(f'ELO lookup ready ({len(_elo_lookup)} teams)')

TEST_NAME_MAP: dict[str, str] = {
    'USA':              'United States',
    'IR Iran':          'Iran',
    "Cote d'Ivoire":    'Ivory Coast',
    "Côte d'Ivoire":    'Ivory Coast',
    'Cabo Verde':       'Cape Verde',
    'Czechia':          'Czech Republic',
    'Cura\\u00e7ao':     'Cura?o',
}

_test_features: dict[str, dict] = {}
for _, row in test_df.iterrows():
    _test_features[row['team']] = row

for teams_name, test_name in TEST_NAME_MAP.items():
    if test_name in _test_features:
        _test_features[teams_name] = _test_features[test_name]

def _resolve(name: str) -> str:
    return TEST_NAME_MAP.get(name, name)

def _build_features(home: str, away: str) -> np.ndarray | None:
    """Compute the difference feature vector for a 2026 match."""
    h = _resolve(home)
    a = _resolve(away)
    hr = _test_features.get(h)
    ar = _test_features.get(a)
    if hr is None:
        print(f'WARNING: "{home}" (resolved: "{h}") not found in test.csv - falling back to ELO')
        return None
    if ar is None:
        print(f'WARNING: "{away}" (resolved: "{a}") not found in test.csv - falling back to ELO')
        return None

    home_gp = (hr['wins_last_4y'] + hr['losses_last_4y'] + hr['draws_last_4y']) or 1
    away_gp = (ar['wins_last_4y'] + ar['losses_last_4y'] + ar['draws_last_4y']) or 1
    home_mv = max(float(hr['squad_total_market_value_eur']), 1e6)
    away_mv = max(float(ar['squad_total_market_value_eur']), 1e6)

    return np.array([
        hr['fifa_rank_pre_tournament']    - ar['fifa_rank_pre_tournament'],
        hr['goals_scored_last_4y']        - ar['goals_scored_last_4y'],
        hr['goals_received_last_4y']      - ar['goals_received_last_4y'],
        (hr['wins_last_4y'] / home_gp)    - (ar['wins_last_4y'] / away_gp),
        np.log(home_mv / away_mv),
        hr['world_cup_titles_before']     - ar['world_cup_titles_before'],
        1,
    ], dtype=float)

print(f'Test features loaded for {len(_test_features)} name variants')''',
        "e3-s4-elo",
    ),
    code(
        '''def get_ensemble_proba(home: str, away: str) -> dict[str, float]:
    """Return {'home_win': p, 'draw': p, 'away_win': p} from ensemble."""
    feat = _build_features(home, away)
    if feat is None:
        return {'home_win': float('nan'), 'draw': float('nan'), 'away_win': float('nan')}
    proba = ensemble_pipe.predict_proba(feat.reshape(1, -1))[0]
    return {cls: float(p) for cls, p in zip(CLASS_ORDER, proba)}


def predict_knockout_winner(home: str, away: str) -> tuple[str, str, float, float]:
    """No draws allowed. Compare renormalized P(home_win) vs P(away_win)."""
    feat = _build_features(home, away)
    if feat is None:
        return _predict_winner_elo(home, away)
    proba = get_ensemble_proba(home, away)
    p_home = proba.get('home_win', 0.0)
    p_away = proba.get('away_win', 0.0)
    total = p_home + p_away
    if total <= 0:
        total = 1.0
    p_h, p_a = p_home / total, p_away / total
    if p_h >= p_a:
        return home, away, p_h, p_a
    return away, home, p_a, p_h


def predict_group_match(home: str, away: str) -> tuple[int, int, str | None]:
    """Draws allowed. If P(draw) is highest, return 1-1 draw."""
    proba = get_ensemble_proba(home, away)
    p_draw = proba.get('draw', 0.0)
    p_home = proba.get('home_win', 0.0)
    p_away = proba.get('away_win', 0.0)
    if p_draw > p_home and p_draw > p_away:
        return 1, 1, None
    winner, loser, wp, lp = predict_knockout_winner(home, away)
    goals_w = 2 if (wp - lp) > 0.30 else 1
    if winner == home:
        return goals_w, 0, winner
    return 0, goals_w, winner


def predict_winner_ensemble(home: str, away: str):
    return predict_knockout_winner(home, away)


def predict_score_ensemble(home: str, away: str):
    hg, ag, _ = predict_group_match(home, away)
    return hg, ag


predict_winner = predict_winner_ensemble
predict_score  = predict_score_ensemble
get_lr_proba   = get_ensemble_proba

print('Ensemble prediction functions ready.')
print('predict_group_match, predict_knockout_winner, get_lr_proba - ready.')
print('predict_winner and predict_score aliased to ensemble versions.')''',
        "e3-s4-predict",
    ),
    md("---\n## Step 5 - Integration Check\n", "e3-s5-md"),
    code(
        '''test_pairs = [
    ('Mexico', 'South Africa'),
    ('South Korea', 'Czechia'),
    ('Canada', 'Bosnia and Herzegovina'),
    ('United States', 'Paraguay'),
]

print('-- Group match (draws allowed) --')
print(f'{"Match":<36} {"Result":<8} {"Winner/Draw":<22} {"Proba (hw / d / aw)"}')
print('-' * 85)
for home, away in test_pairs:
    hg, ag, winner = predict_group_match(home, away)
    proba = get_ensemble_proba(home, away)
    hw = proba.get('home_win', float('nan'))
    d  = proba.get('draw', float('nan'))
    aw = proba.get('away_win', float('nan'))
    label = winner if winner else 'Draw'
    match_label = f'{home} vs {away}'
    print(f'{match_label:<36} {hg}-{ag}     {label:<22} hw={hw:.2f}  d={d:.2f}  aw={aw:.2f}')

print()
print('-- Knockout (no draws) --')
print(f'{"Match":<36} {"Winner":<22} {"Win Prob"}')
print('-' * 65)
for home, away in test_pairs:
    winner, loser, wp, lp = predict_knockout_winner(home, away)
    match_label = f'{home} vs {away}'
    print(f'{match_label:<36} {winner:<22} {wp:.3f}')''',
        "e3-s5",
    ),
    md("---\n## Step 7 - Narration-Ready Takeaway\n", "e3-s7-md"),
    code(
        '''print(
    f"Logistic regression alone: {lr_acc:.1%} accuracy, log-loss {lr_ll:.3f}.\\n"
    f"LightGBM alone: {lgbm_acc:.1%} accuracy, log-loss {lgbm_ll:.3f}.\\n"
    f"Ensemble (50/50): {ens_eq_acc:.1%} accuracy, log-loss {ens_eq_ll:.3f}.\\n"
    f"Ensemble (tuned weights): {ens_acc:.1%} accuracy, log-loss {ens_ll:.3f}.\\n"
    f"Best blend: {pct_lr:.0f}% LR / {pct_lgbm:.0f}% LightGBM."
)

best_single_acc = max(lr_acc, lgbm_acc)
best_single_ll = min(lr_ll, lgbm_ll)
beats_acc = ens_acc > best_single_acc + 1e-9
beats_ll = ens_ll < best_single_ll - 1e-9
beats_equal = ens_ll < ens_eq_ll - 1e-9 or ens_acc > ens_eq_acc + 1e-9

if beats_acc and beats_ll:
    verdict = "The tuned ensemble beat both individual models on accuracy and log-loss."
elif beats_acc or beats_ll:
    parts = []
    if beats_acc:
        parts.append('accuracy')
    if beats_ll:
        parts.append('log-loss')
    verdict = (
        f"The tuned ensemble improved on {' and '.join(parts)} "
        "but only matched the better single model on the other metric."
    )
else:
    verdict = (
        "Even with tuned weights, the ensemble did not beat both models outright - "
        "on this small dataset the best single model may be hard to improve on."
    )
if beats_equal:
    verdict += f" Weight tuning did help versus the 50/50 blend (log-loss {ens_eq_ll:.3f} -> {ens_ll:.3f})."
elif abs(ens_ll - ens_eq_ll) < 1e-9 and abs(ens_acc - ens_eq_acc) < 1e-9:
    verdict += " Tuned weights landed on the same 50/50 blend."
print(verdict)''',
        "e3-s7",
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.14.3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
print(f'Wrote {OUT} ({len(cells)} cells)')
