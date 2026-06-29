"""Validate Section 8b fix: DC accuracy with vs without draw inflation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson as poisson_dist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.compare_wc_accuracy import CV, argmax_outcome, build_wc_eval_df

MAX_GOALS = 10
WC_STR = 'FIFA World Cup'
EXCLUDED = {'Friendly'}


def _load_competitive() -> pd.DataFrame:
    data = ROOT / 'data' / 'historical'
    results = pd.read_csv(data / 'results.csv')
    results['date'] = pd.to_datetime(results['date'])
    results['year'] = results['date'].dt.year
    results = results.dropna(subset=['home_score', 'away_score'])
    results['home_score'] = results['home_score'].astype(int)
    results['away_score'] = results['away_score'].astype(int)
    return results[~results['tournament'].isin(EXCLUDED)].copy()


def _matrix_outcomes(matrix: np.ndarray) -> tuple[float, float, float]:
    p_hw = float(np.sum(np.tril(matrix, -1)))
    p_d = float(np.sum(np.diag(matrix)))
    p_aw = float(np.sum(np.triu(matrix, 1)))
    return p_hw, p_d, p_aw


def _dc_prob_matrix(lam_home: float, lam_away: float, rho: float) -> np.ndarray:
    hpm = poisson_dist.pmf(np.arange(MAX_GOALS + 1), lam_home)
    apm = poisson_dist.pmf(np.arange(MAX_GOALS + 1), lam_away)
    mat = np.outer(hpm, apm)
    mat[0, 0] = max(1e-15, mat[0, 0] * (1.0 - lam_home * lam_away * rho))
    mat[0, 1] = max(1e-15, mat[0, 1] * (1.0 + lam_home * rho))
    mat[1, 0] = max(1e-15, mat[1, 0] * (1.0 + lam_away * rho))
    mat[1, 1] = max(1e-15, mat[1, 1] * (1.0 - rho))
    return mat / mat.sum()


def _fit_dc(df_train, xi, anchor, ref_team=None, seed=42):
    teams = sorted(set(df_train['home_team'].tolist() + df_train['away_team'].tolist()))
    n = len(teams)
    t2i = {t: i for i, t in enumerate(teams)}
    if ref_team is None or ref_team not in t2i:
        ref_team = teams[0]
    ref_idx = t2i[ref_team]
    home_idx = np.array([t2i[t] for t in df_train['home_team']])
    away_idx = np.array([t2i[t] for t in df_train['away_team']])
    hg = df_train['home_score'].values.astype(float)
    ag = df_train['away_score'].values.astype(float)
    neutral = df_train['neutral'].values.astype(bool)
    delta_t = (anchor - df_train['date']).dt.days.values.astype(float)
    weights = np.exp(-xi * delta_t / 365.0)
    rng = np.random.default_rng(seed)
    x0 = np.zeros(2 * n + 1)
    x0[: n - 1] = rng.uniform(-0.1, 0.1, n - 1)
    x0[n - 1 : 2 * n - 1] = rng.uniform(-0.1, 0.1, n)
    x0[2 * n - 1] = 0.2
    x0[2 * n] = -0.1
    bounds = ([(-3.0, 3.0)] * (n - 1) + [(-3.0, 3.0)] * n + [(-1.0, 2.0)] + [(-0.5, 0.5)])
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)

    def neg_ll(params):
        a_raw = params[: n - 1]
        beta = params[n - 1 : 2 * n - 1]
        gamma = params[2 * n - 1]
        rho = params[2 * n]
        alpha = np.insert(a_raw, ref_idx, 0.0)
        h_adv = np.where(neutral, 0.0, gamma)
        loglh = alpha[home_idx] + beta[away_idx] + h_adv
        logla = alpha[away_idx] + beta[home_idx]
        lh, la = np.exp(loglh), np.exp(logla)
        tau_00 = 1.0 - lh[m00] * la[m00] * rho
        tau_01 = 1.0 + lh[m01] * rho
        tau_10 = 1.0 + la[m10] * rho
        tau_11_val = 1.0 - rho
        if np.any(tau_00 <= 0) or np.any(tau_01 <= 0) or np.any(tau_10 <= 0) or tau_11_val <= 0:
            return 1e6
        log_ph = np.where(hg > 0, hg * loglh, 0.0) - lh - gammaln(hg + 1)
        log_pa = np.where(ag > 0, ag * logla, 0.0) - la - gammaln(ag + 1)
        log_tau = np.zeros(len(hg))
        log_tau[m00] = np.log(tau_00)
        log_tau[m01] = np.log(tau_01)
        log_tau[m10] = np.log(tau_10)
        if m11.any():
            log_tau[m11] = np.log(tau_11_val)
        ll = weights * (log_tau + log_ph + log_pa)
        result = ll.sum()
        return -result if np.isfinite(result) else 1e10

    res = minimize(neg_ll, x0, method='L-BFGS-B', bounds=bounds)
    a_raw = res.x[: n - 1]
    beta = res.x[n - 1 : 2 * n - 1]
    gamma = float(res.x[2 * n - 1])
    rho = float(res.x[2 * n])
    alpha = np.insert(a_raw, ref_idx, 0.0)
    nll_val = float(neg_ll(res.x))
    return dict(zip(teams, alpha)), dict(zip(teams, beta)), gamma, rho, nll_val


def main() -> None:
    cache_path = ROOT / 'outputs' / 'dc_grid_cache.json'
    with open(cache_path) as f:
        cached = json.load(f)
    best_sy = cached['best_start_year']
    best_xi = cached['best_xi']

    df = _load_competitive()
    df_final = df[df['year'] >= best_sy].copy()
    anchor = pd.Timestamp('2026-06-01')
    teams = sorted(set(df_final['home_team'].tolist() + df_final['away_team'].tolist()))
    ref = teams[0]

    print(f'Fitting DC (start_year={best_sy}, xi={best_xi}, single seed for speed)...')
    alpha_d, beta_d, gamma, rho, nll = _fit_dc(
        df_final, xi=best_xi, anchor=anchor, ref_team=ref, seed=42
    )
    print(f'  NLL={nll:.2f}, gamma={gamma:.4f}, rho={rho:.4f}')

    a_fb = float(np.mean(list(alpha_d.values())))
    b_fb = float(np.mean(list(beta_d.values())))

    # Draw inflation factor (same logic as notebook Section 5)
    shootouts = pd.read_csv(ROOT / 'data' / 'historical' / 'shootouts.csv')
    shootouts['date'] = pd.to_datetime(shootouts['date'])
    df_wc = df[df['tournament'] == WC_STR].copy()
    shootout_keys = set(
        zip(
            shootouts['date'].dt.strftime('%Y-%m-%d'),
            shootouts['home_team'],
            shootouts['away_team'],
        )
    )
    df_wc['_key'] = list(
        zip(df_wc['date'].dt.strftime('%Y-%m-%d'), df_wc['home_team'], df_wc['away_team'])
    )
    df_wc_ns = df_wc[~df_wc['_key'].isin(shootout_keys)]
    empirical_draw = float((df_wc_ns['home_score'] == df_wc_ns['away_score']).mean())
    model_draws = []
    for _, row in df_wc_ns.iterrows():
        ah = alpha_d.get(row['home_team'], a_fb)
        bh = beta_d.get(row['home_team'], b_fb)
        aa = alpha_d.get(row['away_team'], a_fb)
        ba = beta_d.get(row['away_team'], b_fb)
        g = 0.0 if row.get('neutral', True) else gamma
        lh = np.exp(ah + ba + g)
        la = np.exp(aa + bh)
        _, pd_, _ = _matrix_outcomes(_dc_prob_matrix(lh, la, rho))
        model_draws.append(pd_)
    draw_inflation = float(np.clip(empirical_draw / np.mean(model_draws), 1.0, 1.5))
    print(f'  Draw inflation factor: {draw_inflation:.4f}')

    def _team_params(team: str) -> tuple[float, float]:
        return alpha_d.get(team, a_fb), beta_d.get(team, b_fb)

    def _eval_acc(inflate_group: bool, inflation_factor: float) -> tuple[float, float]:
        def predict_wp(home, away, neutral, *, inflate: bool):
            ah, bh = _team_params(home)
            aa, ba = _team_params(away)
            g = 0.0 if neutral else gamma
            lh = np.exp(ah + ba + g)
            la = np.exp(aa + bh)
            mat = _dc_prob_matrix(lh, la, rho)
            if inflate:
                di = np.arange(mat.shape[0])
                mat = mat.copy()
                mat[di, di] *= inflation_factor
                mat /= mat.sum()
            return _matrix_outcomes(mat)

        fold_accs = []
        y_pred = np.empty(len(wc_eval), dtype=object)
        for i, row in wc_eval.iterrows():
            use_infl = inflate_group and bool(row['is_group_stage'])
            p_hw, p_d, p_aw = predict_wp(
                row['home_team'], row['away_team'], bool(row['is_neutral']),
                inflate=use_infl,
            )
            y_pred[i] = argmax_outcome(p_hw, p_d, p_aw)
        for _, test_idx in CV.split(np.zeros(len(y_true)), y_true):
            fold_accs.append(float((y_pred[test_idx] == y_true[test_idx]).mean()))
        return float(np.mean(fold_accs)), float(np.std(fold_accs))

    wc_eval = build_wc_eval_df()
    y_true = np.asarray(wc_eval['outcome'].tolist(), dtype=object)

    no_infl_acc, no_infl_std = _eval_acc(inflate_group=False, inflation_factor=draw_inflation)
    actual_old_acc, actual_old_std = _eval_acc(
        inflate_group=True, inflation_factor=draw_inflation
    )
    # Sensitivity: if inflation > 1 in full notebook fit, this shows expected direction
    forced_old_acc, forced_old_std = _eval_acc(inflate_group=True, inflation_factor=1.5)

    print('\nSection 8b validation (320 WC matches):')
    print(f'  No inflation (fixed benchmark):     {no_infl_acc:.3f} +/- {no_infl_std:.3f}')
    print(f'  Old path (actual inflation={draw_inflation:.3f}): '
          f'{actual_old_acc:.3f} +/- {actual_old_std:.3f}')
    if draw_inflation > 1.001:
        print(f'  Delta (actual inflation):             {no_infl_acc - actual_old_acc:+.3f}')
    print(f'  Old path (forced inflation=1.5):      {forced_old_acc:.3f} +/- {forced_old_std:.3f}')
    print(f'  Delta (forced 1.5x):                {no_infl_acc - forced_old_acc:+.3f}')

    if draw_inflation > 1.001 and no_infl_acc <= actual_old_acc:
        print('WARNING: expected improvement when actual inflation > 1')
        sys.exit(1)
    if forced_old_acc >= no_infl_acc:
        print('WARNING: forced 1.5x inflation should hurt argmax accuracy')
        sys.exit(1)
    print('OK: benchmark excludes draw inflation; inflation hurts argmax when active')


if __name__ == '__main__':
    main()
