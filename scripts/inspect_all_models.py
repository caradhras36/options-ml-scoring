#!/usr/bin/env python3
"""
For N random trades, show SHAP explanations across ALL 5 models side-by-side.

Usage:
  python3 scripts/inspect_all_models.py <training_csv> [--n 5]
"""

import argparse
import numpy as np
import pandas as pd
import shap
import xgboost as xgb


FEATURE_COLS = [
    'dte', 'delta', 'gamma', 'theta', 'vega', 'iv', 'ivr', 'ivp',
    'moneyness', 'annualized_return', 'credit_received', 'max_loss',
    'rsi14', 'macd_signal', 'bb_position', 'resistance_score',
    'days_to_earnings', 'earnings_in_window', 'post_earnings',
]

OUTCOME_CLASSES = ['breakeven', 'full_win', 'loss', 'partial_win']


def format_value(v, feature):
    if pd.isna(v):
        return 'NaN'
    if feature in ('gamma',):
        return f'{v:.4f}'
    if feature in ('credit_received', 'max_loss', 'expected_value'):
        return f'${v:,.0f}'
    if feature in ('ivr', 'ivp', 'rsi14', 'bb_position', 'resistance_score'):
        return f'{v:.2f}' if abs(v) < 10 else f'{v:.0f}'
    if feature in ('days_to_earnings', 'dte'):
        return f'{v:.0f}' if pd.notna(v) else '—'
    if isinstance(v, float) and abs(v) < 10:
        return f'{v:.3f}'
    return str(v)


def shap_top(explainer, X, top_n=5, class_idx=None):
    """Return top N features by |shap| with their values and signs."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[class_idx or 0]
    if sv.ndim == 3:  # multi-class
        sv = sv[:, :, class_idx or 1]
    row = sv[0]
    pairs = sorted(zip(FEATURE_COLS, X[0], row), key=lambda x: -abs(x[2]))
    return pairs[:top_n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f'Loading {args.data}...')
    df = pd.read_csv(args.data)
    df['days_to_earnings'] = df['days_to_earnings'].fillna(999)
    df['earnings_in_window'] = df['earnings_in_window'].fillna(0).astype(int)
    df['post_earnings'] = df['post_earnings'].fillna(0).astype(int)
    df = df.dropna(subset=FEATURE_COLS)

    # Sample
    half = args.n // 2
    calls = df[df['option_type'] == 'CALL'].sample(half, random_state=args.seed)
    puts = df[df['option_type'] == 'PUT'].sample(args.n - half, random_state=args.seed)
    sample = pd.concat([calls, puts]).reset_index(drop=True)

    # Load all models
    print('Loading models...')
    hit50 = xgb.XGBClassifier(); hit50.load_model('./models/hit50_model.json')
    maxprof = xgb.XGBRegressor(); maxprof.load_model('./models/maxprofit_model.json')
    days50 = xgb.XGBRegressor(); days50.load_model('./models/days50_model.json')
    ev = xgb.XGBRegressor(); ev.load_model('./models/ev_model.json')
    outcome = xgb.XGBClassifier(); outcome.load_model('./models/outcome_model.json')

    exp_hit = shap.TreeExplainer(hit50)
    exp_mp = shap.TreeExplainer(maxprof)
    exp_d50 = shap.TreeExplainer(days50)
    exp_ev = shap.TreeExplainer(ev)
    exp_out = shap.TreeExplainer(outcome)

    for i in range(len(sample)):
        r = sample.iloc[i]
        X = sample.iloc[[i]][FEATURE_COLS].values

        # Predictions
        p_hit = float(hit50.predict_proba(X)[0][1])
        p_mp = float(maxprof.predict(X)[0])
        p_d50 = float(days50.predict(X)[0])
        p_ev = float(ev.predict(X)[0])
        p_out_probs = outcome.predict_proba(X)[0]
        p_out_idx = int(np.argmax(p_out_probs))
        p_out = OUTCOME_CLASSES[p_out_idx]

        print()
        print('=' * 100)
        print(f'Sample {i + 1}: {r["symbol"]} ${r["strike"]:.0f} {r["option_type"]} '
              f'{r.get("expiry", "")}  ({r["dte"]:.0f} DTE)')
        print(f'  Underlying: ${r["underlying"]:.2f}  Moneyness: {r["moneyness"]:.3f}  Delta: {r["delta"]:+.3f}')
        print(f'  ACTUAL: hit={r["hit_50pct"]}  outcome={r["outcome"]}  '
              f'max_profit={r["max_profit_pct"]:+.2%}  EV=${r["expected_value"]:.0f}  '
              f'days_to_50={r.get("days_to_50pct", "—")}')
        print()
        print(f'  MODEL PREDICTIONS:')
        print(f'    1. hit_50pct:      {p_hit:>6.1%}')
        print(f'    2. max_profit_pct: {p_mp:>+6.2%}')
        print(f'    3. days_to_50pct:  {p_d50:>6.1f} days  (only meaningful if hit)')
        print(f'    4. expected_value: ${p_ev:>8,.0f}')
        print(f'    5. outcome:        {p_out}  (probs: ' +
              ', '.join(f'{c}={p:.0%}' for c, p in zip(OUTCOME_CLASSES, p_out_probs)) + ')')
        print()

        # SHAP for each model — top 5 each
        models = [
            ('hit_50pct', exp_hit, None, 'prob'),
            ('max_profit_pct', exp_mp, None, 'target'),
            ('days_to_50pct', exp_d50, None, 'days'),
            ('expected_value', exp_ev, None, '$'),
            (f'outcome ({p_out})', exp_out, p_out_idx, 'prob'),
        ]

        for name, expl, class_idx, unit in models:
            top = shap_top(expl, X, top_n=5, class_idx=class_idx)
            print(f'  [{name}] top 5 factors:')
            for feat, val, sv in top:
                val_str = format_value(val, feat)
                sign = '+' if sv > 0 else ''
                icon = '✅' if sv > 0 else '❌'
                bar = '█' * min(int(abs(sv) * 8), 20)
                print(f'    {feat:20s} {val_str:>10s}  {icon} {sign}{sv:>6.3f}  {bar}')
            print()


if __name__ == '__main__':
    main()
