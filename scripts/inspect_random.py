#!/usr/bin/env python3
"""
Pick N random trades from training data, compute SHAP, show feature values + impacts.

Usage:
  python3 scripts/inspect_random.py <training_csv> [--n 10] [--model models/hit50_model.json]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='Training CSV')
    parser.add_argument('--n', type=int, default=10, help='Number of random samples')
    parser.add_argument('--model', default='./models/hit50_model.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Load
    print(f'Loading {args.data}...')
    df = pd.read_csv(args.data)
    df['days_to_earnings'] = df['days_to_earnings'].fillna(999)
    df['earnings_in_window'] = df['earnings_in_window'].fillna(0).astype(int)
    df['post_earnings'] = df['post_earnings'].fillna(0).astype(int)
    df = df.dropna(subset=FEATURE_COLS)

    # Sample - mix of CC and CSP
    calls = df[df['option_type'] == 'CALL'].sample(args.n // 2, random_state=args.seed)
    puts = df[df['option_type'] == 'PUT'].sample(args.n - len(calls), random_state=args.seed)
    sample = pd.concat([calls, puts]).reset_index(drop=True)

    # Load model + SHAP
    print(f'Loading {args.model}...')
    model = xgb.XGBClassifier()
    model.load_model(args.model)
    explainer = shap.TreeExplainer(model)

    X = sample[FEATURE_COLS].values
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    predictions = model.predict_proba(X)[:, 1]

    # Print each sample
    for i in range(len(sample)):
        r = sample.iloc[i]
        print()
        print('=' * 90)
        print(f'Sample {i + 1}: {r["symbol"]} ${r["strike"]:.0f} {r["option_type"]} '
              f'{r.get("expiry", "")}  ({r["dte"]:.0f} DTE)')
        print(f'  Underlying: ${r["underlying"]:.2f}  Moneyness: {r["moneyness"]:.3f}  '
              f'Delta: {r["delta"]:+.3f}')
        print(f'  → ACTUAL: hit_50pct={r["hit_50pct"]}  outcome={r["outcome"]}  '
              f'max_profit={r["max_profit_pct"]:+.2%}  EV=${r["expected_value"]:.0f}')
        print(f'  → MODEL : P(hit_50pct)={predictions[i]:.1%}')
        print()
        print(f'  {"Feature":22s} {"Value":>10s}    {"SHAP":>8s}   Impact')
        print(f'  {"-" * 22} {"-" * 10}    {"-" * 8}   {"-" * 30}')

        # Sort features by |shap|
        feat_shap = list(zip(FEATURE_COLS, X[i], shap_values[i]))
        feat_shap.sort(key=lambda x: -abs(x[2]))

        for feat, val, sv in feat_shap:
            val_str = format_value(val, feat)
            sign = '+' if sv > 0 else ''
            bar = ('█' * min(int(abs(sv) * 10), 30))
            color_indicator = '✅' if sv > 0 else '❌' if sv < 0 else ' '
            print(f'  {feat:22s} {val_str:>10s}    {sign}{sv:>7.3f}   {color_indicator} {bar}')


if __name__ == '__main__':
    main()
