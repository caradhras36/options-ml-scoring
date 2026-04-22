#!/usr/bin/env python3
"""
Backtest the ML model on held-out recent data.

For each candidate in the test period:
  1. Query model — would it recommend this trade? (hit_50pct >= threshold)
  2. If yes, simulate: open at mid, close at hit_50 OR expiry (whichever first)
  3. Track P&L

Compares vs naive baseline: "accept all candidates".

Usage:
  python3 scripts/backtest.py <training_csv> \\
    --threshold 0.85 \\
    --start 2023-07-01 \\
    --end 2023-12-28
"""

import argparse
import numpy as np
import pandas as pd
import xgboost as xgb


FEATURE_COLS = [
    'dte', 'delta', 'gamma', 'theta', 'vega', 'iv', 'ivr', 'ivp',
    'moneyness', 'annualized_return', 'credit_received', 'max_loss',
    'rsi14', 'macd_signal', 'bb_position', 'resistance_score',
    'days_to_earnings', 'earnings_in_window', 'post_earnings',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument('--start', default='2023-07-01')
    parser.add_argument('--end', default='2023-12-28')
    parser.add_argument('--model', default='./models/hit50_model.json')
    parser.add_argument('--close-at-50', action='store_true',
                        help='Close at 50% profit (otherwise hold to expiry)')
    args = parser.parse_args()

    print(f'Loading {args.data}...')
    df = pd.read_csv(args.data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=FEATURE_COLS)

    # Filter to test period
    test = df[(df['date'] >= args.start) & (df['date'] <= args.end)].copy()
    print(f'Test period: {args.start} → {args.end}  ({len(test):,} candidates)')

    # Load model
    print(f'Loading {args.model}...')
    model = xgb.XGBClassifier()
    model.load_model(args.model)

    # Predict
    X = test[FEATURE_COLS].values
    test['p_hit'] = model.predict_proba(X)[:, 1]

    # Filter by threshold
    picks = test[test['p_hit'] >= args.threshold].copy()
    print(f'Model picks (p_hit ≥ {args.threshold}): {len(picks):,}')

    if len(picks) == 0:
        print('No picks — try lower threshold.')
        return

    # Compute P&L
    # If --close-at-50: assume we close at 50% of credit when hit_50pct is True, else hold to expiry and realize max_profit_pct × credit
    # Otherwise: always hold to expiry (max_profit_pct × credit)
    def compute_pnl(row):
        credit = row['credit_received']
        if args.close_at_50 and row['hit_50pct']:
            return 0.5 * credit  # captured 50% of max profit
        # Else hold to expiry — max_profit_pct is the realized return as fraction of credit
        return row['max_profit_pct'] * credit

    picks['pnl'] = picks.apply(compute_pnl, axis=1)

    # Compare with "accept all" baseline on the same candidates pool
    test['pnl_naive'] = test.apply(compute_pnl, axis=1) if args.close_at_50 else test['max_profit_pct'] * test['credit_received']

    # Results
    print()
    print('=' * 70)
    print('MODEL RESULTS')
    print('=' * 70)
    print(f'  N trades:          {len(picks):,}')
    print(f'  Hit rate (actual): {picks["hit_50pct"].mean():.1%}')
    print(f'  Win rate:          {(picks["pnl"] > 0).mean():.1%}')
    print(f'  Total P&L:         ${picks["pnl"].sum():,.0f}')
    print(f'  Avg per trade:     ${picks["pnl"].mean():.0f}')
    print(f'  Median P&L:        ${picks["pnl"].median():.0f}')
    print(f'  Best trade:        ${picks["pnl"].max():,.0f}')
    print(f'  Worst trade:       ${picks["pnl"].min():,.0f}')
    print(f'  Std dev:           ${picks["pnl"].std():.0f}')

    # Outcome breakdown
    if 'outcome' in picks.columns:
        print(f'\n  Outcome distribution:')
        for cls, pct in (picks['outcome'].value_counts(normalize=True) * 100).items():
            count = picks['outcome'].value_counts()[cls]
            pnl_cls = picks[picks['outcome'] == cls]['pnl'].sum()
            print(f'    {cls:15s} {count:>6,} ({pct:>5.1f}%)  P&L=${pnl_cls:>+12,.0f}')

    print()
    print('=' * 70)
    print('BASELINE (accept all candidates in same pool)')
    print('=' * 70)
    print(f'  N trades:          {len(test):,}')
    print(f'  Hit rate:          {test["hit_50pct"].mean():.1%}')
    print(f'  Total P&L:         ${test["pnl_naive"].sum():,.0f}')
    print(f'  Avg per trade:     ${test["pnl_naive"].mean():.0f}')

    print()
    print('=' * 70)
    print('COMPARISON')
    print('=' * 70)
    model_per = picks["pnl"].sum() / len(picks)
    naive_per = test["pnl_naive"].sum() / len(test)
    lift = (model_per / naive_per - 1) * 100 if naive_per > 0 else 0
    print(f'  Model avg/trade:   ${model_per:.0f}')
    print(f'  Naive avg/trade:   ${naive_per:.0f}')
    print(f'  Lift:              {lift:+.1f}%')

    # Capital efficiency: model uses fewer trades, higher win rate
    model_rate = picks["pnl"].mean() / picks["credit_received"].mean() * 100
    naive_rate = test["pnl_naive"].mean() / test["credit_received"].mean() * 100
    print(f'  Model return/credit: {model_rate:.1f}%')
    print(f'  Naive return/credit: {naive_rate:.1f}%')

    # Monthly P&L
    picks['month'] = picks['date'].dt.to_period('M')
    monthly = picks.groupby('month')['pnl'].agg(['sum', 'count']).reset_index()
    print(f'\n  Monthly P&L:')
    for _, row in monthly.iterrows():
        print(f'    {row["month"]}  {row["count"]:>5} trades  ${row["sum"]:>+12,.0f}')


if __name__ == '__main__':
    main()
