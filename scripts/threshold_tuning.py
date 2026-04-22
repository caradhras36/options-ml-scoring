#!/usr/bin/env python3
"""
Threshold tuning for the hit_50pct classifier.

Sweeps probability thresholds from 0.5 to 0.99 and reports:
  - Precision, Recall, F1 at each threshold
  - Number of trades selected (volume)
  - Actual win rate of selected trades
  - Expected portfolio P&L (using the EV model)
  - Recommends optimal threshold for different strategies

Usage:
  python3 scripts/threshold_tuning.py \
    --data "data/training.csv" \
    --model ./models/hit50_model.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve


FEATURE_COLS = [
    'dte', 'delta', 'gamma', 'theta', 'vega', 'iv', 'ivr', 'ivp',
    'moneyness', 'annualized_return', 'credit_received', 'max_loss',
    'rsi14', 'macd_signal', 'bb_position', 'resistance_score',
    'days_to_earnings', 'earnings_in_window', 'post_earnings',
]


def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray, ev: np.ndarray,
                     mp: np.ndarray, thresholds: list) -> pd.DataFrame:
    """Compute metrics at each threshold."""
    results = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        selected_mask = y_pred == 1
        n_selected = selected_mask.sum()

        if n_selected == 0:
            results.append({'threshold': t, 'n_selected': 0})
            continue

        selected_true = y_true[selected_mask]
        win_rate = selected_true.mean()
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Expected portfolio performance
        avg_ev = ev[selected_mask].mean()
        total_ev = ev[selected_mask].sum()
        avg_mp = mp[selected_mask].mean()

        # Hypothetical: assume you can do N trades; what's your avg return?
        results.append({
            'threshold': t,
            'n_selected': n_selected,
            'pct_of_all': n_selected / len(y_true) * 100,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'actual_win_rate': win_rate,
            'avg_ev_per_trade': avg_ev,
            'total_ev_if_all_done': total_ev,
            'avg_max_profit_pct': avg_mp,
        })
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', default='./models/hit50_model.json')
    parser.add_argument('--ev-model', default='./models/ev_model.json')
    parser.add_argument('--out', default='./threshold_analysis.csv')
    args = parser.parse_args()

    print(f'Loading {args.data}...')
    df = pd.read_csv(args.data)

    # Fill NaN
    df['days_to_earnings'] = df['days_to_earnings'].fillna(999)
    df['earnings_in_window'] = df['earnings_in_window'].fillna(0).astype(int)
    df['post_earnings'] = df['post_earnings'].fillna(0).astype(int)

    feature_df = df[FEATURE_COLS + ['hit_50pct', 'max_profit_pct', 'expected_value']].dropna(subset=FEATURE_COLS)
    print(f'Rows: {len(feature_df):,}')

    X = feature_df[FEATURE_COLS].values
    y = feature_df['hit_50pct'].astype(int).values
    mp = feature_df['max_profit_pct'].values
    ev = feature_df['expected_value'].values

    # Same split as training (random_state=42)
    _, X_test, _, y_test, _, mp_test, _, ev_test = train_test_split(
        X, y, mp, ev, test_size=0.2, random_state=42
    )

    # Load model
    print(f'Loading {args.model}...')
    model = xgb.XGBClassifier()
    model.load_model(args.model)

    print('Predicting probabilities...')
    y_prob = model.predict_proba(X_test)[:, 1]

    # Sweep thresholds
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99]
    results = sweep_thresholds(y_test, y_prob, ev_test, mp_test, thresholds)

    # Print results
    print()
    print('='*120)
    print(f'{"Threshold":>10}  {"N Selected":>12}  {"% Pool":>8}  {"Precision":>10}  {"Recall":>8}  {"F1":>6}  {"Win Rate":>10}  {"Avg EV":>10}  {"Total EV":>14}')
    print('-'*120)
    for _, row in results.iterrows():
        if row['n_selected'] == 0:
            print(f'{row["threshold"]:>10.2f}  {"0":>12}  (no trades selected)')
            continue
        print(
            f'{row["threshold"]:>10.2f}  '
            f'{int(row["n_selected"]):>12,}  '
            f'{row["pct_of_all"]:>7.1f}%  '
            f'{row["precision"]:>10.4f}  '
            f'{row["recall"]:>8.4f}  '
            f'{row["f1"]:>6.4f}  '
            f'{row["actual_win_rate"]:>10.4f}  '
            f'${row["avg_ev_per_trade"]:>9.0f}  '
            f'${row["total_ev_if_all_done"]:>13,.0f}'
        )

    # Find optimal thresholds
    print()
    print('='*120)
    print('RECOMMENDED THRESHOLDS BY STRATEGY')
    print('='*120)

    valid = results[results['n_selected'] > 0].copy()

    # Max F1
    max_f1 = valid.loc[valid['f1'].idxmax()]
    print(f'\n• Max F1 (balanced): threshold = {max_f1["threshold"]:.2f}')
    print(f'  Precision={max_f1["precision"]:.3f}  Recall={max_f1["recall"]:.3f}  F1={max_f1["f1"]:.3f}')
    print(f'  Selects {int(max_f1["n_selected"]):,} trades ({max_f1["pct_of_all"]:.1f}% of pool)')

    # Max Precision (but require min recall)
    high_prec = valid[valid['recall'] >= 0.5].copy()
    if not high_prec.empty:
        best_prec = high_prec.loc[high_prec['precision'].idxmax()]
        print(f'\n• Max Precision (with ≥50% recall): threshold = {best_prec["threshold"]:.2f}')
        print(f'  Precision={best_prec["precision"]:.3f}  Recall={best_prec["recall"]:.3f}')
        print(f'  Win rate on selected: {best_prec["actual_win_rate"]:.3f}')
        print(f'  Selects {int(best_prec["n_selected"]):,} trades')

    # Max avg EV per trade
    best_ev = valid.loc[valid['avg_ev_per_trade'].idxmax()]
    print(f'\n• Max Avg EV per Trade: threshold = {best_ev["threshold"]:.2f}')
    print(f'  Avg EV = ${best_ev["avg_ev_per_trade"]:.0f}/trade  (win rate = {best_ev["actual_win_rate"]:.3f})')
    print(f'  Selects {int(best_ev["n_selected"]):,} trades')

    # Max total EV (maximize volume × avg EV)
    best_total = valid.loc[valid['total_ev_if_all_done'].idxmax()]
    print(f'\n• Max Total EV (volume × EV): threshold = {best_total["threshold"]:.2f}')
    print(f'  Total EV = ${best_total["total_ev_if_all_done"]:,.0f}  ({int(best_total["n_selected"]):,} trades)')

    # Save
    results.to_csv(args.out, index=False)
    print(f'\nResults saved: {args.out}')


if __name__ == '__main__':
    main()
