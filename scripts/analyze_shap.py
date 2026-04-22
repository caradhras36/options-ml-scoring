#!/usr/bin/env python3
"""
SHAP analysis of trained XGBoost models.

Shows:
1. Feature importance (global)
2. SHAP summary plot (direction + magnitude)
3. Dependence plots (how each feature affects prediction)
4. Feature interactions

Usage:
  python3 scripts/analyze_shap.py --data training.csv --model ./models/hit50_model.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


FEATURE_COLS = [
    'dte', 'delta', 'gamma', 'theta', 'vega', 'iv', 'ivr', 'ivp',
    'moneyness', 'annualized_return', 'credit_received', 'max_loss',
    'rsi14', 'macd_signal', 'bb_position', 'resistance_score',
    'days_to_earnings', 'earnings_in_window', 'post_earnings',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Training CSV')
    parser.add_argument('--model', required=True, help='XGBoost model JSON')
    parser.add_argument('--out', default='./shap_output', help='Output directory for plots')
    parser.add_argument('--sample', type=int, default=10000, help='Sample size for SHAP (SHAP is slow)')
    parser.add_argument('--classifier', action='store_true', help='Model is a classifier')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f'Loading {args.data}...')
    df = pd.read_csv(args.data)

    # Fill NaN
    df['days_to_earnings'] = df['days_to_earnings'].fillna(999)
    df['earnings_in_window'] = df['earnings_in_window'].fillna(0).astype(int)
    df['post_earnings'] = df['post_earnings'].fillna(0).astype(int)

    feature_df = df[FEATURE_COLS].dropna()
    print(f'After dropna: {len(feature_df):,}')

    # Sample
    sample = feature_df.sample(min(args.sample, len(feature_df)), random_state=42)
    print(f'SHAP sample size: {len(sample):,}')

    # Load model
    print(f'Loading model {args.model}...')
    if args.classifier:
        model = xgb.XGBClassifier()
    else:
        model = xgb.XGBRegressor()
    model.load_model(args.model)

    # SHAP values
    print('Computing SHAP values...')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # For multi-class, shap_values is a list. Pick first class for viz.
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # ── Plot 1: Summary (beeswarm) ──
    print('Generating summary plot...')
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, sample, feature_names=FEATURE_COLS, show=False, max_display=19)
    plt.tight_layout()
    plt.savefig(out_dir / 'shap_summary.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_dir}/shap_summary.png')

    # ── Plot 2: Bar (mean |SHAP|) ──
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, sample, feature_names=FEATURE_COLS, plot_type='bar', show=False, max_display=19)
    plt.tight_layout()
    plt.savefig(out_dir / 'shap_bar.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_dir}/shap_bar.png')

    # ── Plot 3: Dependence plots for top 6 features ──
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features = np.argsort(mean_abs_shap)[::-1][:6]

    print(f'\nTop 6 features by |SHAP|:')
    for i, idx in enumerate(top_features):
        print(f'  {i+1}. {FEATURE_COLS[idx]}: {mean_abs_shap[idx]:.4f}')

    print('\nGenerating dependence plots...')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, idx in enumerate(top_features):
        ax = axes[i // 3, i % 3]
        shap.dependence_plot(
            idx, shap_values, sample, feature_names=FEATURE_COLS,
            ax=ax, show=False, interaction_index=None,
        )
        ax.set_title(FEATURE_COLS[idx], fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'shap_dependence.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_dir}/shap_dependence.png')

    # ── Summary stats ──
    print(f'\n{"="*60}')
    print('SHAP Summary Statistics')
    print('='*60)
    print(f'\nMean absolute SHAP value per feature (higher = more influence):')
    for idx in np.argsort(mean_abs_shap)[::-1]:
        sign = '(bullish)' if np.mean(shap_values[:, idx]) > 0 else '(bearish)'
        print(f'  {FEATURE_COLS[idx]:25s} {mean_abs_shap[idx]:.4f}  avg={np.mean(shap_values[:, idx]):+.4f} {sign}')

    # Earnings effect specifically
    print(f'\n{"="*60}')
    print('Earnings Impact Analysis')
    print('='*60)

    ew_idx = FEATURE_COLS.index('earnings_in_window')
    pe_idx = FEATURE_COLS.index('post_earnings')
    dte_idx = FEATURE_COLS.index('days_to_earnings')

    # Split by earnings_in_window
    no_earnings = sample['earnings_in_window'] == 0
    with_earnings = sample['earnings_in_window'] == 1

    if with_earnings.sum() > 100:
        print(f'\nearnings_in_window=0 (n={no_earnings.sum():,}): avg SHAP = {shap_values[no_earnings.values, ew_idx].mean():+.4f}')
        print(f'earnings_in_window=1 (n={with_earnings.sum():,}): avg SHAP = {shap_values[with_earnings.values, ew_idx].mean():+.4f}')
        print(f'Effect of having earnings in window: {shap_values[with_earnings.values, ew_idx].mean() - shap_values[no_earnings.values, ew_idx].mean():+.4f}')


if __name__ == '__main__':
    main()
