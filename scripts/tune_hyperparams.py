#!/usr/bin/env python3
"""
Hyperparameter tuning for Options ML models.

Runs grid search on hit_50pct classifier (primary model),
then applies best params to all 5 models.

Usage:
  python3 scripts/tune_hyperparams.py "data/training.csv"
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer
from itertools import product

FEATURE_COLUMNS = [
    'dte', 'delta', 'gamma', 'theta', 'vega', 'iv', 'ivr', 'ivp',
    'moneyness', 'annualized_return', 'credit_received', 'max_loss',
    'rsi14', 'macd_signal', 'bb_position', 'resistance_score',
    'days_to_earnings', 'earnings_in_window', 'post_earnings',
    # New features
    'credit_pct', 'iv_vs_ticker_avg', 'return_5d', 'return_10d', 'return_20d',
    'rv_20d', 'rv_iv_ratio', 'iv_skew_proxy', 'dte_bucket',
    'theta_vega_ratio', 'delta_moneyness',
]

TARGET = 'hit_50pct'


def load_data(path, sample=None):
    """Load training data."""
    df = pd.read_csv(path)
    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=42)
    # Only use columns that exist
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available].copy()
    y = df[TARGET].astype(int)
    return X, y, df


def run_grid_search(X, y):
    """Grid search over XGBoost hyperparameters."""

    param_grid = {
        'n_estimators': [200, 400, 600],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [1.0, 2.0, 5.0],
    }

    # Phase 1: Coarse search on key params (n_estimators, max_depth, learning_rate)
    print("Phase 1: Coarse search (n_estimators, max_depth, learning_rate)")
    print("=" * 60)

    neg_ratio = (y == 0).sum() / (y == 1).sum()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    best_score = 0
    best_params = {}
    results = []

    coarse_combos = list(product(
        param_grid['n_estimators'],
        param_grid['max_depth'],
        param_grid['learning_rate'],
    ))

    print(f"  {len(coarse_combos)} combinations to test")

    for i, (n_est, depth, lr) in enumerate(coarse_combos):
        model = xgb.XGBClassifier(
            n_estimators=n_est,
            max_depth=depth,
            learning_rate=lr,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            scale_pos_weight=neg_ratio,
            random_state=42,
            eval_metric='logloss',
            verbosity=0,
        )

        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
        mean_score = scores.mean()
        results.append({
            'n_estimators': n_est, 'max_depth': depth, 'learning_rate': lr,
            'auc_mean': mean_score, 'auc_std': scores.std(),
        })

        if mean_score > best_score:
            best_score = mean_score
            best_params = {'n_estimators': n_est, 'max_depth': depth, 'learning_rate': lr}

        if (i + 1) % 9 == 0 or i == len(coarse_combos) - 1:
            print(f"  [{i+1}/{len(coarse_combos)}] Best so far: AUC={best_score:.4f} {best_params}")

    print(f"\n  Phase 1 best: AUC={best_score:.4f}")
    print(f"  Params: {best_params}")

    # Phase 2: Fine tune regularization + sampling around best
    print(f"\nPhase 2: Fine-tune (subsample, colsample, min_child_weight, reg)")
    print("=" * 60)

    fine_combos = list(product(
        param_grid['subsample'],
        param_grid['colsample_bytree'],
        param_grid['min_child_weight'],
        param_grid['reg_alpha'],
        param_grid['reg_lambda'],
    ))

    # Sample a subset to keep it manageable
    if len(fine_combos) > 50:
        np.random.seed(42)
        indices = np.random.choice(len(fine_combos), 50, replace=False)
        fine_combos = [fine_combos[i] for i in indices]

    print(f"  {len(fine_combos)} combinations to test")

    for i, (subsample, colsample, mcw, alpha, lam) in enumerate(fine_combos):
        model = xgb.XGBClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            subsample=subsample,
            colsample_bytree=colsample,
            min_child_weight=mcw,
            reg_alpha=alpha,
            reg_lambda=lam,
            scale_pos_weight=neg_ratio,
            random_state=42,
            eval_metric='logloss',
            verbosity=0,
        )

        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
        mean_score = scores.mean()

        if mean_score > best_score:
            best_score = mean_score
            best_params.update({
                'subsample': subsample, 'colsample_bytree': colsample,
                'min_child_weight': mcw, 'reg_alpha': alpha, 'reg_lambda': lam,
            })

        if (i + 1) % 10 == 0 or i == len(fine_combos) - 1:
            print(f"  [{i+1}/{len(fine_combos)}] Best so far: AUC={best_score:.4f}")

    print(f"\n  Phase 2 best: AUC={best_score:.4f}")
    print(f"  Final params: {best_params}")

    # Fill in defaults for any missing params
    final_params = {
        'n_estimators': best_params.get('n_estimators', 200),
        'max_depth': best_params.get('max_depth', 6),
        'learning_rate': best_params.get('learning_rate', 0.1),
        'subsample': best_params.get('subsample', 0.8),
        'colsample_bytree': best_params.get('colsample_bytree', 0.8),
        'min_child_weight': best_params.get('min_child_weight', 1),
        'reg_alpha': best_params.get('reg_alpha', 0),
        'reg_lambda': best_params.get('reg_lambda', 1.0),
    }

    return final_params, best_score, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='Training CSV path')
    parser.add_argument('--sample', type=int, default=200000, help='Sample size for speed (default 200K)')
    args = parser.parse_args()

    print(f"Loading data: {args.data}")
    X, y, df = load_data(args.data, sample=args.sample)
    print(f"Data: {len(X):,} rows, {len(FEATURE_COLUMNS)} features")
    print(f"Target: {y.mean()*100:.1f}% positive")
    print()

    start = time.time()
    best_params, best_score, results = run_grid_search(X, y)
    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"TUNING COMPLETE ({elapsed/60:.1f} min)")
    print(f"{'='*60}")
    print(f"Best AUC: {best_score:.4f}")
    print(f"Current AUC: 0.9024 (V7 clean default params)")
    print(f"Improvement: {(best_score - 0.9024)*100:+.2f}pp")
    print(f"\nBest params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    print(f"\nTo retrain with these params, update train_model.py:")
    print(f"  n_estimators={best_params['n_estimators']},")
    print(f"  max_depth={best_params['max_depth']},")
    print(f"  learning_rate={best_params['learning_rate']},")
    print(f"  subsample={best_params['subsample']},")
    print(f"  colsample_bytree={best_params['colsample_bytree']},")
    print(f"  min_child_weight={best_params['min_child_weight']},")
    print(f"  reg_alpha={best_params['reg_alpha']},")
    print(f"  reg_lambda={best_params['reg_lambda']},")


if __name__ == '__main__':
    main()
