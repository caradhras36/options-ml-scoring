#!/usr/bin/env python3
"""
Train XGBoost models on the generated training data.

Trains 3 models:
  1. hit_50pct classifier — "Will this trade reach 50% profit?"
  2. max_profit_pct regressor — "What % of max profit at expiry?"
  3. days_to_50pct regressor — "How many days to 50% profit?"

Saves models as JSON for loading in both Python and TypeScript.

Usage:
  python3 scripts/train_model.py "data/training.csv"
  python3 scripts/train_model.py training.csv --out ./models/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix,
)


FEATURE_COLS = [
    'dte', 'delta', 'gamma', 'theta', 'vega', 'iv', 'ivr', 'ivp',
    'moneyness', 'annualized_return', 'credit_received', 'max_loss',
    'rsi14', 'macd_signal', 'bb_position', 'resistance_score',
    'days_to_earnings', 'earnings_in_window', 'post_earnings',
]


def load_data(path: str) -> pd.DataFrame:
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def train_hit50_model(X_train, X_test, y_train, y_test, out_dir: Path):
    """Binary classifier: will the trade hit 50% profit?"""
    print('\n' + '='*60)
    print('MODEL 1: hit_50pct (Binary Classifier, class-balanced)')
    print('='*60)

    # Class imbalance: ~88% positive. Balance by giving negatives more weight.
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f'  Class balance: pos={n_pos:,} neg={n_neg:,}  scale_pos_weight={scale_pos_weight:.3f}')

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f'\nAccuracy:  {accuracy_score(y_test, y_pred):.4f}')
    print(f'Precision: {precision_score(y_test, y_pred):.4f}')
    print(f'Recall:    {recall_score(y_test, y_pred):.4f}')
    print(f'F1 Score:  {f1_score(y_test, y_pred):.4f}')
    print(f'AUC-ROC:   {roc_auc_score(y_test, y_prob):.4f}')

    print(f'\nConfusion Matrix:')
    cm = confusion_matrix(y_test, y_pred)
    print(f'  TN={cm[0][0]:,}  FP={cm[0][1]:,}')
    print(f'  FN={cm[1][0]:,}  TP={cm[1][1]:,}')

    # Feature importance
    print(f'\nTop Feature Importances:')
    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1])[:10]:
        print(f'  {feat:25s} {imp:.4f}')

    # Save model
    model_path = out_dir / 'hit50_model.json'
    model.save_model(str(model_path))
    print(f'\nModel saved: {model_path}')

    return model, importances


def train_maxprofit_model(X_train, X_test, y_train, y_test, out_dir: Path):
    """Regressor: what % of max profit at expiry?"""
    print('\n' + '='*60)
    print('MODEL 2: max_profit_pct (Regressor)')
    print('='*60)

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)

    print(f'\nMAE:  {mean_absolute_error(y_test, y_pred):.4f}')
    print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}')
    print(f'R²:   {r2_score(y_test, y_pred):.4f}')

    # Feature importance
    print(f'\nTop Feature Importances:')
    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1])[:10]:
        print(f'  {feat:25s} {imp:.4f}')

    model_path = out_dir / 'maxprofit_model.json'
    model.save_model(str(model_path))
    print(f'\nModel saved: {model_path}')

    return model, importances


def train_days50_model(X_train, X_test, y_train, y_test, out_dir: Path):
    """Regressor: how many days to reach 50% profit?"""
    print('\n' + '='*60)
    print('MODEL 3: days_to_50pct (Regressor — only trades that hit 50%)')
    print('='*60)

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)

    print(f'\nMAE:  {mean_absolute_error(y_test, y_pred):.1f} days')
    print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.1f} days')
    print(f'R²:   {r2_score(y_test, y_pred):.4f}')

    # Feature importance
    print(f'\nTop Feature Importances:')
    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1])[:10]:
        print(f'  {feat:25s} {imp:.4f}')

    model_path = out_dir / 'days50_model.json'
    model.save_model(str(model_path))
    print(f'\nModel saved: {model_path}')

    return model, importances


def train_ev_model(X_train, X_test, y_train, y_test, out_dir: Path):
    """Regressor: expected value in dollars — actionable profit estimate."""
    print('\n' + '='*60)
    print('MODEL 4: expected_value ($ regressor — actionable dollar profit)')
    print('='*60)

    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)

    print(f'\nMAE:  ${mean_absolute_error(y_test, y_pred):.0f}')
    print(f'RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):.0f}')
    print(f'R²:   {r2_score(y_test, y_pred):.4f}')

    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    print(f'\nTop Feature Importances:')
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1])[:10]:
        print(f'  {feat:25s} {imp:.4f}')

    model_path = out_dir / 'ev_model.json'
    model.save_model(str(model_path))
    print(f'\nModel saved: {model_path}')

    return model, importances


def train_outcome_model(X_train, X_test, y_train, y_test, out_dir: Path, classes: list):
    """Multi-class classifier: full_win / partial_win / breakeven / loss"""
    print('\n' + '='*60)
    print('MODEL 5: outcome (Multi-class, class-balanced)')
    print('='*60)

    # Compute per-sample weights for class balance (boost minority classes)
    from collections import Counter
    counts = Counter(y_train.tolist())
    total = sum(counts.values())
    n_classes = len(classes)
    # Inverse-frequency weighting: weight = total / (n_classes × class_count)
    class_weight = {c: total / (n_classes * counts[c]) for c in counts}
    sample_weight = np.array([class_weight[y] for y in y_train])
    print(f'  Class weights:')
    for i, cls in enumerate(classes):
        print(f'    {cls}: n={counts.get(i, 0):,}  weight={class_weight.get(i, 0):.3f}')

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective='multi:softprob', num_class=len(classes),
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight,
              eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)

    print(f'\nAccuracy: {accuracy_score(y_test, y_pred):.4f}')
    cm = confusion_matrix(y_test, y_pred)
    print(f'\nConfusion Matrix (rows=true, cols=predicted):')
    print(f'{"":15s}' + ''.join(f'{c:>15s}' for c in classes))
    for i, c in enumerate(classes):
        print(f'{c:15s}' + ''.join(f'{cm[i][j]:>15,}' for j in range(len(classes))))

    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    print(f'\nTop Feature Importances:')
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1])[:10]:
        print(f'  {feat:25s} {imp:.4f}')

    model_path = out_dir / 'outcome_model.json'
    model.save_model(str(model_path))
    print(f'\nModel saved: {model_path}')

    return model, importances, classes


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost option scoring models')
    parser.add_argument('data', help='Training data CSV or Parquet')
    parser.add_argument('--out', default='./models', help='Output directory for models')
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f'Loading {args.data}...')
    df = load_data(args.data)
    print(f'Total rows: {len(df):,}')
    print(f'Columns: {list(df.columns)}')

    # Drop rows with missing features
    response_cols = ['hit_50pct', 'max_profit_pct', 'days_to_50pct', 'expected_value', 'outcome']
    available_responses = [c for c in response_cols if c in df.columns]
    # Fill NaN in optional features (earnings for ETFs) with sensible defaults before dropping
    # days_to_earnings: leave NaN — XGBoost handles missing values natively by learning
    # a default direction per split. Sentinel 999 caused the model to interpret it as
    # a literal "very far" value and apply consistent SHAP shifts.
    for col in ['earnings_in_window', 'post_earnings']:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    feature_df = df[FEATURE_COLS + available_responses].dropna(subset=FEATURE_COLS)
    print(f'After dropping NaN features: {len(feature_df):,}')

    X = feature_df[FEATURE_COLS].values
    print(f'Feature matrix: {X.shape}')

    # ── Model 1: hit_50pct ──
    y_hit = feature_df['hit_50pct'].astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(X, y_hit, test_size=args.test_size, random_state=42)
    hit50_model, hit50_imp = train_hit50_model(X_train, X_test, y_train, y_test, out_dir)

    # ── Model 2: max_profit_pct ──
    y_mp = feature_df['max_profit_pct'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y_mp, test_size=args.test_size, random_state=42)
    mp_model, mp_imp = train_maxprofit_model(X_train, X_test, y_train, y_test, out_dir)

    # ── Model 3: days_to_50pct (only rows that hit 50%) ──
    hit_mask = feature_df['hit_50pct'] == True
    if hit_mask.sum() > 100:
        X_hit = feature_df.loc[hit_mask, FEATURE_COLS].values
        y_days = feature_df.loc[hit_mask, 'days_to_50pct'].values
        X_train, X_test, y_train, y_test = train_test_split(X_hit, y_days, test_size=args.test_size, random_state=42)
        days_model, days_imp = train_days50_model(X_train, X_test, y_train, y_test, out_dir)
    else:
        print('\nNot enough hit_50pct=True rows for days model')

    # ── Model 4: expected_value (dollar amount) ──
    if 'expected_value' in feature_df.columns:
        ev = feature_df['expected_value'].values
        ev_lo, ev_hi = np.percentile(ev, [0.5, 99.5])
        y_ev = np.clip(ev, ev_lo, ev_hi)
        X_train, X_test, y_train, y_test = train_test_split(X, y_ev, test_size=args.test_size, random_state=42)
        ev_model, ev_imp = train_ev_model(X_train, X_test, y_train, y_test, out_dir)

    # ── Model 5: outcome (multi-class full_win/partial_win/breakeven/loss) ──
    outcome_classes = []
    if 'outcome' in feature_df.columns:
        outcome_classes = sorted(feature_df['outcome'].unique().tolist())
        class_to_int = {c: i for i, c in enumerate(outcome_classes)}
        y_out = feature_df['outcome'].map(class_to_int).values
        X_train, X_test, y_train, y_test = train_test_split(X, y_out, test_size=args.test_size, random_state=42)
        outcome_model, outcome_imp, _ = train_outcome_model(X_train, X_test, y_train, y_test, out_dir, outcome_classes)

    # Save feature columns for inference
    meta = {
        'feature_columns': FEATURE_COLS,
        'training_rows': len(feature_df),
        'tickers': df['symbol'].unique().tolist() if 'symbol' in df.columns else [],
        'date_range': [
            str(df['date'].min()) if 'date' in df.columns else '',
            str(df['date'].max()) if 'date' in df.columns else '',
        ],
    }
    with open(out_dir / 'model_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\n{"="*60}')
    print(f'All models saved to {out_dir}/')
    print(f'Files:')
    for p in sorted(out_dir.iterdir()):
        print(f'  {p.name} ({p.stat().st_size / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
