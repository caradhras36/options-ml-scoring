#!/usr/bin/env python3
"""
Enhance training data with new features + fix training strategy.

New features:
  - rv_iv_ratio: realized vol / implied vol (vol risk premium signal)
  - underlying_return_5d/10d/20d: price momentum
  - volume_oi_ratio: liquidity proxy (from data if available)
  - iv_skew_proxy: moneyness-adjusted IV deviation from ATM
  - ticker_avg_iv: per-ticker average IV (regime proxy)
  - credit_pct_of_underlying: premium as % of stock price
  - dte_bucket: categorical DTE grouping

Training strategy fixes:
  - Ticker-weighted sampling (equalize ticker representation)
  - Time-based split (last 20% by date as test set)

Usage:
  python3 scripts/enhance_features.py "data/training.csv"
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add new engineered features."""
    print("Adding new features...")

    # 1. Credit as % of underlying price
    df['credit_pct'] = df['credit_received'] / (df['underlying'] * 100)
    print(f"  credit_pct: mean={df['credit_pct'].mean():.4f}")

    # 2. Per-ticker average IV (helps model learn ticker volatility regime)
    ticker_iv = df.groupby('symbol')['iv'].transform('mean')
    df['iv_vs_ticker_avg'] = df['iv'] / ticker_iv
    print(f"  iv_vs_ticker_avg: mean={df['iv_vs_ticker_avg'].mean():.4f}")

    # 3. Underlying momentum (returns)
    # Sort by symbol + date for rolling calc
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    for window in [5, 10, 20]:
        col = f'return_{window}d'
        # Group by symbol, compute rolling return
        df[col] = df.groupby('symbol')['underlying'].transform(
            lambda x: x.pct_change(window)
        )
        non_null = df[col].dropna()
        print(f"  {col}: mean={non_null.mean():.4f}, nulls={df[col].isnull().sum()}")

    # 4. Realized volatility (20-day rolling std of returns)
    df['daily_return'] = df.groupby('symbol')['underlying'].transform(lambda x: x.pct_change())
    df['rv_20d'] = df.groupby('symbol')['daily_return'].transform(
        lambda x: x.rolling(20).std() * np.sqrt(252)
    )
    print(f"  rv_20d: mean={df['rv_20d'].dropna().mean():.4f}")

    # 5. RV / IV ratio (realized vs implied — vol risk premium)
    df['rv_iv_ratio'] = df['rv_20d'] / df['iv']
    df['rv_iv_ratio'] = df['rv_iv_ratio'].clip(0, 5)  # cap outliers
    print(f"  rv_iv_ratio: mean={df['rv_iv_ratio'].dropna().mean():.4f}")

    # 6. IV skew proxy: how far this option's IV is from the ticker's median IV on this date
    date_ticker_iv_median = df.groupby(['symbol', 'date'])['iv'].transform('median')
    df['iv_skew_proxy'] = df['iv'] - date_ticker_iv_median
    print(f"  iv_skew_proxy: mean={df['iv_skew_proxy'].dropna().mean():.4f}")

    # 7. DTE bucket
    df['dte_bucket'] = pd.cut(df['dte'], bins=[0, 25, 35, 45, 60], labels=[1, 2, 3, 4]).astype(float)
    print(f"  dte_bucket: distribution={df['dte_bucket'].value_counts().to_dict()}")

    # 8. Theta/Vega ratio (time decay efficiency)
    df['theta_vega_ratio'] = np.where(df['vega'] > 0.001, np.abs(df['theta']) / df['vega'], 0)
    df['theta_vega_ratio'] = df['theta_vega_ratio'].clip(0, 10)
    print(f"  theta_vega_ratio: mean={df['theta_vega_ratio'].mean():.4f}")

    # 9. Delta-adjusted moneyness (interaction feature)
    df['delta_moneyness'] = df['delta'].abs() * df['moneyness']
    print(f"  delta_moneyness: mean={df['delta_moneyness'].mean():.4f}")

    # Cleanup temp columns
    df.drop(columns=['daily_return'], inplace=True)

    return df


def ticker_weighted_sample(df: pd.DataFrame, target_per_ticker: int = 30000) -> pd.DataFrame:
    """Balance ticker representation by downsampling large tickers."""
    print(f"\nTicker-weighted sampling (target {target_per_ticker:,} per ticker)...")
    groups = []
    for ticker, group in df.groupby('symbol'):
        if len(group) > target_per_ticker:
            sampled = group.sample(n=target_per_ticker, random_state=42)
            print(f"  {ticker}: {len(group):,} → {target_per_ticker:,} (downsampled)")
        else:
            sampled = group
            print(f"  {ticker}: {len(group):,} (kept all)")
        groups.append(sampled)
    result = pd.concat(groups).reset_index(drop=True)
    print(f"  Total: {len(result):,} rows")
    return result


def time_based_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """Split by date — last 20% of dates as test set."""
    dates = sorted(df['date'].unique())
    split_idx = int(len(dates) * (1 - test_ratio))
    split_date = dates[split_idx]
    train = df[df['date'] < split_date]
    test = df[df['date'] >= split_date]
    print(f"\nTime-based split:")
    print(f"  Train: {len(train):,} rows ({dates[0]} to {split_date})")
    print(f"  Test:  {len(test):,} rows ({split_date} to {dates[-1]})")
    return train, test, split_date


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='Training CSV path')
    parser.add_argument('--out', default=None, help='Output CSV path')
    parser.add_argument('--target-per-ticker', type=int, default=30000)
    args = parser.parse_args()

    print(f"Loading: {args.data}")
    df = pd.read_csv(args.data)
    print(f"Loaded: {len(df):,} rows, {df['symbol'].nunique()} tickers")

    # Add features
    df = add_features(df)

    # Show new feature list
    new_features = ['credit_pct', 'iv_vs_ticker_avg', 'return_5d', 'return_10d', 'return_20d',
                    'rv_20d', 'rv_iv_ratio', 'iv_skew_proxy', 'dte_bucket', 'theta_vega_ratio',
                    'delta_moneyness']
    print(f"\nNew features added: {len(new_features)}")
    for f in new_features:
        nulls = df[f].isnull().sum()
        print(f"  {f}: nulls={nulls} ({nulls/len(df)*100:.1f}%)")

    # Ticker-weighted sampling
    df_balanced = ticker_weighted_sample(df, args.target_per_ticker)

    # Time-based split info
    train, test, split_date = time_based_split(df_balanced)

    # Save enhanced + balanced dataset
    out_path = args.out or args.data.replace('.csv', '_enhanced.csv')
    df_balanced.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df_balanced):,} rows)")

    # Also save train/test splits
    train_path = out_path.replace('.csv', '_train.csv')
    test_path = out_path.replace('.csv', '_test.csv')
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"Train: {train_path} ({len(train):,} rows)")
    print(f"Test:  {test_path} ({len(test):,} rows)")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Original features: 19")
    print(f"New features: {len(new_features)}")
    print(f"Total features: {19 + len(new_features)}")
    print(f"Original rows: {len(df):,}")
    print(f"After balancing: {len(df_balanced):,}")
    print(f"Train/Test split date: {split_date}")
    print(f"Train hit rate: {train['hit_50pct'].mean()*100:.1f}%")
    print(f"Test hit rate: {test['hit_50pct'].mean()*100:.1f}%")


if __name__ == '__main__':
    main()
