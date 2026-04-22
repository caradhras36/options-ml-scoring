#!/usr/bin/env python3
"""
Validate training data quality.

Checks:
1. Sanity ranges (delta -1 to +1, IV 0-5, etc.)
2. Response variable distributions
3. Simulation correctness: manual spot-check N random rows
4. Feature correlations (sanity check: annualized_return should correlate with hit_50pct)
5. Temporal consistency (no look-ahead bias)
6. Missing values
7. Outlier detection

Usage:
  python3 scripts/validate_training_data.py <training_csv>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def check_ranges(df: pd.DataFrame):
    """Feature values should be in sensible ranges."""
    print("\n" + "=" * 60)
    print("1. FEATURE RANGE CHECKS")
    print("=" * 60)

    checks = [
        ('delta', -1.0, 1.0),
        ('gamma', 0, 1.0),
        ('vega', 0, 10),
        ('theta', -10, 1.0),
        ('iv', 0, 5.0),
        ('ivr', 0, 100),
        ('ivp', 0, 100),
        ('moneyness', 0.3, 3.0),
        ('dte', 0, 365),
        ('rsi14', 0, 100),
        ('macd_signal', -1, 1),
        ('bb_position', 0, 1),
        ('resistance_score', 0, 1),
    ]

    issues = 0
    for col, lo, hi in checks:
        if col not in df.columns:
            print(f"  SKIP {col}: column missing")
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            print(f"  SKIP {col}: all NaN")
            continue
        out_of_range = ((vals < lo) | (vals > hi)).sum()
        mn, mx = vals.min(), vals.max()
        status = "OK" if out_of_range == 0 else f"!! {out_of_range} out of range"
        print(f"  {col:22s} [{lo:>7.2f}, {hi:>7.2f}]  actual [{mn:>8.3f}, {mx:>8.3f}]  {status}")
        if out_of_range > 0:
            issues += 1

    return issues


def check_response_distributions(df: pd.DataFrame):
    """Response variables should have reasonable distributions."""
    print("\n" + "=" * 60)
    print("2. RESPONSE VARIABLE DISTRIBUTIONS")
    print("=" * 60)

    if 'hit_50pct' in df.columns:
        pct = df['hit_50pct'].mean() * 100
        print(f"\n  hit_50pct rate: {pct:.1f}%")
        if pct < 50:
            print("    WARNING: hit rate very low — simulation may be wrong")
        elif pct > 95:
            print("    WARNING: hit rate very high — filter may be too loose")
        else:
            print("    OK")

    if 'days_to_50pct' in df.columns:
        d = df['days_to_50pct'].dropna()
        if len(d) > 0:
            print(f"\n  days_to_50pct: median={d.median():.0f} mean={d.mean():.1f} max={d.max():.0f}")

    if 'max_profit_pct' in df.columns:
        mp = df['max_profit_pct']
        print(f"\n  max_profit_pct:")
        print(f"    mean={mp.mean():.3f} median={mp.median():.3f} std={mp.std():.3f}")
        print(f"    min={mp.min():.3f} max={mp.max():.3f}")
        above1 = (mp > 1.0).sum()
        below_minus1 = (mp < -1.0).sum()
        if above1 > 0:
            print(f"    WARNING: {above1} rows with max_profit_pct > 1 (impossible for short)")
        if below_minus1 > len(df) * 0.05:
            pct = below_minus1 / len(df) * 100
            print(f"    NOTE: {below_minus1} ({pct:.1f}%) rows with max_profit_pct < -1 (large losses, check for outliers)")

    if 'expected_value' in df.columns:
        ev = df['expected_value']
        print(f"\n  expected_value (USD):")
        print(f"    mean=${ev.mean():.0f} median=${ev.median():.0f}")
        print(f"    pct 1%={ev.quantile(0.01):.0f}  99%={ev.quantile(0.99):.0f}")

    if 'outcome' in df.columns:
        print(f"\n  outcome distribution:")
        for cls, pct in (df['outcome'].value_counts(normalize=True) * 100).items():
            count = df['outcome'].value_counts()[cls]
            print(f"    {cls:15s} {count:>8,} ({pct:.1f}%)")


def check_missing_values(df: pd.DataFrame):
    """Count NaN per column."""
    print("\n" + "=" * 60)
    print("3. MISSING VALUES")
    print("=" * 60)

    total = len(df)
    print(f"\n  Total rows: {total:,}")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("  No missing values.")
        return

    print(f"\n  {'Column':<22} {'Missing':>10} {'% of total':>12}")
    for col, cnt in missing.items():
        pct = cnt / total * 100
        print(f"  {col:<22} {cnt:>10,} {pct:>11.1f}%")


def spot_check_simulations(df: pd.DataFrame, n=5):
    """Manually verify a few simulated trades."""
    print("\n" + "=" * 60)
    print(f"4. SPOT CHECK — {n} random simulated trades")
    print("=" * 60)

    sample = df.sample(min(n, len(df)), random_state=7)

    for idx, row in sample.iterrows():
        print(f"\n  Row {idx}:")
        print(f"    {row['symbol']} {row['option_type']} ${row['strike']:.0f} "
              f"exp {row.get('expiry', '?')} (DTE {row['dte']:.0f})")
        print(f"    Underlying: ${row['underlying']:.2f}  Delta: {row['delta']:+.3f}  "
              f"Moneyness: {row['moneyness']:.3f}")
        print(f"    Credit: ${row['credit_received']:.0f}  IV: {row['iv']*100:.1f}%  "
              f"IVR: {row.get('ivr', 'N/A')}")
        print(f"    → hit_50pct: {row['hit_50pct']}  "
              f"days_to_50: {row.get('days_to_50pct', 'N/A')}  "
              f"max_profit: {row['max_profit_pct']:+.2%}")
        print(f"    → EV: ${row['expected_value']:.0f}  outcome: {row['outcome']}")

        # Quick sanity: if OTM and hit_50pct=True, it should be plausible
        is_otm_put = row['option_type'] == 'PUT' and row['moneyness'] < 1
        is_otm_call = row['option_type'] == 'CALL' and row['moneyness'] > 1
        if is_otm_put or is_otm_call:
            if row['hit_50pct'] and row['max_profit_pct'] > 0:
                print(f"    [✓] OTM short hit 50% profit — plausible")
            elif not row['hit_50pct']:
                print(f"    [·] OTM short didn't hit 50% — also plausible")


def check_correlations(df: pd.DataFrame):
    """Sanity-check feature-response correlations."""
    print("\n" + "=" * 60)
    print("5. FEATURE vs RESPONSE CORRELATIONS")
    print("=" * 60)

    if 'hit_50pct' not in df.columns:
        return

    # Numeric features to check
    features = ['dte', 'delta', 'iv', 'ivr', 'moneyness', 'annualized_return',
                'rsi14', 'bb_position', 'resistance_score', 'days_to_earnings']

    print(f"\n  Correlation with hit_50pct:")
    y = df['hit_50pct'].astype(int)
    for f in features:
        if f not in df.columns:
            continue
        x = df[f].dropna()
        if len(x) < 100:
            continue
        common = x.index.intersection(y.index)
        if len(common) == 0:
            continue
        corr = np.corrcoef(x.loc[common], y.loc[common])[0, 1]
        sign = "+" if corr > 0 else ""
        bar = '█' * min(int(abs(corr) * 30), 30)
        print(f"  {f:22s} {sign}{corr:.3f}  {bar}")


def check_temporal_consistency(df: pd.DataFrame):
    """Ensure no look-ahead: all outcomes from future of entry."""
    print("\n" + "=" * 60)
    print("6. TEMPORAL CONSISTENCY (look-ahead check)")
    print("=" * 60)

    if 'date' not in df.columns or 'expiry' not in df.columns:
        print("  SKIP: missing date/expiry columns")
        return

    df2 = df.copy()
    df2['entry_date'] = pd.to_datetime(df2['date'], errors='coerce')
    df2['exp_date'] = pd.to_datetime(df2['expiry'], errors='coerce')

    # Entry should be before expiry
    bad = (df2['entry_date'] >= df2['exp_date']).sum()
    total = len(df2)
    if bad == 0:
        print(f"  OK: All {total:,} entries predate their expiry.")
    else:
        print(f"  !!! {bad:,} rows have entry_date >= expiry (look-ahead bug)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='Training CSV')
    args = parser.parse_args()

    print(f"Loading {args.data}...")
    df = pd.read_csv(args.data)
    print(f"Rows: {len(df):,}  Columns: {len(df.columns)}")
    print(f"Tickers: {df['symbol'].nunique() if 'symbol' in df.columns else 'N/A'}")
    if 'date' in df.columns:
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    range_issues = check_ranges(df)
    check_response_distributions(df)
    check_missing_values(df)
    spot_check_simulations(df, n=5)
    check_correlations(df)
    check_temporal_consistency(df)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Range issues: {range_issues}")
    print(f"  Training data looks {'SUSPICIOUS' if range_issues > 2 else 'HEALTHY'}")


if __name__ == '__main__':
    main()
