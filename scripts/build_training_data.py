#!/usr/bin/env python3
"""
Build ML training data from OptionsDX .txt files.

For each OTM short option candidate (|delta| 0.10-0.40, DTE 21-60):
  - Extract 16 features (Greeks, technicals, moneyness, etc.)
  - Simulate the trade to expiry
  - Record 5 response variables (hit_50pct, days_to_50pct, max_profit_pct, expected_value, outcome)

Usage:
  python3 scripts/build_training_data.py "data/optionsdx"
  python3 scripts/build_training_data.py "data/optionsdx" --ticker NVDA --out training.parquet
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Earnings data fetcher ──────────────────────────────────────────────────

def fetch_earnings_dates(ticker: str) -> list:
    """Fetch historical earnings dates for a ticker via yfinance."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        dates = t.earnings_dates
        if dates is None or len(dates) == 0:
            return []
        return sorted([pd.Timestamp(d).normalize() for d in dates.index])
    except Exception as e:
        print(f'  Warning: failed to fetch earnings for {ticker}: {e}')
        return []


def add_earnings_features(candidates: pd.DataFrame, earnings_dates: list) -> pd.DataFrame:
    """Add days_to_earnings, earnings_in_window, post_earnings features."""
    if not earnings_dates:
        candidates['days_to_earnings'] = np.nan
        candidates['earnings_in_window'] = 0
        candidates['post_earnings'] = 0
        return candidates

    earnings_arr = np.array([d.value for d in earnings_dates])

    def days_and_flags(date, expiry):
        d = pd.Timestamp(date).value
        e = pd.Timestamp(expiry).value
        # Next earnings (>= date): closest future
        future = earnings_arr[earnings_arr >= d]
        next_ed = future.min() if len(future) > 0 else None
        # Last earnings (< date): closest past
        past = earnings_arr[earnings_arr < d]
        last_ed = past.max() if len(past) > 0 else None

        days_to = (next_ed - d) / (86400 * 1e9) if next_ed is not None else np.nan
        in_window = int(next_ed is not None and next_ed <= e)  # earnings between open and expiry
        post = int(last_ed is not None and (d - last_ed) / (86400 * 1e9) <= 30)
        return days_to, in_window, post

    results = candidates.apply(lambda r: days_and_flags(r['date'], r['expiry']), axis=1, result_type='expand')
    results.columns = ['days_to_earnings', 'earnings_in_window', 'post_earnings']
    for col in results.columns:
        candidates[col] = results[col].values
    return candidates


# ── OptionsDX loader ────────────────────────────────────────────────────────

OPTIONSDX_COLS = [
    'quote_unixtime', 'quote_readtime', 'quote_date', 'quote_time_hours',
    'underlying_last', 'expire_date', 'expire_unix', 'dte',
    'c_delta', 'c_gamma', 'c_vega', 'c_theta', 'c_rho', 'c_iv',
    'c_volume', 'c_last', 'c_size', 'c_bid', 'c_ask',
    'strike',
    'p_bid', 'p_ask', 'p_size', 'p_last',
    'p_delta', 'p_gamma', 'p_vega', 'p_theta', 'p_rho', 'p_iv',
    'p_volume', 'strike_distance', 'strike_distance_pct',
]


def load_ticker(ticker_dir: str, ticker: str) -> pd.DataFrame:
    """Load all .txt files for a ticker, return melted call+put DataFrame."""
    files = sorted(glob.glob(os.path.join(ticker_dir, '*.txt')))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(
                f,
                header=0,
                names=OPTIONSDX_COLS,
                skipinitialspace=True,
                low_memory=False,
            )
            dfs.append(df)
        except Exception as e:
            print(f'  Warning: {os.path.basename(f)}: {e}')

    if not dfs:
        return pd.DataFrame()

    raw = pd.concat(dfs, ignore_index=True)

    # Clean dates
    raw['quote_date'] = pd.to_datetime(raw['quote_date'].str.strip(), errors='coerce')
    raw['expire_date'] = pd.to_datetime(raw['expire_date'].str.strip(), errors='coerce')
    raw = raw.dropna(subset=['quote_date', 'expire_date'])

    # Numeric columns
    num_cols = ['underlying_last', 'strike', 'dte',
                'c_delta', 'c_gamma', 'c_vega', 'c_theta', 'c_iv', 'c_bid', 'c_ask', 'c_volume',
                'p_delta', 'p_gamma', 'p_vega', 'p_theta', 'p_iv', 'p_bid', 'p_ask', 'p_volume']
    for col in num_cols:
        raw[col] = pd.to_numeric(raw[col], errors='coerce').fillna(0)

    # Melt calls
    calls = raw[['quote_date', 'expire_date', 'underlying_last', 'strike', 'dte',
                  'c_delta', 'c_gamma', 'c_vega', 'c_theta', 'c_iv', 'c_bid', 'c_ask', 'c_volume']].copy()
    calls.columns = ['date', 'expiry', 'underlying', 'strike', 'dte',
                     'delta', 'gamma', 'vega', 'theta', 'iv', 'bid', 'ask', 'volume']
    calls['option_type'] = 'CALL'
    calls['symbol'] = ticker

    # Melt puts
    puts = raw[['quote_date', 'expire_date', 'underlying_last', 'strike', 'dte',
                'p_delta', 'p_gamma', 'p_vega', 'p_theta', 'p_iv', 'p_bid', 'p_ask', 'p_volume']].copy()
    puts.columns = ['date', 'expiry', 'underlying', 'strike', 'dte',
                    'delta', 'gamma', 'vega', 'theta', 'iv', 'bid', 'ask', 'volume']
    puts['option_type'] = 'PUT'
    puts['symbol'] = ticker

    result = pd.concat([calls, puts], ignore_index=True)
    result['mid'] = (result['bid'] + result['ask']) / 2

    # Filter out zero-price rows
    result = result[(result['bid'] > 0) | (result['ask'] > 0)]

    return result


# ── Technical indicators (vectorized) ───────────────────────────────────────

def calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI with Wilder's smoothing (matches TradingView, StockCharts)."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    # Wilder's EMA: alpha = 1/period (equivalent to ewm with alpha=1/period)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_macd_signal(prices: pd.Series) -> pd.Series:
    ema12 = prices.ewm(span=12).mean()
    ema26 = prices.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return (macd > signal).astype(int) - (macd < signal).astype(int)


def calc_bb_position(prices: pd.Series, period: int = 20) -> pd.Series:
    """Bollinger %B (matches TradingView: uses population std, ddof=0)."""
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std(ddof=0)
    upper = sma + 2 * std
    lower = sma - 2 * std
    return ((prices - lower) / (upper - lower)).clip(0, 1)


def calc_resistance_score(strikes: pd.Series, prices: pd.Series,
                          high90: pd.Series, low90: pd.Series,
                          option_types: pd.Series) -> pd.Series:
    """
    Asymmetric support/resistance score (0-1, higher = safer strike).

    For CALL (short call, CC): compare strike to 90-day HIGH (resistance).
      Safer when strike is ABOVE resistance.
      score = clip((strike - high90) / price + 0.05) / 0.10  → 0 at -5%, 0.5 at resistance, 1 at +5%

    For PUT (short put, CSP): compare strike to 90-day LOW (support).
      Safer when strike is BELOW support.
      score = clip((low90 - strike) / price + 0.05) / 0.10
    """
    # Beyond safe side (positive = safer)
    call_beyond = (strikes - high90) / prices
    put_beyond = (low90 - strikes) / prices
    beyond = np.where(option_types == 'CALL', call_beyond, put_beyond)
    score = (beyond + 0.05) / 0.10
    return pd.Series(np.clip(score, 0, 1), index=strikes.index)


# ── Trade simulation (vectorized) ──────────────────────────────────────────

def simulate_trades(candidates: pd.DataFrame, all_data: pd.DataFrame) -> pd.DataFrame:
    """
    For each candidate (short option entry), track daily mid prices until expiry.
    Compute 5 response variables.
    """
    results = []

    # Group all_data by (symbol, expiry, strike, option_type) for fast lookup
    grouped = all_data.groupby(['symbol', 'expiry', 'strike', 'option_type'])

    total = len(candidates)
    t0 = time.time()

    for i, (idx, row) in enumerate(candidates.iterrows()):
        if i % 10000 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (total - i) / rate
            print(f'  Simulating {i}/{total} ({i/total*100:.1f}%) — {rate:.0f}/s — ETA {eta:.0f}s')

        key = (row['symbol'], row['expiry'], row['strike'], row['option_type'])
        try:
            contract_data = grouped.get_group(key)
        except KeyError:
            continue

        # Future prices after entry date
        future = contract_data[contract_data['date'] > row['date']].sort_values('date')
        if future.empty:
            continue

        open_mid = row['mid']
        if open_mid <= 0:
            continue

        # Track daily profit %
        future_mids = future['mid'].values
        profit_pcts = (open_mid - future_mids) / open_mid

        # R1: hit_50pct
        hit_mask = profit_pcts >= 0.5
        hit_50 = hit_mask.any()
        days_to_50 = int(np.argmax(hit_mask) + 1) if hit_50 else None

        # R3: max_profit_pct (at expiry or last available) — clipped to [-3, 1]
        # -3 means "option value tripled" which is already a catastrophic loss
        max_profit_pct = float(np.clip(profit_pcts[-1], -3.0, 1.0))

        # R4: expected_value ($) — also clip dollar values to cap tail outliers
        # Max win bounded by credit (open_mid × 100); cap loss at 3 × credit
        profit_dollars = (open_mid - future_mids[-1]) * 100
        expected_value = float(np.clip(profit_dollars, -3 * open_mid * 100, open_mid * 100))

        # R5: outcome
        if max_profit_pct >= 0.8:
            outcome = 'full_win'
        elif max_profit_pct >= 0.2:
            outcome = 'partial_win'
        elif max_profit_pct >= -0.2:
            outcome = 'breakeven'
        else:
            outcome = 'loss'

        results.append({
            'idx': idx,
            'hit_50pct': hit_50,
            'days_to_50pct': days_to_50,
            'max_profit_pct': max_profit_pct,
            'expected_value': expected_value,
            'outcome': outcome,
        })

    elapsed = time.time() - t0
    print(f'  Simulation complete: {len(results)} trades in {elapsed:.1f}s')

    return pd.DataFrame(results).set_index('idx') if results else pd.DataFrame()


# ── Main pipeline ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Build ML training data from OptionsDX')
    parser.add_argument('data_dir', help='Directory with ticker subdirs (NVDA/, SPY/, etc.)')
    parser.add_argument('--ticker', help='Process only this ticker')
    parser.add_argument('--exclude', help='Comma-separated tickers to skip (e.g. VIX,SPX)')
    parser.add_argument('--cache-dir', default='data/cache',
                        help='Cache directory for simulated outcomes')
    parser.add_argument('--force-resim', action='store_true',
                        help='Force re-simulation even if cached (default: use cache)')
    parser.add_argument('--out', default='training_data.parquet', help='Output file')
    parser.add_argument('--min-dte', type=int, default=21)
    parser.add_argument('--max-dte', type=int, default=60)
    parser.add_argument('--min-delta', type=float, default=0.10)
    parser.add_argument('--max-delta', type=float, default=0.40)
    parser.add_argument('--sample', type=float, default=1.0, help='Fraction to sample (0.1 = 10%)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Find ticker directories
    excluded = set((args.exclude or '').upper().split(',')) - {''}
    if args.ticker:
        ticker_dirs = [(args.ticker.upper(), str(data_dir / args.ticker.upper()))]
    else:
        ticker_dirs = [
            (d.name, str(d))
            for d in sorted(data_dir.iterdir())
            if d.is_dir() and not d.name.startswith('.') and d.name.upper() not in excluded
        ]

    print(f'Tickers: {[t for t, _ in ticker_dirs]}')
    print(f'DTE: {args.min_dte}-{args.max_dte}, Delta: {args.min_delta}-{args.max_delta}')
    print(f'Sample: {args.sample*100:.0f}%')
    print()

    all_training = []

    for ticker, ticker_path in ticker_dirs:
        print(f'=== {ticker} ===')
        t0 = time.time()

        # Load data
        print(f'  Loading...')
        df = load_ticker(ticker_path, ticker)
        if df.empty:
            print(f'  No data, skipping')
            continue
        print(f'  Loaded {len(df):,} rows in {time.time()-t0:.1f}s')

        # Underlying price series (one per date, for technicals)
        daily_underlying = (
            df.groupby('date')['underlying']
            .first()
            .sort_index()
        )

        # Compute technical indicators on underlying
        print(f'  Computing technicals...')
        rsi = calc_rsi(daily_underlying)
        macd = calc_macd_signal(daily_underlying)
        bb = calc_bb_position(daily_underlying)
        high90 = daily_underlying.rolling(90, min_periods=20).max()
        low90 = daily_underlying.rolling(90, min_periods=20).min()

        # IV rank: current IV vs 252-day range
        # Use ATM call IV (closest to |delta|=0.5) per day as proxy
        atm_iv = (
            df[df['option_type'] == 'CALL']
            .assign(atm_dist=lambda x: (x['delta'] - 0.5).abs())
            .sort_values('atm_dist')
            .groupby('date')['iv']
            .first()
            .sort_index()
        )
        iv_high = atm_iv.rolling(252, min_periods=20).max()
        iv_low = atm_iv.rolling(252, min_periods=20).min()

        # IVP: percentile rank of current IV in last 252 days (0-100)
        # pct=True returns 0-1; multiply by 100
        ivp_series = atm_iv.rolling(252, min_periods=20).rank(pct=True) * 100

        # Filter to short option candidates (with data-quality bounds)
        print(f'  Filtering candidates...')
        mask = (
            (df['dte'] >= args.min_dte) &
            (df['dte'] <= args.max_dte) &
            (df['delta'].abs() >= args.min_delta) &
            (df['delta'].abs() <= args.max_delta) &
            (df['mid'] > 0.05) &
            # Data quality filters
            (df['iv'] > 0.05) & (df['iv'] < 3.0) &                 # IV 5%-300%
            (df['vega'].abs() < 100) &                             # no absurd vega
            (df['gamma'] >= 0) & (df['gamma'] < 1.0) &             # gamma sanity
            (df['theta'] < 0) & (df['theta'] > -10) &              # theta must decay
            (df['strike'] / df['underlying'] > 0.3) &              # moneyness 0.3-3
            (df['strike'] / df['underlying'] < 3.0)
        )
        candidates = df[mask].copy()
        print(f'  Candidates: {len(candidates):,}')

        if candidates.empty:
            continue

        # Sample if requested
        if args.sample < 1.0:
            candidates = candidates.sample(frac=args.sample, random_state=42)
            print(f'  Sampled: {len(candidates):,}')

        # Add features
        print(f'  Extracting features...')
        candidates['moneyness'] = candidates['strike'] / candidates['underlying']
        # Capital at risk:
        #   PUT (CSP): strike × 100 (cash secured)
        #   CALL (CC): underlying × 100 (market value of shares owned)
        capital = np.where(
            candidates['option_type'] == 'CALL',
            candidates['underlying'] * 100,
            candidates['strike'] * 100,
        )
        candidates['annualized_return'] = (
            (candidates['mid'] * 100 / capital) * (365 / candidates['dte'])
        )
        candidates['credit_received'] = candidates['mid'] * 100
        candidates['max_loss'] = candidates['strike'] * 100

        # Map technicals by date
        candidates['rsi14'] = candidates['date'].map(rsi)
        candidates['macd_signal'] = candidates['date'].map(macd)
        candidates['bb_position'] = candidates['date'].map(bb)

        # Resistance score
        candidates['_high90'] = candidates['date'].map(high90)
        candidates['_low90'] = candidates['date'].map(low90)
        candidates['resistance_score'] = calc_resistance_score(
            candidates['strike'], candidates['underlying'],
            candidates['_high90'], candidates['_low90'],
            candidates['option_type']
        )
        candidates.drop(columns=['_high90', '_low90'], inplace=True)

        # IV Rank (clipped 0-100)
        candidates['_iv_high'] = candidates['date'].map(iv_high)
        candidates['_iv_low'] = candidates['date'].map(iv_low)
        iv_range = candidates['_iv_high'] - candidates['_iv_low']
        candidates['ivr'] = np.where(
            iv_range > 0,
            ((candidates['iv'] - candidates['_iv_low']) / iv_range * 100).clip(0, 100),
            50
        )
        candidates.drop(columns=['_iv_high', '_iv_low'], inplace=True)

        # IV Percentile (percentile rank of current IV over last 252 days)
        candidates['ivp'] = candidates['date'].map(ivp_series)

        # Earnings features
        print(f'  Fetching earnings dates...')
        earnings_dates = fetch_earnings_dates(ticker)
        print(f'  Found {len(earnings_dates)} earnings dates')
        candidates = add_earnings_features(candidates, earnings_dates)

        # Simulate trades (use cache if available, keyed by candidate identity)
        sim_cache = cache_dir / f'sim_{ticker}.pkl'
        cached_sim = None
        if sim_cache.exists() and not args.force_resim:
            try:
                cached_sim = pd.read_pickle(sim_cache)
                print(f'  Using cached simulations: {len(cached_sim):,} rows')
            except Exception as e:
                print(f'  Cache read failed ({e}), re-simulating')
                cached_sim = None

        # Candidates identity key: (symbol, date, strike, expiry, option_type)
        id_cols = ['symbol', 'date', 'strike', 'expiry', 'option_type']

        if cached_sim is not None:
            # Merge cached outcomes onto current candidates
            result = candidates.merge(cached_sim, on=id_cols, how='inner')
            print(f'  Matched cached simulations: {len(result):,} / {len(candidates):,} candidates')
        else:
            print(f'  Simulating trades...')
            outcomes = simulate_trades(candidates, df)

            if outcomes.empty:
                print(f'  No outcomes, skipping')
                continue

            # Save simulation cache (outcomes + identity cols)
            sim_to_cache = candidates.loc[outcomes.index, id_cols].join(outcomes)
            sim_to_cache.to_pickle(sim_cache)
            print(f'  Cached {len(sim_to_cache):,} simulations to {sim_cache.name}')

            # Join features + outcomes
            result = candidates.join(outcomes, how='inner')

        print(f'  Training rows: {len(result):,}')

        all_training.append(result)

    if not all_training:
        print('\nNo training data generated!')
        sys.exit(1)

    # Combine all tickers
    combined = pd.concat(all_training, ignore_index=True)

    # Select final columns
    feature_cols = [
        'symbol', 'date', 'expiry', 'strike', 'option_type', 'dte',
        'delta', 'gamma', 'theta', 'vega', 'iv', 'ivr', 'ivp',
        'moneyness', 'annualized_return', 'credit_received', 'max_loss',
        'rsi14', 'macd_signal', 'bb_position', 'resistance_score',
        'days_to_earnings', 'earnings_in_window', 'post_earnings',
        'underlying',
        # Response variables
        'hit_50pct', 'days_to_50pct', 'max_profit_pct', 'expected_value', 'outcome',
    ]

    # Keep only columns that exist
    final_cols = [c for c in feature_cols if c in combined.columns]
    combined = combined[final_cols]

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if str(out_path).endswith('.parquet'):
        combined.to_parquet(out_path, index=False)
    else:
        combined.to_csv(out_path, index=False)

    size_mb = out_path.stat().st_size / 1024 / 1024

    print(f'\n{"="*60}')
    print(f'Training data saved: {out_path}')
    print(f'Total rows: {len(combined):,}')
    print(f'File size: {size_mb:.1f} MB')
    print(f'\nResponse variable stats:')
    print(f'  hit_50pct:       {combined["hit_50pct"].mean()*100:.1f}% hit rate')
    if 'days_to_50pct' in combined.columns:
        d50 = combined['days_to_50pct'].dropna()
        if len(d50) > 0:
            print(f'  days_to_50pct:   median {d50.median():.0f} days')
    print(f'  max_profit_pct:  mean {combined["max_profit_pct"].mean()*100:.1f}%')
    print(f'  expected_value:  mean ${combined["expected_value"].mean():.0f}')
    print(f'  outcome:')
    for outcome, count in combined['outcome'].value_counts().items():
        print(f'    {outcome}: {count:,} ({count/len(combined)*100:.1f}%)')


if __name__ == '__main__':
    main()
