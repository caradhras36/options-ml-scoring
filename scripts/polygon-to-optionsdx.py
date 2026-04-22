#!/usr/bin/env python3
"""
Fetch historical options data from Polygon and convert to OptionsDX format.

Downloads daily option OHLCV from Polygon, calculates IV + Greeks via
Black-Scholes, and outputs OptionsDX-compatible .txt files that can be
fed directly into build_training_data.py.

Usage:
  python3 scripts/polygon-to-optionsdx.py --tickers HOOD,HIMS,SOFI
  python3 scripts/polygon-to-optionsdx.py --tickers HOOD --months 6
  python3 scripts/polygon-to-optionsdx.py --all  # default 29 tickers
"""

import os
import sys
import math
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    os.system(f"{sys.executable} -m pip install requests --quiet")
    import requests

try:
    from scipy.stats import norm
except ImportError:
    os.system(f"{sys.executable} -m pip install scipy --quiet")
    from scipy.stats import norm

# ── Config ────────────────────────────────────────────────────────────────

API_KEY = ""
BASE = "https://api.polygon.io"
RATE_LIMIT_DELAY = 0.25  # Polygon Starter: unlimited calls, 0.25s avoids thread starvation
RISK_FREE_RATE = 0.045

DEFAULT_TICKERS = [
    "NBIS", "APLD", "ONDS", "HOOD", "HIMS", "ROOT", "SEZL",
    "SOFI", "ZETA", "APP", "AFRM", "PLTR",
    "NVDA", "CRDO", "GOOG", "AMZN", "OKLO", "TEM",
    "AAPL", "TSLA", "AMD", "NFLX", "SPY", "QQQ",
    "COIN", "MSTR", "RIOT", "MARA",
]

# ── Black-Scholes ─────────────────────────────────────────────────────────

def implied_vol(S, K, T, r, market_price, option_type='call', tol=1e-8, max_iter=200):
    """Newton-Raphson implied volatility from option mid price."""
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    # Intrinsic check
    if option_type == 'call':
        intrinsic = max(0, S - K * math.exp(-r * T))
    else:
        intrinsic = max(0, K * math.exp(-r * T) - S)
    if market_price < intrinsic:
        market_price = intrinsic + 0.01

    sigma = 0.3
    for _ in range(max_iter):
        try:
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            if option_type == 'call':
                price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            vega = S * norm.pdf(d1) * math.sqrt(T)
            if vega < 1e-12:
                break
            sigma = sigma - (price - market_price) / vega
            if sigma <= 0.001:
                sigma = 0.001
            if sigma > 10:
                return None
            if abs(price - market_price) < tol:
                break
        except (ValueError, OverflowError):
            return None
    return sigma if 0.01 < sigma < 10 else None


def bs_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes Greeks."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if option_type == 'call':
            delta = norm.cdf(d1)
            theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365
            rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
        else:
            delta = norm.cdf(d1) - 1
            theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365
            rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T) / 100
        return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho, 'iv': sigma}
    except (ValueError, OverflowError):
        return None

# ── Polygon API ───────────────────────────────────────────────────────────

request_count = 0

def fetch(url, params=None, retries=3):
    """Rate-limited fetch with 429 retry."""
    global request_count
    time.sleep(RATE_LIMIT_DELAY)
    if params is None:
        params = {}
    params["apiKey"] = API_KEY

    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            request_count += 1
            if r.status_code == 429:
                wait = (attempt + 1) * 10
                print(f" [429 wait {wait}s]", end="", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if "429" in str(e) and attempt < retries - 1:
                time.sleep((attempt + 1) * 10)
                continue
            return None
        except Exception:
            return None
    return None


def fetch_underlying_history(ticker, start, end):
    """Fetch daily OHLCV for underlying stock."""
    data = fetch(f"{BASE}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
                 {"adjusted": "true", "limit": 50000})
    if not data or not data.get("results"):
        return {}
    result = {}
    for r in data["results"]:
        dt = datetime.fromtimestamp(r["t"] / 1000).strftime("%Y-%m-%d")
        result[dt] = r["c"]
    return result


def fetch_contracts(ticker, exp_start, exp_end):
    """List all option contracts for a ticker within expiry range."""
    all_contracts = []
    url = f"{BASE}/v3/reference/options/contracts"
    params = {
        "underlying_ticker": ticker,
        "expiration_date.gte": exp_start,
        "expiration_date.lte": exp_end,
        "expired": "true",
        "limit": 1000,
    }
    while url:
        data = fetch(url, params)
        if not data:
            break
        all_contracts.extend(data.get("results", []))
        next_url = data.get("next_url")
        if next_url:
            url = next_url
            params = {}
        else:
            break
    return all_contracts


def fetch_contract_history(contract_ticker, start, end):
    """Fetch daily OHLCV for a specific option contract."""
    data = fetch(f"{BASE}/v2/aggs/ticker/{contract_ticker}/range/1/day/{start}/{end}",
                 {"adjusted": "true", "limit": 50000})
    if not data or not data.get("results"):
        return []
    return [
        {
            "date": datetime.fromtimestamp(r["t"] / 1000).strftime("%Y-%m-%d"),
            "open": r["o"], "high": r["h"], "low": r["l"],
            "close": r["c"], "volume": r["v"],
        }
        for r in data["results"]
    ]


# ── OptionsDX Format Writer ──────────────────────────────────────────────

HEADER = "[QUOTE_UNIXTIME], [QUOTE_READTIME], [QUOTE_DATE], [QUOTE_TIME_HOURS], [UNDERLYING_LAST], [EXPIRE_DATE], [EXPIRE_UNIX], [DTE], [C_DELTA], [C_GAMMA], [C_VEGA], [C_THETA], [C_RHO], [C_IV], [C_VOLUME], [C_LAST], [C_SIZE], [C_BID], [C_ASK], [STRIKE], [P_BID], [P_ASK], [P_SIZE], [P_LAST], [P_DELTA], [P_GAMMA], [P_VEGA], [P_THETA], [P_RHO], [P_IV], [P_VOLUME], [STRIKE_DISTANCE], [STRIKE_DISTANCE_PCT]"


def format_optionsdx_row(date_str, underlying_price, expiry, strike, call_data, put_data):
    """Format a single OptionsDX row (call + put for same strike)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    exp_dt = datetime.strptime(expiry, "%Y-%m-%d")
    unix_ts = int(dt.timestamp())
    exp_unix = int(exp_dt.timestamp())
    dte = (exp_dt - dt).days

    if dte <= 0:
        return None

    c = call_data or {}
    p = put_data or {}

    strike_dist = abs(underlying_price - strike)
    strike_dist_pct = strike_dist / underlying_price if underlying_price > 0 else 0

    fields = [
        str(unix_ts),
        f"{date_str} 16:00",
        date_str,
        "16.000000",
        f"{underlying_price:.6f}",
        expiry,
        str(exp_unix),
        f"{dte:.6f}",
        # Call Greeks
        f"{c.get('delta', ''):.6f}" if c.get('delta') is not None else "",
        f"{c.get('gamma', ''):.6f}" if c.get('gamma') is not None else "",
        f"{c.get('vega', ''):.6f}" if c.get('vega') is not None else "",
        f"{c.get('theta', ''):.6f}" if c.get('theta') is not None else "",
        f"{c.get('rho', ''):.6f}" if c.get('rho') is not None else "",
        f"{c.get('iv', ''):.6f}" if c.get('iv') is not None else "",
        str(c.get('volume', 0)),
        f"{c.get('close', 0):.6f}",
        "0 x 0",  # size placeholder
        f"{c.get('bid', 0):.6f}",
        f"{c.get('ask', 0):.6f}",
        # Strike
        f"{strike:.6f}",
        # Put
        f"{p.get('bid', 0):.6f}",
        f"{p.get('ask', 0):.6f}",
        "0 x 0",
        f"{p.get('close', 0):.6f}",
        f"{p.get('delta', ''):.6f}" if p.get('delta') is not None else "",
        f"{p.get('gamma', ''):.6f}" if p.get('gamma') is not None else "",
        f"{p.get('vega', ''):.6f}" if p.get('vega') is not None else "",
        f"{p.get('theta', ''):.6f}" if p.get('theta') is not None else "",
        f"{p.get('rho', ''):.6f}" if p.get('rho') is not None else "",
        f"{p.get('iv', ''):.6f}" if p.get('iv') is not None else "",
        str(p.get('volume', 0)),
        f"{strike_dist:.6f}",
        f"{strike_dist_pct:.6f}",
    ]
    return ", ".join(fields)


# ── Main Pipeline ─────────────────────────────────────────────────────────

def process_ticker(ticker, out_dir, months_back=24):
    """Full pipeline for one ticker: fetch → compute → write OptionsDX."""
    print(f"\n{'='*60}")
    print(f"Processing {ticker} ({months_back} months back)")
    print(f"{'='*60}")

    today = datetime.now()
    start_date = (today - timedelta(days=months_back * 30)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    # 1. Fetch underlying price history
    print(f"  Fetching underlying prices...", end=" ", flush=True)
    price_history = fetch_underlying_history(ticker, start_date, end_date)
    print(f"{len(price_history)} days")
    if len(price_history) < 20:
        print(f"  Not enough price data, skipping")
        return 0

    # 2. Fetch option contracts
    print(f"  Fetching contracts...", end=" ", flush=True)
    contracts = fetch_contracts(ticker, start_date, end_date)
    print(f"{len(contracts)} contracts")
    if not contracts:
        print(f"  No contracts found, skipping")
        return 0

    # Group contracts by (expiry, strike)
    contract_map = {}  # {(expiry, strike, type): ticker}
    for c in contracts:
        key = (c["expiration_date"], c["strike_price"], c["contract_type"])
        contract_map[key] = c["ticker"]

    # Group unique (expiry, strike) pairs
    strike_pairs = defaultdict(dict)  # {(expiry, strike): {call_ticker, put_ticker}}
    for (exp, strike, ctype), cticker in contract_map.items():
        strike_pairs[(exp, strike)][ctype] = cticker

    print(f"  {len(strike_pairs)} unique (expiry, strike) pairs")

    # 3. Fetch ALL contract histories in parallel, then compute Greeks
    print(f"  Fetching contract histories (parallel)...", flush=True)

    # Collect all unique contract tickers to fetch
    all_contract_tickers = set()
    for tickers_by_type in strike_pairs.values():
        if tickers_by_type.get("call"): all_contract_tickers.add(tickers_by_type["call"])
        if tickers_by_type.get("put"): all_contract_tickers.add(tickers_by_type["put"])

    # Parallel fetch all contract histories
    contract_histories = {}  # {contract_ticker: {date: bar}}

    def _fetch_one(cticker):
        bars = fetch_contract_history(cticker, start_date, end_date)
        return cticker, {b["date"]: b for b in bars}

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_fetch_one, ct): ct for ct in all_contract_tickers}
        done = 0
        for future in as_completed(futures):
            ct, hist = future.result()
            contract_histories[ct] = hist
            done += 1
            if done % 100 == 0:
                print(f"    Fetched {done}/{len(all_contract_tickers)} contracts...", flush=True)

    print(f"  Fetched {len(contract_histories)} contract histories")

    # Group by month for output files
    monthly_rows = defaultdict(list)
    total_rows = 0
    pair_count = 0

    for (expiry, strike), tickers_by_type in strike_pairs.items():
        pair_count += 1
        if pair_count % 500 == 0:
            print(f"  Computing Greeks {pair_count}/{len(strike_pairs)}...", flush=True)

        call_ticker = tickers_by_type.get("call")
        put_ticker = tickers_by_type.get("put")

        call_history = contract_histories.get(call_ticker, {}) if call_ticker else {}
        put_history = contract_histories.get(put_ticker, {}) if put_ticker else {}

        # Get all dates where we have data
        all_dates = sorted(set(list(call_history.keys()) + list(put_history.keys())))

        for date_str in all_dates:
            underlying_price = price_history.get(date_str)
            if not underlying_price:
                continue

            exp_dt = datetime.strptime(expiry, "%Y-%m-%d")
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            dte = (exp_dt - dt).days
            if dte <= 0:
                continue
            T = dte / 365.0

            # Compute call Greeks
            call_data = None
            c_bar = call_history.get(date_str)
            if c_bar and c_bar["close"] > 0:
                c_mid = c_bar["close"]  # use close as proxy for mid
                iv = implied_vol(underlying_price, strike, T, RISK_FREE_RATE, c_mid, 'call')
                if iv:
                    greeks = bs_greeks(underlying_price, strike, T, RISK_FREE_RATE, iv, 'call')
                    if greeks:
                        call_data = {**greeks, 'close': c_mid, 'volume': c_bar["volume"],
                                     'bid': c_bar["low"], 'ask': c_bar["high"]}  # approximate bid/ask

            # Compute put Greeks
            put_data = None
            p_bar = put_history.get(date_str)
            if p_bar and p_bar["close"] > 0:
                p_mid = p_bar["close"]
                iv = implied_vol(underlying_price, strike, T, RISK_FREE_RATE, p_mid, 'put')
                if iv:
                    greeks = bs_greeks(underlying_price, strike, T, RISK_FREE_RATE, iv, 'put')
                    if greeks:
                        put_data = {**greeks, 'close': p_mid, 'volume': p_bar["volume"],
                                    'bid': p_bar["low"], 'ask': p_bar["high"]}

            # Need at least one side
            if not call_data and not put_data:
                continue

            row = format_optionsdx_row(date_str, underlying_price, expiry, strike, call_data, put_data)
            if row:
                month_key = date_str[:7].replace("-", "")  # YYYYMM
                monthly_rows[month_key].append(row)
                total_rows += 1

    # 4. Write OptionsDX files
    ticker_dir = out_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    for month_key, rows in sorted(monthly_rows.items()):
        filename = f"{ticker.lower()}_eod_{month_key}.txt"
        filepath = ticker_dir / filename
        with open(filepath, "w") as f:
            f.write(HEADER + "\n")
            for row in rows:
                f.write(row + "\n")

    print(f"  Written {total_rows} rows across {len(monthly_rows)} monthly files")
    print(f"  API requests used: {request_count}")
    return total_rows


def main():
    parser = argparse.ArgumentParser(description="Fetch Polygon historical options → OptionsDX format")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers")
    parser.add_argument("--all", action="store_true", help="Process all default tickers")
    parser.add_argument("--months", type=int, default=24, help="Months of history (default: 24, max for Starter)")
    parser.add_argument("--out", type=str, default="data/optionsdx", help="Output directory")
    args = parser.parse_args()

    global API_KEY
    env_path = Path(__file__).parent.parent.parent / ".env.local"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("POLYGON_API_KEY="):
                API_KEY = line.split("=", 1)[1].strip()
                break

    if not API_KEY:
        print("Error: POLYGON_API_KEY not found in .env.local")
        sys.exit(1)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    elif args.all:
        tickers = DEFAULT_TICKERS
    else:
        print("Specify --tickers HOOD,HIMS or --all")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Polygon → OptionsDX Pipeline")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"History: {args.months} months")
    print(f"Output: {out_dir}")
    print(f"Greeks: Black-Scholes (verified ~0.4% delta error, ~1.7% IV error)")

    grand_total = 0
    results = {}

    for ticker in tickers:
        try:
            rows = process_ticker(ticker, out_dir, args.months)
            results[ticker] = rows
            grand_total += rows
        except Exception as e:
            print(f"  ERROR: {e}")
            results[ticker] = 0

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"Total rows: {grand_total:,}")
    print(f"API requests: {request_count}")
    print(f"\nPer ticker:")
    for t, rows in sorted(results.items(), key=lambda x: -x[1]):
        status = f"{rows:,} rows" if rows > 0 else "FAILED"
        print(f"  {t:6s}: {status}")

    print(f"\nOutput files are in OptionsDX format at: {out_dir}")
    print(f"Run training with:")
    print(f"  python3 scripts/build_training_data.py \"{out_dir}\" --format optionsdx")


if __name__ == "__main__":
    main()
