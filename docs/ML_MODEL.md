# options-ml-scoring ML Model — Technical Reference

Complete documentation of the machine learning pipeline powering the CC/CSP recommendations.

---

## Overview

options-ml-scoring uses 5 XGBoost models trained on **1.73M simulated option trades** to score and explain covered call (CC) and cash-secured put (CSP) opportunities in real time.

**Training data source:** OptionsDX historical EOD option chains (2020-2023) for 6 underlyings: AAPL, NVDA, QQQ, SPX, SPY, TSLA.

**Serving:** Python FastAPI server at `localhost:8001` (`scripts/serve_model.py`). Next.js calls it per recommendation via `/api/recommendations`.

---

## Pipeline

```
OptionsDX .txt files (21.8M option rows, 7.8 GB)
          ↓
scripts/build_training_data.py
  ├─ Parse OptionsDX format (call/put pairs per strike/date)
  ├─ Filter candidates: 21-60 DTE, |delta| 0.10-0.40, mid > $0.05
  ├─ Compute 19 features per candidate
  ├─ Simulate each trade day-by-day to expiry
  ├─ Compute 5 response variables
  └─ Cache simulation outputs (sim_{ticker}.pkl)
          ↓
training/all_tickers_v3.csv (1.73M rows, 495 MB)
          ↓
scripts/train_model.py
  ├─ 5 XGBoost models
  ├─ Class-balanced (inverse frequency weighting)
  └─ 80/20 train/test split
          ↓
models/*.json (5 models, ~1.4 MB each)
          ↓
scripts/serve_model.py (FastAPI, port 8001)
  ├─ POST /predict → single prediction
  ├─ POST /predict/batch → multiple at once
  ├─ POST /explain → SHAP TreeExplainer
  └─ GET  /earnings/{ticker} → yfinance earnings (cached 24h)
          ↓
Next.js /api/recommendations
```

---

## Features (19 total)

### Option-specific (from Public.com chain)

| Feature | Range | Source | Description |
|---------|-------|--------|-------------|
| `dte` | 7-730 | calculated | Days to expiry |
| `delta` | -1 to +1 | Public.com | Option delta (long perspective) |
| `gamma` | ≥0 | Public.com | Rate of delta change |
| `theta` | ≤0 | Public.com | Daily time decay |
| `vega` | ≥0 | Public.com | Sensitivity to 1% IV change |
| `iv` | 0.05-5.0 | Public.com | Implied volatility (decimal) |
| `moneyness` | 0.5-2.0 | calculated | `strike / underlying_price` |
| `credit_received` | $5-50,000 | calculated | `mid × 100` per contract |
| `max_loss` | $500-1M | calculated | `strike × 100` (for naked short) |
| `annualized_return` | 0-1.0 | calculated | Return on capital at risk, annualized |

### Technical indicators (from Polygon historical price)

| Feature | Range | Description |
|---------|-------|-------------|
| `rsi14` | 0-100 | 14-period Relative Strength Index |
| `macd_signal` | -1/0/+1 | MACD histogram sign: +1 bullish cross, -1 bearish, 0 neutral |
| `bb_position` | 0-1 | Position within 20-period Bollinger Bands (%B) |
| `resistance_score` | 0-1 | Strike's distance from 90-day high/low pivots (normalized) |

### Volatility regime

| Feature | Range | Description |
|---------|-------|-------------|
| `ivr` | 0-100 | IV rank: `(current - 52wLow) / (52wHigh - 52wLow) × 100` |
| `ivp` | 0-100 | IV percentile: % of days in 252-day history below current IV |

### Earnings context (from yfinance)

| Feature | Range | Description |
|---------|-------|-------------|
| `days_to_earnings` | 0-999 | Days until next earnings announcement (999 if unknown/ETF) |
| `earnings_in_window` | 0/1 | 1 if earnings falls before option expiry |
| `post_earnings` | 0/1 | 1 if last earnings was ≤30 days ago |

### Special handling

**Capital at risk** (used in `annualized_return`):
- **CSP (PUT)**: `strike × 100` (cash secured)
- **CC (CALL)**: `underlying_price × 100` (market value of shares owned)

This matters because OTM call strikes > underlying, so using strike for CCs would understate the true return on equity.

---

## Response Variables (5 total)

Each candidate option is simulated from entry date to expiry. Every trading day's mid price is checked to compute:

| # | Variable | Type | Description |
|---|----------|------|-------------|
| 1 | `hit_50pct` | binary | Did premium decay to ≤50% of open within the DTE window? |
| 2 | `days_to_50pct` | int (nullable) | Days until the 50% profit threshold was first hit |
| 3 | `max_profit_pct` | float | `(open_mid - final_mid) / open_mid` at expiry |
| 4 | `expected_value` | float ($) | `(open_mid - final_mid) × 100` — actual dollar P&L |
| 5 | `outcome` | multi-class | `full_win` (≥80%), `partial_win` (20-80%), `breakeven` (-20-20%), `loss` (<-20%) |

**Training data distribution:**
- hit_50pct rate: **88.0%** (options with DTE 21-60, delta 0.10-0.40 usually hit 50%)
- Median days to 50%: **6 days**
- full_win: 76.1%, loss: 17.7%, partial_win: 3.9%, breakeven: 2.2%

---

## Models

All 5 models use the same 19-feature input. Each has its own XGBoost tree ensemble.

### Shared hyperparameters

```python
n_estimators=200, max_depth=6, learning_rate=0.1,
subsample=0.8, colsample_bytree=0.8, random_state=42
```

### Model 1: `hit_50pct` (Binary Classifier)

**Question**: Will this trade reach 50% profit before expiry?

**Balanced with `scale_pos_weight`** = `n_neg / n_pos` ≈ 0.137 (since 88% of training samples are positive).

**Performance** (20% holdout, V6 — latest):
| Metric | V3 | V4 | V5 | **V6 (current)** |
|--------|----|----|----|----|
| Accuracy | 92.7% | 86.4% | 86.8% | **92.1%** |
| Precision | 98.9% | 98.95% | 98.97% | **99.1%** |
| Recall | 85.0% | 85.4% | 85.9% | **91.6%** |
| F1 | 91.5% | — | — | **95.2%** |
| AUC-ROC | 0.959 | 0.961 | 0.962 | **0.982** |

**Key V6 improvement**: Changing `days_to_earnings` from sentinel value (999) to `NaN` (native XGBoost missing-value handling) yielded +2pp AUC and +5.7pp recall.

**Top features (by importance):**
1. annualized_return
2. moneyness
3. max_loss
4. delta
5. ivr
6. rsi14

### Model 2: `max_profit_pct` (Regressor)

**Question**: What % of max profit at expiry?

**Performance (V6):**
- MAE: **0.47** (improved from 0.89 in V3)
- R²: **0.70** (improved from 0.54 in V4)

**Top features:** credit_received, ivr, annualized_return, macd_signal

### Model 3: `days_to_50pct` (Regressor, hit-only subset)

**Question**: How many days to reach 50% profit?

Only trained on rows where `hit_50pct == True`.

**Performance (V6):**
- MAE: **2.8 days** (improved from 3.5 in V3)
- R²: **0.63** (improved from 0.48 in V5)

**Top features:** dte, max_loss, ivr, iv, moneyness

### Model 4: `expected_value` (Regressor, dollars)

**Question**: How many dollars will I make/lose?

Target clipped to 0.5th / 99.5th percentiles for stability.

**Performance (V6):**
- MAE: **$596** (improved from $1,311 in V3)
- R²: **0.87** (improved from 0.77 in V3)

**Top features:** iv, post_earnings (!), delta, max_loss, ivr

**Serving-side cap:** Predictions are clamped to physical bounds:
```python
ev_upper = credit_received  # max win = collected premium
ev_lower = -(max_loss - credit)  # max loss (naked short)
```
Without this cap, XGBoost can extrapolate beyond theoretical maxima.

### Model 5: `outcome` (Multi-class Classifier)

**Classes**: breakeven, full_win, loss, partial_win

**Balanced with inverse-frequency `sample_weight`** (boosts minority classes, especially `loss` and `partial_win`).

**Performance (V6):**
- Accuracy: **82.7%** (improved from 75.4% in V5)
- Loss recall: ~80% (improved from 74% in V5)

---

## Composite ML Score (0-100)

Weighted combination returned per prediction:

```
score_hit    = hit_50pct_probability × 100
score_profit = clip(max_profit_pct + 1) × 50        # maps [-1,1] → [0,100]
score_speed  = clip(1 - days_to_50pct / 45) × 100   # faster = higher

ml_score = 0.50 × score_hit + 0.30 × score_profit + 0.20 × score_speed
```

---

## Threshold Tuning

The `hit_50pct` classifier supports user-adjustable confidence thresholds. Defaults and recommendations:

| Threshold | N Selected | Precision | Recall | Avg EV/trade | Use case |
|-----------|-----------|-----------|--------|--------------|----------|
| 0.50 | 258K | 98.9% | 85.0% | $1,613 | Max volume |
| **0.85** | **143K** | **99.95%** | **47.6%** | **$2,191** | **Default (balanced)** |
| 0.90 | 113K | 99.98% | 37.7% | $2,357 | Conservative |
| 0.95 | 70K | 99.99% | 23.4% | $2,646 | Very selective |
| 0.99 | 15K | 100.0% | 5.1% | $3,464 | Only highest-conviction |

---

## Profile-Based Strategies

The recommendations route applies different logic per profile:

### Conservative
- Effective threshold: `max(0.95, user_slider)` (stricter)
- Requires predicted_outcome == `full_win`
- Requires predicted_ev > 0
- **Sorted by hit_probability descending** (safest first)

### Balanced (default)
- Effective threshold: user_slider (default 0.85)
- No outcome filter
- **Sorted by ml_score** (composite)

### Aggressive
- Effective threshold: `min(0.75, user_slider)` (looser)
- Excludes predicted_outcome == `loss`
- **Sorted by predicted_ev descending** (dollar-maximizing)

---

## SHAP Explanations

For every prediction, the UI can request a SHAP-based explanation via `POST /explain`.

**Algorithm**: TreeSHAP (exact, not sampled) on the `hit_50pct` model.

**Output** (`ExplanationResponse`):
```typescript
{
  hit_50pct_probability: number,
  base_value: number,                    // model's prior (average prediction)
  top_factors: Array<{
    feature: string,
    value: number,
    shap_value: number,                  // contribution to this prediction
    impact: "positive" | "negative",
    direction_text: string,              // human-readable explanation
  }>,
  natural_language: string,              // e.g., "The model predicts a 97% chance..."
}
```

**Example output:**
> "The model predicts a 97.1% chance of reaching 50% profit.
> ✅ Favorable: IV at 45% boosts the odds, moneyness of 0.94 boosts success chance, earnings 45 days away boosts.
> ⚠️ Unfavorable: IVR at 65 reduces the odds."

The SHAP bar chart in the UI shows top 6 factors ranked by `|shap_value|`, colored green (positive) or red (negative).

---

## Data Caching

| Cache | Location | TTL | Purpose |
|-------|----------|-----|---------|
| Option chain | `market_cache` table (SQLite) | 15 min | Avoid repeated Public.com calls |
| Underlying price | `market_cache` table | 15 min | Same |
| IV history | `iv_history` table (SQLite) | N/A (append-only) | Compute IVR / IVP over 252 days |
| Simulation cache | `cache/sim_{ticker}.pkl` | Until retrained | Skip trade simulation during pipeline reruns |
| Earnings dates | In-memory (Python) | 24h | yfinance rate limits |

---

## Performance Observations

### Rebuild times
- Full training data rebuild (no cache): **~10 minutes**
- With simulation cache: **~1 minute**
- Model training (all 5): **~2 minutes**

### Simulation throughput
Python vectorized: **~3,500 trades/second** per ticker.

### Live scoring latency (per request)
- Single predict: ~30ms (5 model inference + SHAP on demand)
- Batch of 100: ~200ms

---

## Known Limitations

1. **Training universe narrow**: 6 underlyings only (AAPL, NVDA, QQQ, SPX, SPY, TSLA). Model may generalize poorly to low-float or meme stocks.

2. **VIX excluded**: Extreme moneyness (up to 9.4x) distorted feature space. Safe to skip since users don't trade VIX options.

3. **EV model extrapolation**: XGBoost regressors aren't bounded. Predictions are clipped to physical max profit / max loss at serving time.

4. **IVR/IVP cold start**: `iv_history` table accumulates during app usage. First day after install shows "—" for IVR/IVP. Full 252-day rank takes ~1 year of usage.

5. **Earnings for ETFs**: SPY, QQQ, SPX have no earnings. `days_to_earnings = NaN` (XGBoost handles natively via learned default direction per split). Previously used sentinel 999 which caused false-positive SHAP shifts.

6. **Class imbalance artifacts**: The `hit_50pct` classifier's 0.5 default threshold was tuned to 0.85 after class balancing shifted recall/precision tradeoff.

---

## File Map

```
scripts/
  build_training_data.py      # OptionsDX → training CSV
  train_model.py              # CSV → 5 XGBoost JSON models
  serve_model.py              # FastAPI inference server
  threshold_tuning.py         # Analyze precision/recall at each threshold
  analyze_shap.py             # Generate SHAP plots (dependence, beeswarm, bar)

models/
  hit50_model.json            # Classifier: will it hit 50%?
  maxprofit_model.json        # Regressor: final % of max profit
  days50_model.json           # Regressor: days to 50% profit
  ev_model.json               # Regressor: dollar P&L
  outcome_model.json          # Multi-class: full_win/partial_win/breakeven/loss
  model_meta.json             # Feature list, training rows, tickers

app/api/recommendations/
  route.ts                    # Main endpoint; fetches chain, calls ML, filters, sorts
  explain/route.ts            # Proxies to ML /explain

app/recommendations/page.tsx  # UI with profile/DTE/delta filters + SHAP panel
```

---

## Retraining

To incorporate new data or features:

```bash
# 1. Rebuild training data (uses cache by default; add --force-resim to redo simulations)
python3 scripts/build_training_data.py \
  "data/optionsdx" \
  --exclude VIX \
  --out "data/training.csv"

# 2. Retrain models
python3 scripts/train_model.py \
  "data/training.csv" \
  --out ./models

# 3. Restart ML server
lsof -ti:8001 | xargs kill
python3 scripts/serve_model.py --models ./models --port 8001 &
```

Analyze model explanations with SHAP:

```bash
python3 scripts/analyze_shap.py \
  --data "data/training.csv" \
  --model ./models/hit50_model.json \
  --classifier \
  --out ./shap_output \
  --sample 15000
```

Outputs: `shap_bar.png`, `shap_summary.png` (beeswarm), `shap_dependence.png` (top 6 features).

Validate training data quality:

```bash
python3 scripts/validate_training_data.py \
  "data/training.csv"
```

Inspect random samples with SHAP across all 5 models:

```bash
python3 scripts/inspect_all_models.py \
  "data/training.csv" \
  --n 5
```

Backtest on held-out period:

```bash
python3 scripts/backtest.py \
  "data/training.csv" \
  --threshold 0.85 \
  --start 2023-07-01 \
  --end 2023-12-28
```

---

## Backtest Results (H2 2023)

Tested on the last 6 months of training data (2023-07-01 to 2023-12-28) using V6 model with threshold=0.85:

| | Model Picks | Baseline (all) |
|---|---|---|
| N trades | 11,543 | 28,398 |
| Hit rate (actual) | **99.9%** | 87.0% |
| Win rate | **97.4%** | — |
| Avg P&L / trade | **$734** | $468 |
| Total P&L | $8.48M | $13.28M |
| Return on credit | **93.1%** | 65.0% |
| **Lift vs baseline** | **+57.1%** per trade | — |

Monthly P&L (all positive):
- Jul 2023: $2.06M (2,617 trades)
- Aug 2023: $3.16M (3,677 trades)
- Sep 2023: $595K (1,397 trades)
- Oct 2023: $1.15M (1,817 trades)
- Nov 2023: $1.43M (1,850 trades)
- Dec 2023: $82K (185 trades)

---

## Version History

| Version | Date | Changes | Hit50 AUC |
|---------|------|---------|-----------|
| V1 | 2026-04-15 | Initial: 15 features, 1.79M rows | 0.947 |
| V2 | 2026-04-15 | +IVP, +earnings features (19 features) | 0.960 |
| V3 | 2026-04-15 | VIX excluded, CC annualized_return fixed | 0.961 |
| V4 | 2026-04-15 | Outlier cleanup (IV, vega, moneyness, max_profit_pct clipped) | 0.961 |
| V5 | 2026-04-16 | Asymmetric resistance/support (CC vs CSP), Wilder RSI, population BB std | 0.962 |
| **V6** | **2026-04-16** | **`days_to_earnings` NaN instead of 999 sentinel** | **0.982** |

Key lessons:
- V6 NaN fix was the single biggest improvement (+2pp AUC, +14pp EV R²)
- VIX exclusion prevented extreme moneyness distortion
- Asymmetric resistance had modest impact but is logically correct
- Outlier clipping improved MAE significantly even when R² dropped (healthier distribution)

---

## Additional Scripts

```
scripts/
  validate_training_data.py   # Data quality checks: ranges, distributions, correlations, temporal consistency
  inspect_random.py           # N random trades with SHAP explanation (single model)
  inspect_all_models.py       # N random trades with SHAP across all 5 models
  backtest.py                 # Historical P&L simulation with model picks vs baseline
```

## Integration Points

### Next.js → ML Server

```
POST /predict          → single candidate scoring
POST /predict/batch    → bulk scoring (used by /api/recommendations)
POST /explain          → SHAP TreeExplainer for hit_50pct model
GET  /earnings/{ticker} → yfinance earnings dates (24h cache)
GET  /prices/{ticker}   → yfinance daily OHLCV (1h cache, saved to daily_close)
GET  /health           → server status + loaded models
GET  /model-info       → feature importances + training metadata
```

### Kelly Criterion Integration

Position sizing is computed per recommendation using:
```
f* = (p × b - q) / b × kelly_fraction

p = hit_50pct_probability (from ML)
b = avg_win / avg_loss = 0.50 / 1.50 = 0.333
kelly_fraction = 0.5 (half-Kelly, conservative)
max_position = 5% of portfolio NLV
```

### Position Monitor

`/api/positions/monitor` evaluates each open short option position:
- Fetches live Greeks from Public.com option chain
- Runs ML prediction per position
- Generates alerts: PROFIT_50, PROFIT_85, DTE_LOW, LOSS_50, ML_BEARISH, ITM
- Recommends action: hold / close_50 / close_profit / roll / danger
