#!/usr/bin/env python3
"""
FastAPI inference server for Options ML models.

Loads trained XGBoost models and serves predictions via REST API.
Next.js calls this from API routes for live scoring.

Usage:
  python3 scripts/serve_model.py --models ./models --port 8001

Endpoints:
  POST /predict      — score a single option candidate
  POST /predict/batch — score multiple candidates at once
  GET  /health       — health check
  GET  /model-info   — model metadata and feature importances
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Options ML Scoring", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model state ─────────────────────────────────────────────────────

models = {}
model_meta = {}
FEATURE_COLS = []


# ── Request/Response types ─────────────────────────────────────────────────

class OptionCandidate(BaseModel):
    dte: float
    delta: float
    gamma: float
    theta: float
    vega: float
    iv: float
    ivr: Optional[float] = 50.0
    ivp: Optional[float] = 50.0
    moneyness: float
    annualized_return: float
    credit_received: float
    max_loss: float
    rsi14: Optional[float] = 50.0
    macd_signal: Optional[float] = 0.0
    bb_position: Optional[float] = 0.5
    resistance_score: Optional[float] = 0.5
    days_to_earnings: Optional[float] = None  # None → NaN (XGBoost handles missing)
    earnings_in_window: Optional[int] = 0
    post_earnings: Optional[int] = 0


class PredictionResult(BaseModel):
    hit_50pct_probability: float
    predicted_max_profit_pct: float
    predicted_days_to_50pct: Optional[float]
    predicted_ev_dollars: Optional[float]   # dollar EV from model 4
    predicted_outcome: Optional[str]        # full_win/partial_win/breakeven/loss
    outcome_probabilities: Optional[dict]   # probabilities per outcome class
    ml_score: float
    confidence: float
    risk_reward: str
    passes_threshold: bool                  # true if hit50_prob >= 0.85


class BatchRequest(BaseModel):
    candidates: list[OptionCandidate]


class BatchResponse(BaseModel):
    predictions: list[PredictionResult]


# ── Prediction logic ──────────────────────────────────────────────────────

OUTCOME_CLASSES = ['breakeven', 'full_win', 'loss', 'partial_win']
HIT_THRESHOLD = 0.85  # minimum probability to "pass"


def features_to_array(c: OptionCandidate) -> np.ndarray:
    return np.array([[
        c.dte, c.delta, c.gamma, c.theta, c.vega, c.iv,
        c.ivr or 50, c.ivp or 50, c.moneyness, c.annualized_return,
        c.credit_received, c.max_loss,
        c.rsi14 or 50, c.macd_signal or 0, c.bb_position or 0.5,
        c.resistance_score or 0.5,
        c.days_to_earnings if c.days_to_earnings is not None else np.nan,
        c.earnings_in_window or 0,
        c.post_earnings or 0,
    ]])


def predict_single(c: OptionCandidate) -> PredictionResult:
    X = features_to_array(c)

    # Model 1: P(hit 50% profit)
    hit50_prob = 0.5
    if 'hit50' in models:
        hit50_prob = float(models['hit50'].predict_proba(X)[0][1])

    # Model 2: Expected max profit %
    max_profit = 0.0
    if 'maxprofit' in models:
        max_profit = float(models['maxprofit'].predict(X)[0])

    # Model 3: Days to 50%
    days_50 = None
    if 'days50' in models and hit50_prob > 0.5:
        days_50 = max(1, float(models['days50'].predict(X)[0]))

    # Model 4: Expected value in $
    ev_dollars = None
    if 'ev' in models:
        ev_raw = float(models['ev'].predict(X)[0])
        # Cap at physical bounds: max profit = credit_received, max loss = -(max_loss - credit)
        ev_upper = c.credit_received  # can't win more than the credit collected
        ev_lower = -(c.max_loss - c.credit_received)  # max loss for short options
        ev_dollars = max(ev_lower, min(ev_upper, ev_raw))

    # Model 5: Outcome multi-class
    pred_outcome = None
    outcome_probs = None
    if 'outcome' in models:
        probs = models['outcome'].predict_proba(X)[0]
        outcome_probs = {OUTCOME_CLASSES[i]: round(float(probs[i]), 3) for i in range(len(OUTCOME_CLASSES))}
        pred_outcome = OUTCOME_CLASSES[int(np.argmax(probs))]

    # Composite ML score (0-100)
    score_hit = hit50_prob * 100
    score_profit = max(0, min(100, (max_profit + 1) * 50))
    score_speed = 0
    if days_50 is not None:
        score_speed = max(0, min(100, (1 - days_50 / 45) * 100))

    ml_score = 0.50 * score_hit + 0.30 * score_profit + 0.20 * score_speed
    ml_score = max(0, min(100, ml_score))

    # Threshold pass (tuned to 0.85 from threshold analysis → ~99.95% precision)
    passes = hit50_prob >= HIT_THRESHOLD

    confidence = abs(hit50_prob - 0.5) * 2

    if hit50_prob >= HIT_THRESHOLD and max_profit > 0.3:
        risk_reward = "favorable"
    elif hit50_prob < 0.4 or max_profit < -0.2 or pred_outcome == 'loss':
        risk_reward = "unfavorable"
    else:
        risk_reward = "neutral"

    return PredictionResult(
        hit_50pct_probability=round(hit50_prob, 4),
        predicted_max_profit_pct=round(max_profit, 4),
        predicted_days_to_50pct=round(days_50, 1) if days_50 else None,
        predicted_ev_dollars=round(ev_dollars, 2) if ev_dollars is not None else None,
        predicted_outcome=pred_outcome,
        outcome_probabilities=outcome_probs,
        ml_score=round(ml_score, 1),
        confidence=round(confidence, 3),
        risk_reward=risk_reward,
        passes_threshold=passes,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
        "feature_columns": FEATURE_COLS,
    }


@app.get("/model-info")
def model_info():
    info = {**model_meta}

    # Add feature importances
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            imp = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
            info[f'{name}_importances'] = dict(sorted(imp.items(), key=lambda x: -x[1]))

    return info


@app.post("/predict", response_model=PredictionResult)
def predict(candidate: OptionCandidate):
    if not models:
        raise HTTPException(503, "No models loaded")
    return predict_single(candidate)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    if not models:
        raise HTTPException(503, "No models loaded")

    predictions = [predict_single(c) for c in req.candidates]
    return BatchResponse(predictions=predictions)


# ── SHAP Explanation endpoint ─────────────────────────────────────────────

shap_explainer = None


def get_shap_explainer():
    """Lazy-init TreeSHAP explainer for hit50 model."""
    global shap_explainer
    if shap_explainer is not None:
        return shap_explainer
    try:
        import shap
        if 'hit50' in models:
            shap_explainer = shap.TreeExplainer(models['hit50'])
            return shap_explainer
    except ImportError:
        return None
    return None


class ExplanationResponse(BaseModel):
    hit_50pct_probability: float
    base_value: float              # model's baseline prediction
    top_factors: list[dict]        # sorted by |shap| impact, with explanation
    natural_language: str          # human-readable summary


def explain_with_shap(c: OptionCandidate) -> ExplanationResponse:
    explainer = get_shap_explainer()
    if explainer is None:
        raise HTTPException(503, "SHAP not available — install `shap` package")

    X = features_to_array(c)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_row = shap_values[0]  # single sample

    base_value = float(explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray)
                       else explainer.expected_value[0])

    # Build per-feature factors
    feature_values = X[0]
    factors = []
    for i, feat in enumerate(FEATURE_COLS):
        factors.append({
            'feature': feat,
            'value': float(feature_values[i]),
            'shap_value': float(shap_row[i]),
            'impact': 'positive' if shap_row[i] > 0 else 'negative',
            'direction_text': _direction_text(feat, float(feature_values[i]), float(shap_row[i])),
        })

    factors.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    top6 = factors[:6]

    # Natural language summary from top 3
    hit_prob = float(models['hit50'].predict_proba(X)[0][1])
    nl_parts = [f"The model predicts a {hit_prob * 100:.1f}% chance of reaching 50% profit."]
    positive = [f for f in factors if f['shap_value'] > 0][:3]
    negative = [f for f in factors if f['shap_value'] < 0][:3]

    if positive:
        nl_parts.append("✅ Favorable: " + ", ".join(f["direction_text"] for f in positive) + ".")
    if negative:
        nl_parts.append("⚠️ Unfavorable: " + ", ".join(f["direction_text"] for f in negative) + ".")

    return ExplanationResponse(
        hit_50pct_probability=round(hit_prob, 4),
        base_value=round(base_value, 4),
        top_factors=top6,
        natural_language=" ".join(nl_parts),
    )


def _direction_text(feature: str, value: float, shap: float) -> str:
    """Human-readable phrase for each feature's contribution."""
    sign = "boosts" if shap > 0 else "reduces"
    descriptions = {
        'dte': f"DTE of {value:.0f} days {sign} hit probability",
        'delta': f"delta {value:.2f} {sign} the odds",
        'iv': f"IV at {value * 100:.0f}% {sign} the odds",
        'ivr': f"IVR at {value:.0f} {sign} the odds",
        'ivp': f"IVP at {value:.0f} {sign} the odds",
        'moneyness': f"moneyness of {value:.2f} {sign} success chance",
        'annualized_return': f"annualized return of {value * 100:.1f}% {sign} score",
        'rsi14': f"RSI {value:.0f} {sign} the outlook",
        'macd_signal': f"MACD signal ({int(value)}) {sign}",
        'bb_position': f"Bollinger position {value:.2f} {sign}",
        'resistance_score': f"resistance proximity ({value:.2f}) {sign}",
        'days_to_earnings': f"earnings {int(value)} days away {sign}",
        'earnings_in_window': f"earnings {'within' if value > 0 else 'not in'} window {sign}",
        'post_earnings': f"{'post-earnings' if value > 0 else 'not post-earnings'} {sign}",
        'max_loss': f"max loss ${value:,.0f} {sign}",
        'credit_received': f"credit ${value:.0f} {sign}",
        'gamma': f"gamma {value:.3f} {sign}",
        'theta': f"theta {value:.3f} {sign}",
        'vega': f"vega {value:.2f} {sign}",
    }
    return descriptions.get(feature, f"{feature}={value:.2f} {sign} prediction")


@app.post("/explain", response_model=ExplanationResponse)
def explain(candidate: OptionCandidate):
    """Return SHAP-based explanation of why the model gave this score."""
    if not models:
        raise HTTPException(503, "No models loaded")
    return explain_with_shap(candidate)


# ── Earnings dates endpoint ───────────────────────────────────────────────

_earnings_cache: dict[str, dict] = {}


@app.get("/earnings/{ticker}")
def earnings(ticker: str):
    """
    Return next/last earnings dates for a ticker via yfinance.
    Cached for 24h (earnings dates don't change often).
    """
    ticker = ticker.upper()
    cached = _earnings_cache.get(ticker)
    if cached and cached.get('cached_at', 0) > (os_time_now() - 86400):
        return cached

    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        raise HTTPException(503, "yfinance not installed")

    try:
        t = yf.Ticker(ticker)
        dates = t.earnings_dates
        if dates is None or len(dates) == 0:
            result = {
                'ticker': ticker,
                'next_earnings': None,
                'last_earnings': None,
                'days_to_earnings': None,
                'cached_at': os_time_now(),
            }
            _earnings_cache[ticker] = result
            return result

        # Find next and last
        now = pd.Timestamp.now(tz='UTC')
        dates_utc = [pd.Timestamp(d).tz_convert('UTC') if pd.Timestamp(d).tz is not None
                     else pd.Timestamp(d).tz_localize('UTC') for d in dates.index]

        future = [d for d in dates_utc if d >= now]
        past = [d for d in dates_utc if d < now]

        next_ed = min(future) if future else None
        last_ed = max(past) if past else None

        days_to = None
        if next_ed is not None:
            days_to = int((next_ed - now).total_seconds() / 86400)

        days_since = None
        if last_ed is not None:
            days_since = int((now - last_ed).total_seconds() / 86400)

        result = {
            'ticker': ticker,
            'next_earnings': next_ed.strftime('%Y-%m-%d') if next_ed is not None else None,
            'last_earnings': last_ed.strftime('%Y-%m-%d') if last_ed is not None else None,
            'days_to_earnings': days_to,
            'days_since_earnings': days_since,
            'cached_at': os_time_now(),
        }
        _earnings_cache[ticker] = result
        return result
    except Exception as e:
        result = {
            'ticker': ticker,
            'error': str(e),
            'next_earnings': None,
            'days_to_earnings': None,
            'cached_at': os_time_now(),
        }
        _earnings_cache[ticker] = result
        return result


def os_time_now():
    import time
    return time.time()


# ── Historical prices endpoint (yfinance) ─────────────────────────────────

_prices_cache: dict[str, dict] = {}


@app.get("/prices/{ticker}")
def prices(ticker: str, days: int = 90):
    """
    Return daily close prices from yfinance.
    Cached for 1h during market hours.
    """
    ticker = ticker.upper()
    cache_key = f"{ticker}_{days}"
    cached = _prices_cache.get(cache_key)
    if cached and cached.get('cached_at', 0) > (os_time_now() - 3600):
        return cached

    try:
        import yfinance as yf
    except ImportError:
        raise HTTPException(503, "yfinance not installed")

    try:
        t = yf.Ticker(ticker)
        # Fetch a bit more than requested so we have buffer for indicators needing lookback
        period = f"{max(days, 120)}d"
        hist = t.history(period=period, interval='1d')
        if hist is None or len(hist) == 0:
            raise HTTPException(404, f"No price data for {ticker}")

        closes = [float(c) for c in hist['Close'].tolist()]
        dates = [d.strftime('%Y-%m-%d') for d in hist.index]

        result = {
            'ticker': ticker,
            'days_requested': days,
            'days_returned': len(closes),
            'dates': dates,
            'closes': closes,
            'highs': [float(h) for h in hist['High'].tolist()],
            'lows': [float(l) for l in hist['Low'].tolist()],
            'volumes': [int(v) for v in hist['Volume'].tolist()],
            'cached_at': os_time_now(),
        }
        _prices_cache[cache_key] = result
        return result
    except Exception as e:
        raise HTTPException(500, f"yfinance error: {e}")


# ── Startup ────────────────────────────────────────────────────────────────

def load_models(model_dir: str):
    global models, model_meta, FEATURE_COLS

    model_dir = Path(model_dir)

    # Load metadata
    meta_path = model_dir / 'model_meta.json'
    if meta_path.exists():
        with open(meta_path) as f:
            model_meta = json.load(f)
            FEATURE_COLS = model_meta.get('feature_columns', [])

    # Load XGBoost models
    model_specs = [
        ('hit50', 'hit50_model.json', 'classifier'),
        ('maxprofit', 'maxprofit_model.json', 'regressor'),
        ('days50', 'days50_model.json', 'regressor'),
        ('ev', 'ev_model.json', 'regressor'),
        ('outcome', 'outcome_model.json', 'classifier'),
    ]
    for name, filename, kind in model_specs:
        path = model_dir / filename
        if path.exists():
            model = xgb.XGBClassifier() if kind == 'classifier' else xgb.XGBRegressor()
            model.load_model(str(path))
            models[name] = model
            print(f'  Loaded {name} from {path}')
        else:
            print(f'  {filename} not found, skipping')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='./models', help='Model directory')
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()

    print(f'Loading models from {args.models}...')
    load_models(args.models)
    print(f'Models loaded: {list(models.keys())}')
    print(f'Starting server on {args.host}:{args.port}')

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
