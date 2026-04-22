# options-ml-scoring

A 5-model XGBoost pipeline that scores covered calls and cash-secured puts using 19 features (Greeks, IV regime, technicals, earnings proximity).

Trained on 363K simulated option trades from OptionsDX historical EOD chains (2020-2023) across AAPL, NVDA, QQQ, SPX, SPY, TSLA.

**Read the full writeup:** [docs/blog-draft-ml-model.mdx](docs/blog-draft-ml-model.mdx)

## What the models predict

| Model | Question | Key Metric |
|-------|----------|------------|
| `hit50` | Will this short option reach 50% profit? | AUC 0.982 |
| `maxprofit` | What % of max profit at expiry? | R² 0.70 |
| `days50` | How many days to 50% profit? | MAE 2.8 days |
| `ev` | Expected dollar P&L? | R² 0.87 |
| `outcome` | Full win / partial / breakeven / loss? | Accuracy 82.7% |

## Important caveats

- Trained on **simulated trades using EOD mid-prices** — no slippage, no commissions, no bid-ask spread
- Narrow universe: 6 tickers (all mega-cap / index)
- Backtest numbers are directionally interesting but not a P&L you'd actually realize
- The 88% base win rate for delta-selected premium selling is already high — the model improves on an already-good baseline
- `annualized_return` is the top feature by SHAP importance and is derived from premium, which is closely related to the target. Not target leakage (it's known at entry), but worth noting.

See [Known Limitations](docs/blog-draft-ml-model.mdx#known-limitations) for more.

## Project structure

```
scripts/
  build_training_data.py   # OptionsDX → feature-engineered training CSV
  train_model.py           # Train all 5 XGBoost models
  backtest.py              # Historical P&L simulation
  threshold_tuning.py      # Precision/recall at each confidence threshold
  analyze_shap.py          # Generate SHAP visualizations
  inspect_random.py        # SHAP breakdown for random samples
  inspect_all_models.py    # SHAP across all 5 models
  validate_training_data.py # Data quality checks
  serve_model.py           # FastAPI server for real-time scoring
models/                    # Pre-trained XGBoost models (JSON)
shap_output/               # SHAP visualizations (PNG)
docs/                      # Technical docs + blog writeup
```

## Quick start

```bash
pip install -r requirements.txt

# Score a single option (using pre-trained models)
python scripts/serve_model.py
# → FastAPI server at http://localhost:8001
# → POST /score with 19 features → returns probability, EV, SHAP explanation

# Retrain from scratch (requires OptionsDX data)
python scripts/build_training_data.py --data-dir /path/to/optionsdx/ --out data/training.csv
python scripts/train_model.py --data data/training.csv --out models/
python scripts/analyze_shap.py --data data/training.csv --model-dir models/
```

## SHAP feature importance

![SHAP summary](shap_output/shap_summary.png)

Top features by mean |SHAP value|:
1. Annualized return
2. Moneyness (strike / underlying)
3. Max loss
4. Delta
5. IV Rank
6. RSI-14

## License

MIT
