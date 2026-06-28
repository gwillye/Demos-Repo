# Sales / Demand Forecasting 📈

Daily **sales forecasting** with **Gradient Boosting**, benchmarked against a strong
**seasonal-naive** baseline — and an interactive **Plotly** forecast chart (GitHub-Pages ready).

👉 **Live chart:** open `forecast.html`.

## Result (held-out 90-day test)
| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| **Gradient Boosting** | **41.0** | **48.2** | **8.4%** |
| Seasonal-naive (last week) | 68.1 | 96.7 | 13.5% |

The ML model cuts error **~38%** vs the naive baseline. Why? The series has scheduled
**payday promos** (5th/20th): the model learns them from a `promo` calendar feature
(2nd-most-important), while a last-week baseline simply can't anticipate them. A nice
illustration that **ML earns its keep when there's structure a naive rule misses** — and,
honestly, *not before* (without the promo signal the baseline is very hard to beat).

## How it works
`forecast.py`:
1. **Generates** a plausible daily series (trend + weekly/yearly seasonality + payday promos + noise).
2. **Engineers features** — calendar, **Fourier** seasonality (sin/cos), **lags** (1/7/14/364), rolling means, promo flag.
3. **Trains** `GradientBoostingRegressor` (one-step-ahead) and evaluates MAE / RMSE / MAPE on a 90-day hold-out.
4. **Renders** `forecast.html` (actual vs forecast vs baseline).

## Run
```bash
pip install -r requirements.txt
python forecast.py
```

## Possible extensions
- Recursive **multi-step** forecasting + prediction intervals (quantile loss).
- Swap in **LightGBM / XGBoost**; add holiday calendars; cross-validate with rolling origin.
- Real data (retail/energy) instead of the synthetic generator.

> Data Science / time-series portfolio piece. Synthetic data (deterministic seed); no client data.
