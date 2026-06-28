# Time-Series Forecasting (seasonal-trend model)

A self-contained **forecasting demo**: it fits trend + weekly/yearly seasonality to a daily series and forecasts a held-out tail, reporting MAPE/MAE.

## Real-world context
Demand, traffic and revenue series are dominated by trend + seasonality. A transparent seasonal-trend regression is a strong, explainable baseline before reaching for heavier models (Prophet/ARIMA/LSTM).

## What it does
1. Generates a seeded daily series (trend + weekly + yearly seasonality + noise).
2. Builds **Fourier seasonal features** (weekly & yearly) plus a linear trend.
3. Fits a linear regression on the first 80% and **forecasts the last 20%**.
4. Reports **MAPE / MAE** and plots train / actual / forecast.

## Results (reproducible, seed = 42)
See [`results/RESULTS.md`](results/RESULTS.md) and `results/forecast.png`.

## Run
```bash
pip install -r requirements.txt
python forecast_demo.py
```

## Stack
Python · scikit-learn · NumPy · matplotlib
