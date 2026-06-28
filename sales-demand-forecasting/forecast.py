# -*- coding: utf-8 -*-
"""
forecast.py — Daily sales / demand forecasting (ML vs seasonal baseline).

Generates a plausible **synthetic** daily-sales series (trend + weekly + yearly
seasonality + promo spikes), engineers calendar/lag/rolling features, and forecasts
with **Gradient Boosting**, benchmarked against a **seasonal-naive** baseline.
Evaluates on a held-out test window (MAE / RMSE / MAPE) and renders an interactive
**Plotly** chart (`forecast.html`, GitHub-Pages ready).

Run:  python forecast.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go

TEST_DAYS = 90


def generate_sales(days: int = 730, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end="2025-12-31", periods=days, freq="D")
    t = np.arange(days)
    trend = 400 + 0.25 * t
    weekly = 60 * np.sin(2 * np.pi * (dates.dayofweek) / 7) + 40 * (dates.dayofweek >= 5)
    yearly = 120 * np.sin(2 * np.pi * (dates.dayofyear) / 365.25 - 1.3)
    # Promos on scheduled paydays (5th & 20th) — a KNOWN calendar a model can learn,
    # but a seasonal-naive (last-week) baseline cannot anticipate.
    promo_flag = dates.day.isin([5, 20]).astype(int)
    promo = promo_flag * rng.normal(240, 30, days)
    noise = rng.normal(0, 35, days)
    sales = np.clip(trend + weekly + yearly + promo + noise, 0, None)
    return pd.DataFrame({"date": dates, "sales": sales.round(0), "promo": promo_flag})


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["dow"] = d.date.dt.dayofweek
    d["month"] = d.date.dt.month
    d["doy"] = d.date.dt.dayofyear
    d["is_weekend"] = (d.dow >= 5).astype(int)
    d["t"] = np.arange(len(d))                       # linear time index (trend)
    # Fourier seasonal features (smooth weekly + yearly seasonality)
    d["dow_sin"], d["dow_cos"] = np.sin(2*np.pi*d.dow/7), np.cos(2*np.pi*d.dow/7)
    d["doy_sin"], d["doy_cos"] = np.sin(2*np.pi*d.doy/365.25), np.cos(2*np.pi*d.doy/365.25)
    for lag in (1, 7, 14, 364):
        d[f"lag{lag}"] = d.sales.shift(lag)
    d["roll7"] = d.sales.shift(1).rolling(7).mean()
    d["roll30"] = d.sales.shift(1).rolling(30).mean()
    return d.dropna().reset_index(drop=True)


def mape(y, p):
    y = np.asarray(y, float)
    return float(np.mean(np.abs((y - p) / np.clip(y, 1, None))) * 100)


def main():
    df = make_features(generate_sales())
    feats = [c for c in df.columns if c not in ("date", "sales")]
    train, test = df.iloc[:-TEST_DAYS], df.iloc[-TEST_DAYS:]
    # One-step-ahead GBR: lag-1 carries the current (trended) level; Fourier + calendar
    # carry the weekly/yearly seasonality; lag-7/14/364 carry autocorrelation.
    model = GradientBoostingRegressor(n_estimators=500, max_depth=3,
                                      learning_rate=0.04, subsample=0.9, random_state=0)
    model.fit(train[feats], train.sales)
    pred = model.predict(test[feats])
    baseline = test["lag7"].values  # seasonal-naive (same weekday last week)

    def report(name, y, p):
        rmse = mean_squared_error(y, p) ** 0.5
        print(f"  {name:<16} MAE={mean_absolute_error(y, p):6.1f}  RMSE={rmse:6.1f}  MAPE={mape(y, p):5.1f}%")
        return rmse

    print("Test window:", TEST_DAYS, "days")
    report("GradientBoosting", test.sales, pred)
    report("Seasonal-naive", test.sales, baseline)
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    print("Top features:", ", ".join(f"{k}={v:.2f}" for k, v in imp.head(5).items()))

    fig = go.Figure()
    fig.add_scatter(x=train.date.iloc[-120:], y=train.sales.iloc[-120:],
                    name="history (train)", line=dict(color="#9aa", width=1.5))
    fig.add_scatter(x=test.date, y=test.sales, name="actual (test)",
                    line=dict(color="#1f2a44", width=2.5))
    fig.add_scatter(x=test.date, y=pred, name="Gradient Boosting forecast",
                    line=dict(color="#00CC96", width=2.5, dash="dot"))
    fig.add_scatter(x=test.date, y=baseline, name="seasonal-naive baseline",
                    line=dict(color="#EF553B", width=1.5, dash="dash"))
    fig.update_layout(title="Daily sales forecast — Gradient Boosting vs seasonal baseline",
                      template="plotly_white", height=500, legend=dict(orientation="h"))
    fig.write_html("forecast.html", include_plotlyjs="cdn")
    print("OK -> forecast.html")


if __name__ == "__main__":
    main()
