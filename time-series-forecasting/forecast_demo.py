"""Time-series forecasting demo — seasonal-trend linear model on a synthetic daily series.

Self-contained (numpy + scikit-learn + matplotlib), seeded. Fits trend + weekly/yearly
seasonality, forecasts a held-out tail, reports MAPE/MAE. Saves results/RESULTS.md +
results/forecast.png.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

SEED = 42
np.random.seed(SEED)


def make_series(n=730):
    t = np.arange(n)
    trend = 0.04 * t
    weekly = 8 * np.sin(2 * np.pi * t / 7)
    yearly = 20 * np.sin(2 * np.pi * t / 365)
    noise = np.random.normal(0, 3, n)
    return 100 + trend + weekly + yearly + noise


def features(t):
    return np.column_stack([
        t,
        np.sin(2 * np.pi * t / 7), np.cos(2 * np.pi * t / 7),
        np.sin(2 * np.pi * t / 365), np.cos(2 * np.pi * t / 365),
    ])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(here, "results")
    os.makedirs(res, exist_ok=True)

    n = 730
    t = np.arange(n)
    y = make_series(n)
    split = int(n * 0.8)

    model = LinearRegression().fit(features(t[:split]), y[:split])
    pred = model.predict(features(t[split:]))

    mape = mean_absolute_percentage_error(y[split:], pred)
    mae = mean_absolute_error(y[split:], pred)

    plt.figure(figsize=(10, 4))
    plt.plot(t[:split], y[:split], color="#4c78a8", label="train")
    plt.plot(t[split:], y[split:], color="#222", label="actual (test)")
    plt.plot(t[split:], pred, color="#e45756", ls="--", label="forecast")
    plt.axvline(split, color="gray", ls=":")
    plt.title(f"Seasonal-trend forecast — MAPE {mape:.1%}, MAE {mae:.2f}")
    plt.xlabel("day")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(res, "forecast.png"), dpi=110)
    plt.close()

    with open(os.path.join(res, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("# Time-Series Forecasting — Results\n\n")
        f.write(f"Synthetic daily series: **{n} days** (trend + weekly + yearly seasonality + noise). "
                f"Model: linear regression on trend + Fourier seasonal terms. 80/20 holdout.\n\n")
        f.write(f"- **MAPE: {mape:.1%}**\n")
        f.write(f"- **MAE: {mae:.2f}**\n\n")
        f.write("See `forecast.png` (train / actual / forecast).\n")

    print(f"MAPE={mape:.1%} MAE={mae:.2f}")
    print("self-check:", "OK" if mape < 0.10 else "FAIL")


if __name__ == "__main__":
    main()
