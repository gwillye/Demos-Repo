"""Marketing Mix Modeling (MMM) — channel contribution & ROI.

A classic growth/marketing-analytics problem: given weekly spend across
channels and the resulting sales, estimate **how much each channel contributed**
and its **ROI**, so budget can be reallocated to what actually works.

This demo generates realistic synthetic data (with **adstock** carryover and a
known ground truth), fits a linear MMM, and recovers each channel's contribution
and ROI — then a self-check confirms the model ranks the channels correctly and
exports an interactive Plotly chart.

Run:  python mmm.py   ->  prints results + writes mmm_contributions.html
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

CHANNELS = ["TV", "Search", "Social", "Email"]
DECAY = {"TV": 0.6, "Search": 0.2, "Social": 0.3, "Email": 0.1}      # adstock carryover
TRUE_BETA = {"TV": 3.0, "Search": 5.0, "Social": 2.0, "Email": 8.0}  # sales per adstocked unit
BASE_SALES = 5000.0
WEEKS = 160


def adstock(x: np.ndarray, decay: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for i, v in enumerate(x):
        carry = v + decay * carry
        out[i] = carry
    return out


def make_data(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    spend = {c: rng.gamma(shape=2.0, scale=500, size=WEEKS) for c in CHANNELS}
    df = pd.DataFrame(spend)
    sales = np.full(WEEKS, BASE_SALES)
    for c in CHANNELS:
        sales = sales + TRUE_BETA[c] * adstock(df[c].to_numpy(), DECAY[c])
    sales = sales + rng.normal(0, 2000, WEEKS)   # noise
    df["sales"] = sales
    return df


def fit_mmm(df: pd.DataFrame):
    X = pd.DataFrame({c: adstock(df[c].to_numpy(), DECAY[c]) for c in CHANNELS})
    y = df["sales"].to_numpy()
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    beta = dict(zip(CHANNELS, model.coef_))
    contribution = {c: beta[c] * X[c].sum() for c in CHANNELS}
    roi = {c: contribution[c] / df[c].sum() for c in CHANNELS}
    return beta, contribution, roi, r2


def main() -> None:
    df = make_data()
    beta, contribution, roi, r2 = fit_mmm(df)

    print(f"MMM fit: R2 = {r2:.3f}\n")
    print(f"{'channel':<8} {'est.beta':>9} {'true':>6} {'contribution':>14} {'ROI':>7}")
    for c in CHANNELS:
        print(f"{c:<8} {beta[c]:>9.2f} {TRUE_BETA[c]:>6.1f} {contribution[c]:>14,.0f} {roi[c]:>7.2f}")

    best_roi = max(roi, key=roi.get)
    print(f"\nBest ROI channel: {best_roi} (ROI {roi[best_roi]:.2f})")

    _write_chart(contribution, roi)
    _self_check(beta, roi, r2)


def _write_chart(contribution, roi) -> None:
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_bar(x=CHANNELS, y=[contribution[c] for c in CHANNELS], name="Contribution to sales")
    fig.add_scatter(x=CHANNELS, y=[roi[c] for c in CHANNELS], name="ROI", yaxis="y2",
                    mode="lines+markers")
    fig.update_layout(
        title="Marketing Mix Modeling — channel contribution & ROI",
        yaxis=dict(title="Contribution (sales)"),
        yaxis2=dict(title="ROI", overlaying="y", side="right"),
        template="plotly_white",
    )
    fig.write_html("mmm_contributions.html", include_plotlyjs="cdn")
    print("chart written: mmm_contributions.html")


def _self_check(beta, roi, r2) -> None:
    assert r2 > 0.90, f"fit too weak: R2={r2}"
    # recovered betas must rank the channels the same as ground truth
    rank_est = sorted(CHANNELS, key=lambda c: beta[c])
    rank_true = sorted(CHANNELS, key=lambda c: TRUE_BETA[c])
    assert rank_est == rank_true, f"channel ranking off: {rank_est} vs {rank_true}"
    # Email has the highest unit-effect ground truth -> should show strong ROI
    assert max(roi, key=roi.get) in ("Email", "Search"), "implausible best-ROI channel"
    print("self-check: OK (R2>0.90, channel ranking recovered, ROI plausible)")


if __name__ == "__main__":
    main()
