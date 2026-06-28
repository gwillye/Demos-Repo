"""
RFM Customer Segmentation - demo
--------------------------------
Builds a seeded e-commerce transaction log, computes RFM (Recency, Frequency,
Monetary) per customer, assigns marketing segments via the classic RFM quintile
matrix, and cross-checks with KMeans clustering.

Real-world context: RFM segmentation is the workhorse of CRM / retail marketing.
It turns a raw order history into actionable groups - "Champions" to reward,
"At Risk" to win back, "Hibernating" to re-activate - so campaigns target the
right people instead of blasting everyone.

Data is SYNTHETIC (seeded). To use real data, feed `rfm_demo` a transactions
DataFrame with columns: customer_id, order_date (datetime/date), amount.

Run:  python rfm_demo.py
Out:  results/RESULTS.md, results/segments.png, results/rfm_scatter.png,
      results/segments.html (interactive)
"""
from pathlib import Path
from datetime import date, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RNG = np.random.default_rng(7)
N = 1500
REF = date(2025, 1, 1)   # fixed "today" for reproducibility


def make_transactions(n):
    rows = []
    for cid in range(n):
        arch = RNG.choice([0, 1, 2], p=[.35, .40, .25])  # loyal / occasional / lapsed
        if arch == 0:
            orders, last_gap, spend = RNG.integers(8, 25), RNG.integers(1, 40), RNG.normal(180, 60)
        elif arch == 1:
            orders, last_gap, spend = RNG.integers(2, 8), RNG.integers(20, 160), RNG.normal(110, 40)
        else:
            orders, last_gap, spend = RNG.integers(1, 4), RNG.integers(150, 360), RNG.normal(80, 30)
        spend = max(20.0, float(spend))   # avg order value, kept positive
        last = REF - timedelta(days=int(last_gap))
        for _ in range(int(orders)):
            d = last - timedelta(days=int(RNG.integers(0, 300)))
            rows.append((cid, d, round(max(10.0, RNG.normal(spend, spend * 0.3)), 2)))
    return pd.DataFrame(rows, columns=["customer_id", "order_date", "amount"])


def compute_rfm(tx):
    g = tx.groupby("customer_id")
    rfm = pd.DataFrame({
        "recency": g["order_date"].max().apply(lambda d: (REF - d).days),
        "frequency": g.size(),
        "monetary": g["amount"].sum().round(2),
    })
    # quintile scores via rank (avoids duplicate-edge errors); R: recent = 5
    rfm["R"] = pd.qcut(rfm["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    return rfm


def segment(r):
    R, fm = r.R, (r.F + r.M) / 2
    if R >= 4 and fm >= 4:   return "Champions"
    if R >= 3 and fm >= 3:   return "Loyal"
    if R >= 4 and fm < 3:    return "New / Promising"
    if R == 3 and fm < 3:    return "Potential Loyalists"
    if R <= 2 and fm >= 3:   return "At Risk"
    if R <= 2 and fm <= 2:   return "Hibernating / Lost"
    return "Others"


def main():
    tx = make_transactions(N)
    rfm = compute_rfm(tx)
    rfm["segment"] = rfm.apply(segment, axis=1)

    X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
    rfm["cluster"] = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(X)

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    seg = rfm["segment"].value_counts()
    seg_profile = rfm.groupby("segment")[["recency", "frequency", "monetary"]].mean().round(1)
    clu_profile = rfm.groupby("cluster")[["recency", "frequency", "monetary"]].mean().round(1)

    lines = ["# RFM Customer Segmentation - Results\n\n",
             f"{len(tx):,} synthetic orders across **{N} customers** (ref date {REF}).\n\n",
             "## Segment sizes (RFM matrix)\n\n```\n", seg.to_string(), "\n```\n\n",
             "## Segment profiles (mean R/F/M)\n\n```\n", seg_profile.to_string(), "\n```\n\n",
             "## KMeans cluster profiles (mean R/F/M)\n\n```\n", clu_profile.to_string(), "\n```\n"]
    (out / "RESULTS.md").write_text("".join(lines), encoding="utf-8")

    plt.figure(figsize=(7, 4))
    seg.sort_values().plot.barh(color="#4C72B0")
    plt.title("Customers per RFM segment"); plt.xlabel("customers")
    plt.tight_layout(); plt.savefig(out / "segments.png", dpi=110); plt.close()

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(rfm["recency"], rfm["monetary"], c=rfm["cluster"],
                     s=rfm["frequency"] * 6, cmap="viridis", alpha=.6, edgecolor="none")
    plt.xlabel("Recency (days since last order)"); plt.ylabel("Monetary (total spend)")
    plt.title("RFM clusters (size = frequency)"); plt.colorbar(sc, label="cluster")
    plt.tight_layout(); plt.savefig(out / "rfm_scatter.png", dpi=110); plt.close()

    # interactive view (GitHub-Pages ready): segment sizes + RFM map coloured by segment
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#999999"]
    seg_sorted = seg.sort_values()
    fig = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6],
                        subplot_titles=("Customers per segment", "RFM map (size = frequency)"))
    fig.add_bar(x=seg_sorted.values, y=seg_sorted.index, orientation="h",
                marker_color="#4C72B0", showlegend=False, row=1, col=1)
    for i, (name, grp) in enumerate(rfm.groupby("segment")):
        fig.add_scatter(x=grp["recency"], y=grp["monetary"], mode="markers", name=name,
                        marker=dict(size=3 + grp["frequency"], opacity=.6,
                                    color=palette[i % len(palette)]),
                        hovertemplate=f"<b>{name}</b><br>recency=%{{x}}d<br>"
                                      "monetary=%{y:.0f}<extra></extra>", row=1, col=2)
    fig.update_xaxes(title="Recency (days)", row=1, col=2)
    fig.update_yaxes(title="Monetary (total spend)", row=1, col=2)
    fig.update_layout(template="plotly_white", height=470,
                      title=f"RFM segmentation - {N} customers, {len(seg)} segments",
                      legend=dict(title="segment", font=dict(size=9)))
    fig.write_html(out / "segments.html", include_plotlyjs="cdn")

    print("Done. Segments:", dict(seg))
    print("See results/RESULTS.md + results/*.png + results/segments.html")


if __name__ == "__main__":
    main()
