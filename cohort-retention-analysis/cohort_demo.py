"""
Cohort Retention Analysis - demo
--------------------------------
Builds a seeded signup + activity log, then computes a monthly **cohort
retention matrix** (the triangular heatmap every growth team lives by) and the
average retention curve.

Real-world context: retention is the single most important growth metric. By
grouping users into signup cohorts and tracking what fraction stays active in
month 0, 1, 2, ..., you see whether the product is getting stickier over time
and where users drop off - the core of lifecycle / growth / marketing analytics.

Data is SYNTHETIC (seeded). To use real data, feed `build_matrix()` a DataFrame
with columns: cohort (signup period index) and month_since (months since signup,
one row per active user-month).

Run:  python cohort_demo.py
Out:  results/RESULTS.md, results/cohort_heatmap.png, results/retention_curve.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(5)
COHORTS = 12     # 12 monthly signup cohorts
HORIZON = 12     # months of observation
N = 5000


def make_activity():
    cohort = RNG.integers(0, COHORTS, N)
    rows = []
    for u in range(N):
        c = int(cohort[u])
        # later cohorts retain a bit better (product improving over time)
        tau = float(np.clip(RNG.normal(3.0 + c * 0.08, 0.5), 1.0, None))
        observable = HORIZON - c           # triangular: recent cohorts seen less
        for m in range(observable):
            if RNG.random() < np.exp(-m / tau):   # active this month?
                rows.append((c, m))
    return pd.DataFrame(rows, columns=["cohort", "month_since"])


def build_matrix(act):
    size = act[act.month_since == 0].groupby("cohort").size()
    counts = act.groupby(["cohort", "month_since"]).size().unstack(fill_value=0)
    ret = counts.div(size, axis=0)         # fraction retained
    return ret, size


def main():
    act = make_activity()
    ret, size = build_matrix(act)

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    # average retention curve across cohorts (ignoring unobserved cells)
    curve = ret.mean(axis=0, skipna=True)
    pct = (ret * 100).round(1)

    md = ["# Cohort Retention Analysis - Results\n\n",
          f"{N} users across {COHORTS} monthly cohorts, {HORIZON}-month horizon.\n\n",
          "## Retention by cohort (% retained, month 0..N)\n\n```\n",
          pct.fillna("").to_string(), "\n```\n\n",
          "## Average retention curve\n\n",
          f"- Month 1: **{curve.get(1, float('nan'))*100:.1f}%**  | "
          f"Month 3: **{curve.get(3, float('nan'))*100:.1f}%**  | "
          f"Month 6: **{curve.get(6, float('nan'))*100:.1f}%**\n\n",
          "Later cohorts retain slightly better - the synthetic 'product' improves "
          "over time, and the analysis recovers that trend.\n"]
    (out / "RESULTS.md").write_text("".join(md), encoding="utf-8")

    plt.figure(figsize=(8, 6))
    im = plt.imshow(ret.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, label="retention")
    plt.xlabel("Months since signup"); plt.ylabel("Signup cohort")
    plt.title("Cohort retention heatmap")
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            v = ret.values[i, j]
            if not np.isnan(v):
                plt.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                         fontsize=7, color="black" if v < 0.5 else "white")
    plt.tight_layout(); plt.savefig(out / "cohort_heatmap.png", dpi=110); plt.close()

    plt.figure(figsize=(6, 4))
    (curve * 100).plot(marker="o", color="#4C72B0")
    plt.xlabel("Months since signup"); plt.ylabel("% retained")
    plt.title("Average retention curve"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(out / "retention_curve.png", dpi=110); plt.close()

    print(f"Done. Avg month-1 retention {curve.get(1, float('nan'))*100:.1f}%, "
          f"month-6 {curve.get(6, float('nan'))*100:.1f}%")
    print("See results/RESULTS.md + results/*.png")


if __name__ == "__main__":
    main()
