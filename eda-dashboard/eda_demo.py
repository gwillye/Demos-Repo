"""EDA mini-dashboard — synthetic sales dataset → a 2x2 panel of charts + key stats.

Self-contained (pandas + numpy + matplotlib), seeded. Saves results/dashboard.png and
results/RESULTS.md, and prints a self-check.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)


def make_data(n=1500):
    cats = np.random.choice(
        ["Electronics", "Home", "Fashion", "Sports", "Books"],
        n, p=[0.30, 0.20, 0.25, 0.15, 0.10])
    price = np.round(np.random.gamma(2.0, 40, n) + 5, 2)
    units = np.random.poisson(3, n) + 1
    revenue = np.round(price * units, 2)
    satisfaction = np.clip(5 - price / 200 + np.random.normal(0, 0.6, n), 1, 5)
    month = np.random.randint(1, 13, n)
    return pd.DataFrame({
        "category": cats, "price": price, "units": units,
        "revenue": revenue, "satisfaction": satisfaction, "month": month,
    })


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(here, "results")
    os.makedirs(res, exist_ok=True)

    df = make_data()

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Sales — exploratory dashboard (synthetic data)", fontsize=14)

    # 1) revenue distribution
    ax[0, 0].hist(df["revenue"], bins=40, color="#4c78a8")
    ax[0, 0].set_title("Revenue distribution")
    ax[0, 0].set_xlabel("revenue")

    # 2) revenue by category
    by_cat = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    ax[0, 1].bar(by_cat.index, by_cat.values, color="#54a24b")
    ax[0, 1].set_title("Total revenue by category")
    ax[0, 1].tick_params(axis="x", rotation=30)

    # 3) correlation heatmap
    num = df[["price", "units", "revenue", "satisfaction"]].corr()
    im = ax[1, 0].imshow(num, cmap="coolwarm", vmin=-1, vmax=1)
    ax[1, 0].set_xticks(range(len(num)), num.columns, rotation=30)
    ax[1, 0].set_yticks(range(len(num)), num.columns)
    for i in range(len(num)):
        for j in range(len(num)):
            ax[1, 0].text(j, i, f"{num.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax[1, 0].set_title("Correlation matrix")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046)

    # 4) monthly revenue trend
    by_month = df.groupby("month")["revenue"].sum()
    ax[1, 1].plot(by_month.index, by_month.values, marker="o", color="#e45756")
    ax[1, 1].set_title("Monthly revenue trend")
    ax[1, 1].set_xlabel("month")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(res, "dashboard.png"), dpi=110)
    plt.close()

    corr_ps = df["price"].corr(df["satisfaction"])
    with open(os.path.join(res, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("# EDA Dashboard — Key stats\n\n")
        f.write(f"- Rows: **{len(df)}**\n")
        f.write(f"- Total revenue: **{df['revenue'].sum():,.0f}**\n")
        f.write(f"- Top category by revenue: **{by_cat.index[0]}** ({by_cat.iloc[0]:,.0f})\n")
        f.write(f"- Avg satisfaction: **{df['satisfaction'].mean():.2f}** / 5\n")
        f.write(f"- Corr(price, satisfaction): **{corr_ps:.2f}** (higher price → lower satisfaction)\n\n")
        f.write("See `dashboard.png` for the 2x2 panel.\n")

    print(f"rows={len(df)} top_cat={by_cat.index[0]} corr(price,sat)={corr_ps:.2f}")
    print("self-check:", "OK" if len(df) == 1500 and corr_ps < 0 else "FAIL")


if __name__ == "__main__":
    main()
