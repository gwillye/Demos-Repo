"""
Market Basket Analysis - demo
-----------------------------
Builds a seeded retail transaction log (shopping baskets) and mines
**association rules** (support / confidence / lift) with a lightweight Apriori
over item pairs - no heavy dependencies.

Real-world context: "customers who buy X also buy Y" powers cross-sell,
product placement, bundle offers and recommendation. The classic finding -
{diapers} -> {beer} - is the textbook example; here we plant a few real
associations in the data and show the analysis recovers them, ranked by lift.

Data is SYNTHETIC (seeded). To use real data, pass `mine_rules()` a list of
baskets (each a set/list of item names).

Run:  python market_basket_demo.py
Out:  results/RESULTS.md, results/cooccurrence.png
"""
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(3)
N = 6000
ITEMS = ["bread", "butter", "milk", "eggs", "coffee", "sugar", "beer", "chips",
         "diapers", "soda", "cheese", "wine", "pasta", "tomato_sauce", "apples"]
# Planted associations (co-purchase boosts) the miner should recover.
ASSOC = [("bread", "butter"), ("coffee", "sugar"), ("beer", "chips"),
         ("diapers", "beer"), ("pasta", "tomato_sauce"), ("wine", "cheese")]


def make_baskets(n):
    baskets = []
    for _ in range(n):
        k = int(RNG.integers(2, 7))
        basket = set(RNG.choice(ITEMS, size=k, replace=False).tolist())
        for a, b in ASSOC:
            if a in basket and RNG.random() < 0.60:
                basket.add(b)
            if b in basket and RNG.random() < 0.35:
                basket.add(a)
        baskets.append(basket)
    return baskets


def mine_rules(baskets, min_support=0.02, min_conf=0.30):
    n = len(baskets)
    item_count = {it: 0 for it in ITEMS}
    pair_count = {}
    for b in baskets:
        for it in b:
            item_count[it] = item_count.get(it, 0) + 1
        for x, y in combinations(sorted(b), 2):
            pair_count[(x, y)] = pair_count.get((x, y), 0) + 1
    support = {it: c / n for it, c in item_count.items()}
    rules = []
    for (x, y), c in pair_count.items():
        sup_xy = c / n
        if sup_xy < min_support:
            continue
        for a, b in [(x, y), (y, x)]:
            conf = sup_xy / support[a]
            lift = conf / support[b]
            if conf >= min_conf:
                rules.append({"antecedent": a, "consequent": b,
                              "support": round(sup_xy, 3),
                              "confidence": round(conf, 3),
                              "lift": round(lift, 2)})
    return pd.DataFrame(rules).sort_values("lift", ascending=False), support, pair_count


def main():
    baskets = make_baskets(N)
    rules, support, pair_count = mine_rules(baskets)

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    top = rules.head(12).to_string(index=False)
    md = ["# Market Basket Analysis - Results\n\n",
          f"{N:,} synthetic baskets over {len(ITEMS)} products.\n\n",
          "## Top association rules (by lift)\n\n```\n", top, "\n```\n\n",
          "Lift > 1 means the two items are bought together more than chance would "
          "predict. The planted pairs (bread+butter, coffee+sugar, beer+chips, "
          "diapers+beer, pasta+tomato_sauce, wine+cheese) surface at the top.\n"]
    (out / "RESULTS.md").write_text("".join(md), encoding="utf-8")

    # co-occurrence (lift) heatmap
    idx = {it: i for i, it in enumerate(ITEMS)}
    M = np.ones((len(ITEMS), len(ITEMS)))
    for (x, y), c in pair_count.items():
        sup_xy = c / N
        lift = sup_xy / (support[x] * support[y]) if support[x] and support[y] else 0
        M[idx[x], idx[y]] = M[idx[y], idx[x]] = lift
    np.fill_diagonal(M, np.nan)
    plt.figure(figsize=(7.5, 6.5))
    im = plt.imshow(M, cmap="YlOrRd")
    plt.colorbar(im, label="lift")
    plt.xticks(range(len(ITEMS)), ITEMS, rotation=90)
    plt.yticks(range(len(ITEMS)), ITEMS)
    plt.title("Item co-purchase lift")
    plt.tight_layout(); plt.savefig(out / "cooccurrence.png", dpi=110); plt.close()

    print(f"Done. {len(rules)} rules; top lift {rules['lift'].iloc[0]:.2f} "
          f"({rules['antecedent'].iloc[0]} -> {rules['consequent'].iloc[0]})")
    print("See results/RESULTS.md + results/cooccurrence.png")


if __name__ == "__main__":
    main()
