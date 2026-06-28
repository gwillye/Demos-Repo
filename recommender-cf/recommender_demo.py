"""Recommender demo — item-item collaborative filtering on a synthetic ratings matrix.

Self-contained (numpy + matplotlib), seeded. Builds a sparse ratings matrix with latent
structure, predicts held-out ratings via item-item cosine similarity, compares RMSE to a
baseline, and lists top-N recommendations. Saves results/RESULTS.md + results/rmse.png.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
rng = np.random.default_rng(SEED)


def make_ratings(n_users=200, n_items=60, k=4, observed=0.30):
    U = rng.normal(size=(n_users, k))
    V = rng.normal(size=(n_items, k))
    true = U @ V.T
    true = 1 + 4 * (true - true.min()) / (true.max() - true.min())
    true = np.clip(true + rng.normal(0, 0.3, true.shape), 1, 5)
    mask = rng.random((n_users, n_items)) < observed
    return true, mask


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(here, "results")
    os.makedirs(res, exist_ok=True)

    true, mask = make_ratings()
    obs = np.argwhere(mask)
    rng.shuffle(obs)
    n_test = int(0.2 * len(obs))
    test_idx = obs[:n_test]

    R = np.where(mask, true, np.nan)
    train = R.copy()
    for (u, i) in test_idx:
        train[u, i] = np.nan

    global_mean = np.nanmean(train)
    item_mean = np.nanmean(train, axis=0)
    item_mean = np.where(np.isnan(item_mean), global_mean, item_mean)

    centered = np.where(np.isnan(train), 0.0, train - item_mean)
    norms = np.linalg.norm(centered, axis=0)
    norms[norms == 0] = 1.0
    sim = (centered.T @ centered) / np.outer(norms, norms)
    np.fill_diagonal(sim, 0.0)

    def predict(u, i):
        rated = ~np.isnan(train[u])
        if rated.sum() == 0:
            return item_mean[i]
        s = sim[i, rated]
        r = train[u, rated] - item_mean[rated]
        denom = np.abs(s).sum()
        return item_mean[i] + (s @ r) / denom if denom > 0 else item_mean[i]

    preds = np.clip([predict(u, i) for (u, i) in test_idx], 1, 5)
    truth = np.array([true[u, i] for (u, i) in test_idx])
    base = np.array([item_mean[i] for (u, i) in test_idx])
    rmse_cf = np.sqrt(np.mean((preds - truth) ** 2))
    rmse_base = np.sqrt(np.mean((base - truth) ** 2))
    improve = (rmse_base - rmse_cf) / rmse_base

    # top-5 recommendations for user 0 (unrated items)
    u0 = 0
    unrated = np.where(np.isnan(train[u0]))[0]
    scored = sorted(((predict(u0, i), i) for i in unrated), reverse=True)[:5]

    plt.figure(figsize=(5, 4))
    plt.bar(["item-mean\nbaseline", "item-item\nCF"], [rmse_base, rmse_cf],
            color=["#9aa7b4", "#4c78a8"])
    plt.ylabel("RMSE (lower is better)")
    plt.title(f"CF beats baseline by {improve:.0%}")
    plt.tight_layout()
    plt.savefig(os.path.join(res, "rmse.png"), dpi=110)
    plt.close()

    with open(os.path.join(res, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("# Recommender (item-item CF) — Results\n\n")
        f.write("Synthetic ratings: **200 users x 60 items**, ~30% observed, latent rank 4. "
                "20% of observed ratings held out for testing.\n\n")
        f.write(f"- **RMSE — item-item CF: {rmse_cf:.3f}**\n")
        f.write(f"- RMSE — item-mean baseline: {rmse_base:.3f}\n")
        f.write(f"- **Improvement over baseline: {improve:.0%}**\n\n")
        f.write("Top-5 recommendations for user 0 (predicted rating · item id):\n")
        for score, i in scored:
            f.write(f"- item {i}: {score:.2f}\n")

    print(f"RMSE_cf={rmse_cf:.3f} RMSE_base={rmse_base:.3f} improve={improve:.0%}")
    print("self-check:", "OK" if rmse_cf < rmse_base else "FAIL")


if __name__ == "__main__":
    main()
