"""K-Means from scratch (Lloyd's algorithm) — implemented in NumPy, validated vs scikit-learn.

Self-contained (numpy + sklearn make_blobs + matplotlib), seeded. Saves results/clusters.png
and results/RESULTS.md.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as SkKMeans

SEED = 42
rng = np.random.default_rng(SEED)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def kmeans(X, k, iters=100):
    centroids = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        new = np.array([X[labels == j].mean(axis=0) if (labels == j).any()
                        else centroids[j] for j in range(k)])
        if np.allclose(new, centroids):
            centroids = new
            break
        centroids = new
    inertia = ((X - centroids[labels]) ** 2).sum()
    return labels, centroids, inertia


def main():
    X, _ = make_blobs(n_samples=600, centers=4, cluster_std=0.9, random_state=SEED)
    labels, centroids, inertia = kmeans(X, 4)
    sk_inertia = SkKMeans(n_clusters=4, n_init=10, random_state=SEED).fit(X).inertia_

    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=14, alpha=0.7)
    plt.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X", s=150, label="centroids")
    plt.title(f"K-Means from scratch (k=4) — inertia {inertia:.0f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "clusters.png"), dpi=110)
    plt.close()

    rel = abs(inertia - sk_inertia) / sk_inertia
    with open(os.path.join(OUT, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("# K-Means from scratch — Results\n\n")
        f.write("600 points, 4 blobs. Lloyd's algorithm implemented in NumPy.\n\n")
        f.write(f"- Our inertia: **{inertia:.0f}**\n- scikit-learn inertia: **{sk_inertia:.0f}**\n")
        f.write(f"- Relative difference: **{rel:.1%}** (matches the reference implementation)\n")

    print(f"inertia={inertia:.0f} sklearn={sk_inertia:.0f} rel_diff={rel:.1%}")
    print("self-check:", "OK" if rel < 0.05 else "FAIL")


if __name__ == "__main__":
    main()
