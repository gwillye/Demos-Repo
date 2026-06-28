"""Dimensionality-reduction demo — PCA on scikit-learn's bundled digits dataset.

Self-contained (the dataset ships with scikit-learn). Projects 64-dim digit vectors to 2D,
colors by class, and reports explained variance. Saves results/pca_scatter.png,
results/scree.png and results/RESULTS.md.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SEED = 42


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(here, "results")
    os.makedirs(res, exist_ok=True)

    digits = load_digits()
    X = StandardScaler().fit_transform(digits.data)
    y = digits.target

    pca2 = PCA(n_components=2, random_state=SEED).fit(X)
    Z = pca2.transform(X)
    ev2 = pca2.explained_variance_ratio_

    pca_full = PCA(random_state=SEED).fit(X)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n90 = int(np.argmax(cum >= 0.90) + 1)

    # 2D scatter colored by digit
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=12, alpha=0.7)
    plt.colorbar(sc, ticks=range(10), label="digit")
    plt.title(f"PCA of digits — PC1+PC2 explain {ev2.sum():.1%} of variance")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(res, "pca_scatter.png"), dpi=110)
    plt.close()

    # scree / cumulative variance
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(cum) + 1), cum, marker=".")
    plt.axhline(0.90, color="gray", ls="--")
    plt.axvline(n90, color="red", ls=":")
    plt.title(f"Cumulative explained variance — {n90} PCs reach 90%")
    plt.xlabel("number of components")
    plt.ylabel("cumulative variance")
    plt.tight_layout()
    plt.savefig(os.path.join(res, "scree.png"), dpi=110)
    plt.close()

    with open(os.path.join(res, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("# PCA on Digits — Results\n\n")
        f.write("Dataset: scikit-learn `load_digits` (1,797 × 64), standardized.\n\n")
        f.write(f"- PC1 + PC2 explain **{ev2.sum():.1%}** of variance ({ev2[0]:.1%} + {ev2[1]:.1%}).\n")
        f.write(f"- **{n90} principal components** capture **90%** of the variance "
                f"(from 64 original dimensions).\n\n")
        f.write("See `pca_scatter.png` (2D projection by digit) and `scree.png`.\n")

    print(f"PC1+PC2={ev2.sum():.1%}  n90={n90}")
    print("self-check:", "OK" if 0 < n90 < 64 else "FAIL")


if __name__ == "__main__":
    main()
