"""Handwritten-digit classifier on scikit-learn's bundled 8x8 digits dataset.

Self-contained — the dataset ships with scikit-learn (no download). Trains an SVM,
reports accuracy + confusion matrix, and saves a sample-predictions grid.
Saves results/RESULTS.md, results/confusion_matrix.png, results/samples.png.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

SEED = 42


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(here, "results")
    os.makedirs(res, exist_ok=True)

    digits = load_digits()
    X, y, images = digits.data, digits.target, digits.images
    Xtr, Xte, ytr, yte, _, img_te = train_test_split(
        X, y, images, test_size=0.25, random_state=SEED, stratify=y)

    clf = SVC(gamma=0.001, kernel="rbf")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = accuracy_score(yte, pred)
    cm = confusion_matrix(yte, pred)
    report = classification_report(yte, pred, digits=3)

    # confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion matrix — accuracy {acc:.3f}")
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.colorbar(fraction=0.046)
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.tight_layout()
    plt.savefig(os.path.join(res, "confusion_matrix.png"), dpi=110)
    plt.close()

    # sample predictions grid
    fig, axes = plt.subplots(2, 6, figsize=(9, 3.4))
    for ax, im, t, p in zip(axes.ravel(), img_te, yte, pred):
        ax.imshow(im, cmap="gray_r")
        ax.set_title(f"{p}" + ("" if t == p else f"≠{t}"),
                     color="black" if t == p else "red", fontsize=10)
        ax.axis("off")
    fig.suptitle("Sample predictions (red = miss)")
    plt.tight_layout()
    plt.savefig(os.path.join(res, "samples.png"), dpi=110)
    plt.close()

    with open(os.path.join(res, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("# Digit Classifier — Results\n\n")
        f.write(f"Dataset: scikit-learn `load_digits` — **{len(X)}** 8x8 images, 10 classes. "
                f"Model: **SVM (RBF, gamma=0.001)**.\n\n")
        f.write(f"- **Test accuracy: {acc:.3f}**\n\n")
        f.write("```\n" + report + "\n```\n\n")
        f.write("See `confusion_matrix.png` and `samples.png`.\n")

    print(f"accuracy = {acc:.3f}")
    print("self-check:", "OK" if acc > 0.95 else "FAIL")


if __name__ == "__main__":
    main()
