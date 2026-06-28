"""Sentiment analysis demo — TF-IDF + Logistic Regression on synthetic product reviews.

Self-contained (no downloads), seeded for reproducibility. Writes results/RESULTS.md
and results/roc_curve.png, and prints a self-check.
"""
import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, roc_curve

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

POS = ["excellent", "amazing", "love", "great", "fantastic", "perfect", "recommend",
       "wonderful", "fast", "reliable", "beautiful", "happy", "best", "smooth", "worth"]
NEG = ["terrible", "awful", "hate", "broken", "disappointing", "slow", "useless",
       "worst", "poor", "defective", "waste", "unhappy", "cheap", "buggy", "refund"]
NEUTRAL = ["the", "a", "product", "item", "delivery", "price", "quality", "service",
           "arrived", "package", "seller", "store", "bought", "using", "after"]


def make_review(sentiment):
    """Weak signal + contamination so the task isn't trivially separable."""
    pool = POS if sentiment == 1 else NEG
    other = NEG if sentiment == 1 else POS
    words = []
    for _ in range(random.randint(8, 16)):
        r = random.random()
        if r < 0.25:
            words.append(random.choice(pool))       # on-sentiment signal
        elif r < 0.33:
            words.append(random.choice(other))      # contamination (opposite words)
        else:
            words.append(random.choice(NEUTRAL))
    return " ".join(words)


def make_data(n=2000, label_noise=0.06):
    """Text reflects the true sentiment; a fraction of observed labels are flipped
    (real-world mislabeling) so a perfect score is not achievable."""
    X, y = [], []
    for _ in range(n):
        s = random.randint(0, 1)
        X.append(make_review(s))
        y.append(1 - s if random.random() < label_noise else s)
    return X, np.array(y)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(here, "results")
    os.makedirs(res, exist_ok=True)

    X, y = make_data(2000)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(Xtr, ytr)

    proba = pipe.predict_proba(Xte)[:, 1]
    pred = pipe.predict(Xte)
    auc = roc_auc_score(yte, proba)
    report = classification_report(yte, pred, digits=3)

    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    names = np.array(vec.get_feature_names_out())
    coef = clf.coef_[0]
    top_pos = names[np.argsort(coef)[-10:]][::-1]
    top_neg = names[np.argsort(coef)[:10]]

    fpr, tpr, _ = roc_curve(yte, proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Sentiment classifier — ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(res, "roc_curve.png"), dpi=110)
    plt.close()

    with open(os.path.join(res, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("# Sentiment Analysis — Results\n\n")
        f.write("Synthetic reviews: **2000** (balanced). TF-IDF (1–2 gram) + Logistic Regression.\n\n")
        f.write(f"- **ROC-AUC: {auc:.3f}**\n\n")
        f.write("```\n" + report + "\n```\n\n")
        f.write("Top positive words: " + ", ".join(top_pos) + "\n\n")
        f.write("Top negative words: " + ", ".join(top_neg) + "\n")

    print(f"ROC-AUC = {auc:.3f}")
    print("self-check:", "OK" if auc > 0.8 else "FAIL")


if __name__ == "__main__":
    main()
