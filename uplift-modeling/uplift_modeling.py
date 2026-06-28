"""
Uplift Modeling for Campaign Targeting (T-learner)
--------------------------------------------------
Goal: instead of predicting WHO will convert, predict who is *persuadable* —
i.e., whose conversion probability is actually *increased* by the campaign.
This is the right objective for marketing spend: target the "persuadables",
not the "sure things" (convert anyway) or "lost causes" (never convert), and
avoid "sleeping dogs" (campaign annoys them -> negative effect).

Method: T-learner — two models, one trained on the treated group and one on the
control group; uplift(x) = P(convert | x, treated) - P(convert | x, control).
Evaluation: Qini curve + actual-uplift-by-decile on a held-out set.

Synthetic but realistic RCT data with a KNOWN heterogeneous treatment effect,
so we can check the model recovers the right ranking. Run: `python uplift_modeling.py`
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)
N = 20_000


def make_data(n=N):
    age = RNG.normal(40, 12, n).clip(18, 80)
    recency = RNG.exponential(30, n).clip(1, 365)     # days since last purchase
    frequency = RNG.poisson(5, n)                      # past purchases
    monetary = RNG.gamma(2, 50, n)                     # avg ticket
    engagement = RNG.beta(2, 5, n)                     # email engagement (0-1)
    X = np.column_stack([age, recency, frequency, monetary, engagement])

    # Baseline conversion p0(x): engaged/frequent/recent convert more anyway
    lin0 = -2.0 + 1.6 * engagement + 0.05 * frequency - 0.004 * recency
    p0 = 1 / (1 + np.exp(-lin0))

    # Heterogeneous treatment effect tau(x):
    #  - persuadables: mid-engagement -> large positive uplift
    #  - sleeping dogs: already-very-engaged -> slight NEGATIVE effect
    tau = 0.18 * np.exp(-((engagement - 0.45) ** 2) / 0.04)
    tau -= 0.10 * (engagement > 0.80)

    T = RNG.integers(0, 2, n)                           # randomized treatment (RCT)
    p = np.where(T == 1, np.clip(p0 + tau, 1e-3, 1 - 1e-3), p0)
    y = (RNG.random(n) < p).astype(int)

    cols = ["age", "recency", "frequency", "monetary", "engagement"]
    df = pd.DataFrame(X, columns=cols)
    df["treatment"], df["converted"], df["true_uplift"] = T, y, tau
    return df, cols


def qini_curve(df):
    """Cumulative incremental conversions when targeting customers by predicted uplift (desc)."""
    s = df.sort_values("uplift_pred", ascending=False).reset_index(drop=True)
    nt = (s.treatment == 1).cumsum()
    nc = (s.treatment == 0).cumsum().replace(0, np.nan)
    rt = (s.converted * (s.treatment == 1)).cumsum()
    rc = (s.converted * (s.treatment == 0)).cumsum()
    qini = (rt - rc * (nt / nc)).fillna(0).to_numpy()
    x = np.arange(1, len(s) + 1)
    rand = qini[-1] * x / len(s)                        # random targeting baseline
    coef = np.trapezoid(qini - rand, x) / len(s)        # area between curves (Qini coefficient)
    return x, qini, rand, coef


def main():
    df, cols = make_data()
    tr, te = train_test_split(df, test_size=0.4, random_state=42, stratify=df["treatment"])

    # T-learner: one model per arm
    m1 = GradientBoostingClassifier(random_state=0).fit(
        tr[tr.treatment == 1][cols], tr[tr.treatment == 1]["converted"])
    m0 = GradientBoostingClassifier(random_state=0).fit(
        tr[tr.treatment == 0][cols], tr[tr.treatment == 0]["converted"])

    te = te.copy()
    te["uplift_pred"] = m1.predict_proba(te[cols])[:, 1] - m0.predict_proba(te[cols])[:, 1]

    # Actual uplift by predicted-uplift decile (decile 9 = highest predicted uplift)
    te["decile"] = pd.qcut(te["uplift_pred"], 10, labels=False, duplicates="drop")
    rows = []
    for d in sorted(te["decile"].unique()):
        g = te[te.decile == d]
        ct = g[g.treatment == 1]["converted"].mean()
        cc = g[g.treatment == 0]["converted"].mean()
        rows.append({"decile": int(d), "n": len(g),
                     "pred_uplift": g.uplift_pred.mean(), "actual_uplift": ct - cc})
    dec = pd.DataFrame(rows).sort_values("decile", ascending=False)

    x, qini, rand, coef = qini_curve(te)

    overall = te[te.treatment == 1]["converted"].mean() - te[te.treatment == 0]["converted"].mean()
    top = dec[dec.decile >= 8]["actual_uplift"].mean()
    bot = dec[dec.decile <= 1]["actual_uplift"].mean()

    print("=== Uplift Modeling (T-learner) ===")
    print(f"Test set: {len(te):,} customers | overall avg uplift (ATE): {overall:+.4f}")
    print(f"Qini coefficient (area vs random): {coef:.1f}  (>0 = better than random targeting)")
    print("\nActual uplift by predicted-uplift decile (10 = best):")
    for _, r in dec.iterrows():
        bar = "#" * max(0, int(r.actual_uplift * 200))
        print(f"  decile {int(r.decile):>2} | n={int(r.n):>4} | pred {r.pred_uplift:+.3f} | actual {r.actual_uplift:+.4f} {bar}")
    print(f"\nTop-2 deciles actual uplift: {top:+.4f}  vs  bottom-2: {bot:+.4f}")

    # Plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(x, qini, label="Uplift model")
    ax[0].plot(x, rand, "--", color="gray", label="Random targeting")
    ax[0].set_title(f"Qini curve (coef={coef:.1f})")
    ax[0].set_xlabel("Customers targeted (by uplift, desc)")
    ax[0].set_ylabel("Cumulative incremental conversions")
    ax[0].legend()
    ax[1].bar(dec.decile, dec.actual_uplift, color="#2a7")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title("Actual uplift by decile")
    ax[1].set_xlabel("Predicted-uplift decile (9 = best)")
    ax[1].set_ylabel("Actual uplift (treated - control conv.)")
    fig.tight_layout()
    fig.savefig("uplift_results.png", dpi=110)
    print("\nSaved plot -> uplift_results.png")

    # Self-check: model must rank uplift (top deciles >> bottom deciles) and beat random
    assert top > bot, "self-check FAILED: top deciles should have higher actual uplift than bottom"
    assert coef > 0, "self-check FAILED: Qini coefficient should be positive"
    print("SELF-CHECK OK: model ranks persuadables above sleeping-dogs/lost-causes, and beats random targeting.")


if __name__ == "__main__":
    main()
