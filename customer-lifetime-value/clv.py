"""Customer Lifetime Value (CLV/LTV) — predict, then validate on a holdout.

The core customer-analytics metric: how much future value will each customer
bring? A naive "sum past spend" ignores churn. This builds a simple, interpretable
predictive CLV from a **calibration window** (frequency / recency / monetary +
a p(alive) recency decay) and then **validates it on a future holdout window** —
the honest test: does predicted CLV actually correlate with future spend?

Run:  python clv.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

N_CUST = 4000
T_CAL = 365     # days of history used to build the model
T_HOLD = 180    # future window used only to validate
MARGIN = 0.30


def simulate(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    horizon = T_CAL + T_HOLD
    for cid in range(N_CUST):
        rate = rng.gamma(2.0, 0.01)          # latent purchases/day (heterogeneous)
        lifetime = rng.exponential(400)       # latent churn time
        aov = rng.gamma(4.0, 25)              # latent average order value
        t = rng.uniform(0, 60)                # acquisition offset
        while t < horizon:
            gap = rng.exponential(1 / max(rate, 1e-4))
            t += gap
            if t > lifetime or t >= horizon:
                break
            rows.append((cid, t, max(5.0, rng.normal(aov, aov * 0.2))))
    return pd.DataFrame(rows, columns=["customer", "t", "value"])


def build_features(tx: pd.DataFrame) -> pd.DataFrame:
    cal = tx[tx.t <= T_CAL]
    g = cal.groupby("customer")
    f = g.agg(n_txn=("t", "size"), first_t=("t", "min"),
              last_t=("t", "max"), monetary=("value", "mean")).reset_index()
    f["T_obs"] = (T_CAL - f.first_t).clip(lower=1)
    f["rate"] = f.n_txn / f["T_obs"]
    # p(alive): silent for many of your own purchase-cycles => likely churned (BG/NBD intuition)
    expected_gap = (f["T_obs"] / f.n_txn).clip(lower=1)
    time_since_last = (T_CAL - f.last_t).clip(lower=0)
    f["p_alive"] = np.exp(-time_since_last / (2 * expected_gap))
    f["pred_future_txn"] = f.rate * T_HOLD * f.p_alive
    f["pred_clv"] = f.pred_future_txn * f.monetary * MARGIN
    return f


def main() -> None:
    tx = simulate()
    feats = build_features(tx)

    # actual spend in the holdout window (validation target)
    hold = tx[(tx.t > T_CAL) & (tx.t <= T_CAL + T_HOLD)]
    actual = hold.groupby("customer")["value"].sum().rename("actual_future_spend")
    df = feats.merge(actual, on="customer", how="left").fillna({"actual_future_spend": 0.0})

    corr = float(np.corrcoef(df.pred_clv, df.actual_future_spend)[0, 1])

    # decile lift: does the top-predicted decile capture an outsized share of future spend?
    df["decile"] = pd.qcut(df.pred_clv.rank(method="first"), 10, labels=False)
    top = df[df.decile == 9]
    top_share = top.actual_future_spend.sum() / df.actual_future_spend.sum()

    print(f"{len(df)} customers | calibration {T_CAL}d, holdout {T_HOLD}d")
    print(f"predicted CLV vs actual future spend — correlation: {corr:.3f}")
    print(f"top-decile customers capture {top_share:.1%} of actual future revenue")
    print("\nTop 5 customers by predicted CLV:")
    for _, r in df.nlargest(5, "pred_clv").iterrows():
        print(f"  #{int(r.customer):<5} pred_clv={r.pred_clv:7.0f}  actual_future={r.actual_future_spend:7.0f}")

    _self_check(corr, top_share)


def _self_check(corr: float, top_share: float) -> None:
    # Individual-level LTV is inherently noisy; the business metric is decile LIFT,
    # so the headline assertion is on top-decile revenue capture (a top decile = 10%
    # of customers; capturing >25% of future revenue is a real, useful lift).
    assert top_share > 0.30, f"top decile should capture outsized revenue, got {top_share:.1%}"
    assert corr > 0.25, f"predicted CLV should still positively correlate, got {corr}"
    print("\nself-check: OK (top-decile lift >30% of future revenue; positive correlation)")


if __name__ == "__main__":
    main()
