"""Budget optimization on top of the Marketing Mix Model.

Measuring channel ROI is only half the job — the decision MMM exists for is
*"given a fixed budget, how should I split it across channels?"*. With
**diminishing returns** (saturation), the answer is not "dump it all on the best
channel": each channel's marginal return falls as you spend more.

This models a saturating response per channel and uses constrained optimization
(SLSQP) to find the allocation that maximizes total response for a fixed budget,
then reports the lift over an even split. Run:  python optimize.py
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

# Illustrative saturating-response params per channel (in practice fitted from the MMM):
#   response(x) = max_effect * x / (x + half_sat)
CH = {
    "TV":     dict(max_effect=500_000, half_sat=60_000),
    "Search": dict(max_effect=350_000, half_sat=25_000),
    "Social": dict(max_effect=200_000, half_sat=40_000),
    "Email":  dict(max_effect=400_000, half_sat=20_000),
}
CHANNELS = list(CH)
BUDGET = 200_000.0


def response(x: np.ndarray) -> np.ndarray:
    me = np.array([CH[c]["max_effect"] for c in CHANNELS])
    hs = np.array([CH[c]["half_sat"] for c in CHANNELS])
    return me * x / (x + hs)


def total_response(x: np.ndarray) -> float:
    return float(response(np.asarray(x)).sum())


def optimize(budget: float):
    n = len(CHANNELS)
    x0 = np.full(n, budget / n)
    cons = {"type": "eq", "fun": lambda x: x.sum() - budget}
    bounds = [(0, budget)] * n
    res = minimize(lambda x: -total_response(x), x0, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"ftol": 1e-9, "maxiter": 500})
    return res.x, res


def main() -> None:
    even = np.full(len(CHANNELS), BUDGET / len(CHANNELS))
    opt, res = optimize(BUDGET)

    s_even, s_opt = total_response(even), total_response(opt)
    lift = (s_opt - s_even) / s_even

    print(f"Budget: R$ {BUDGET:,.0f}\n")
    print(f"{'channel':<8} {'even split':>12} {'optimized':>12} {'marginal@opt':>14}")
    me = np.array([CH[c]["max_effect"] for c in CHANNELS])
    hs = np.array([CH[c]["half_sat"] for c in CHANNELS])
    marg = me * hs / (opt + hs) ** 2
    for i, c in enumerate(CHANNELS):
        print(f"{c:<8} {even[i]:>12,.0f} {opt[i]:>12,.0f} {marg[i]:>14.3f}")

    print(f"\nPredicted response — even: {s_even:,.0f} | optimized: {s_opt:,.0f}")
    print(f"Lift from reallocation: {lift:+.1%}")
    _write_chart(even, opt)
    _self_check(res, opt, s_opt, s_even, marg)


def _write_chart(even, opt) -> None:
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_bar(name="Even split", x=CHANNELS, y=even)
    fig.add_bar(name="Optimized", x=CHANNELS, y=opt)
    fig.update_layout(barmode="group", template="plotly_white",
                      title="MMM budget optimization — even vs optimal allocation",
                      yaxis_title="Spend (R$)")
    fig.write_html("budget_optimization.html", include_plotlyjs="cdn")
    print("chart written: budget_optimization.html")


def _self_check(res, opt, s_opt, s_even, marg) -> None:
    assert res.success, f"optimizer did not converge: {res.message}"
    assert abs(opt.sum() - BUDGET) < 1.0, "budget constraint violated"
    assert (opt >= -1e-6).all(), "negative allocation"
    assert s_opt >= s_even - 1e-6, "optimized worse than even split"
    # at the optimum, channels with spend>0 should have ~equal marginal return
    active = opt > 1.0
    assert marg[active].std() / marg[active].mean() < 0.05, "marginal returns not equalized"
    print("self-check: OK (converged, budget respected, lift >= 0, marginals equalized)")


if __name__ == "__main__":
    main()
