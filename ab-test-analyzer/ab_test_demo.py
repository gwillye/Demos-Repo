"""
A/B Test Analyzer - demo
------------------------
Simulates a marketing A/B test (control vs. treatment) and runs a proper
statistical analysis from BOTH schools: the frequentist read (two-proportion
z-test, 95% confidence interval, chi-square cross-check, Welch t-test on
revenue-per-converter) AND the Bayesian read (Beta-Binomial posterior with
P(treatment > control) and a credible interval) - because a p-value answers
"how surprising is this data if there were no effect?" while the team actually
wants "how likely is B better, and by how much?". It also closes the loop with a
**power / sample-size** analysis - the question you should ask *before* running a
test ("how many users do I need, and what lift can this sample even detect?").

Real-world context: every growth / marketing / product team runs experiments
(new landing page, email subject, checkout flow). The hard part is reading them
correctly - is the lift real or noise? - and *designing* them so they can detect
the lift you care about. This is the stats layer that turns "the variant looks
better" into a defensible ship / no-ship decision, and "let's just run it" into a
properly-powered experiment.

Data is SYNTHETIC (seeded). To analyze a real test, feed `analyze()` the
conversion counts and revenue arrays from your two groups.

Run:  python ab_test_demo.py
Out:  results/RESULTS.md, results/conversion.png, results/ab_test.html (interactive)
"""
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RNG = np.random.default_rng(11)
ALPHA = 0.05


def simulate(n_c=6000, n_t=6000, p_c=0.100, p_t=0.116, rev=44.0):
    conv_c = RNG.random(n_c) < p_c
    conv_t = RNG.random(n_t) < p_t
    rev_c = RNG.normal(rev, 12, int(conv_c.sum())).clip(1)
    rev_t = RNG.normal(rev + 2, 12, int(conv_t.sum())).clip(1)
    return conv_c, conv_t, rev_c, rev_t


def two_prop_ztest(x_c, n_c, x_t, n_t):
    p_c, p_t = x_c / n_c, x_t / n_t
    p_pool = (x_c + x_t) / (n_c + n_t)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    z = (p_t - p_c) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return p_c, p_t, z, p


def diff_ci(p_c, n_c, p_t, n_t, conf=0.95):
    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    zc = stats.norm.ppf(1 - (1 - conf) / 2)
    d = p_t - p_c
    return d - zc * se, d + zc * se, zc * se


def bayesian(x_c, n_c, x_t, n_t, draws=200_000):
    """Beta-Binomial posteriors with a flat Beta(1,1) prior on each rate.
    Returns P(treatment > control), expected absolute lift, a 95% credible
    interval on that lift, and posterior samples for plotting."""
    sc = RNG.beta(1 + x_c, 1 + n_c - x_c, draws)   # posterior of control rate
    st = RNG.beta(1 + x_t, 1 + n_t - x_t, draws)   # posterior of treatment rate
    d = st - sc
    return dict(p_t_gt_c=float((st > sc).mean()),
                exp_lift=float(d.mean()),
                ci=(float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))),
                sc=sc, st=st)


def power_analysis(p_c, n_per_arm, design_mde=0.10, alpha=ALPHA, target_power=0.80):
    """Design-side stats for a two-proportion test.
    - n_for_design : users PER ARM needed to detect a `design_mde` relative lift
      at `target_power` (the question to ask *before* launching).
    - mde_at_n     : the smallest relative lift THIS sample (`n_per_arm`) can
      detect at `target_power` - i.e. what the test is actually powered for.
    Also returns the closures so the caller can draw a power curve."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(target_power)

    def req_n(mde):
        p1, p2 = p_c, p_c * (1 + mde)
        pbar = (p1 + p2) / 2
        num = za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        return num ** 2 / (p2 - p1) ** 2

    def power_at(mde, n):
        p1, p2 = p_c, p_c * (1 + mde)
        pbar = (p1 + p2) / 2
        se_null = np.sqrt(2 * pbar * (1 - pbar) / n)
        se_alt = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
        return float(stats.norm.cdf((abs(p2 - p1) - za * se_null) / se_alt))

    grid = np.linspace(0.01, 0.50, 500)
    detectable = [m for m in grid if power_at(m, n_per_arm) >= target_power]
    return dict(n_for_design=int(np.ceil(req_n(design_mde))), design_mde=design_mde,
                mde_at_n=(detectable[0] if detectable else float("nan")),
                alpha=alpha, target_power=target_power, req_n=req_n, power_at=power_at)


def main():
    conv_c, conv_t, rev_c, rev_t = simulate()
    n_c, n_t = conv_c.size, conv_t.size
    x_c, x_t = int(conv_c.sum()), int(conv_t.sum())
    p_c, p_t, z, pval = two_prop_ztest(x_c, n_c, x_t, n_t)
    lo, hi, half = diff_ci(p_c, n_c, p_t, n_t)
    lift = (p_t - p_c) / p_c

    # chi-square cross-check
    table = np.array([[x_c, n_c - x_c], [x_t, n_t - x_t]])
    chi2, p_chi, _, _ = stats.chi2_contingency(table, correction=False)
    # revenue per converter (Welch t-test)
    t, p_rev = stats.ttest_ind(rev_t, rev_c, equal_var=False)
    # Bayesian read (Beta-Binomial)
    bay = bayesian(x_c, n_c, x_t, n_t)
    # design-side: power / sample size
    pa = power_analysis(p_c, n_c)
    obs_power = pa["power_at"](lift, n_c)   # post-hoc power for the observed lift

    sig = "YES" if pval < ALPHA else "NO"
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    md = f"""# A/B Test Analyzer - Results

| Group | Users | Conversions | Rate |
|---|---|---|---|
| Control (A)   | {n_c:,} | {x_c:,} | {p_c:.3%} |
| Treatment (B) | {n_t:,} | {x_t:,} | {p_t:.3%} |

## Conversion
- **Relative lift: {lift:+.1%}** (absolute {p_t - p_c:+.3%})
- 95% CI for the difference: **[{lo:+.3%}, {hi:+.3%}]**
- Two-proportion z-test: z = {z:.2f}, **p = {pval:.4f}**
- Chi-square cross-check: chi2 = {chi2:.2f}, p = {p_chi:.4f}
- **Statistically significant at alpha={ALPHA}? {sig}**

## Revenue per converter (secondary metric)
- Control mean R$ {rev_c.mean():.2f} vs Treatment R$ {rev_t.mean():.2f}
- Welch t-test: t = {t:.2f}, p = {p_rev:.4f}

## Bayesian read (Beta-Binomial, flat prior)
- **P(treatment > control) = {bay['p_t_gt_c']:.1%}** - the decision-useful number a
  p-value can't give you directly.
- Expected absolute lift **{bay['exp_lift']:+.3%}**, 95% credible interval
  **[{bay['ci'][0]:+.3%}, {bay['ci'][1]:+.3%}]**.
- Reading both schools together: the frequentist test says *"this isn't noise"*; the
  Bayesian posterior says *"and here's how confident we are that B wins, and by how much."*

## Power / sample size (design the test, not just read it)
- To detect a **{pa['design_mde']:+.0%} relative lift** at **{pa['target_power']:.0%} power**
  (alpha={pa['alpha']}), you'd need **~{pa['n_for_design']:,} users per arm**.
- This test ({n_c:,}/arm) is powered to detect down to a **{pa['mde_at_n']:+.1%}**
  relative lift at {pa['target_power']:.0%} power - anything smaller it would likely **miss**.
- Post-hoc power for the *observed* {lift:+.1%} lift: **{obs_power:.0%}**.
- Takeaway: the observed effect was large enough to detect, but a true lift below
  ~{pa['mde_at_n']:+.1%} would have left this test **underpowered** - size the next
  experiment for the smallest lift that's worth shipping.

## Verdict
{"Ship B - the lift is real and unlikely to be noise." if sig == "YES"
 else "Inconclusive - keep running or increase sample size."}
"""
    (out / "RESULTS.md").write_text(md, encoding="utf-8")

    plt.figure(figsize=(5.5, 4.5))
    rates = [p_c, p_t]
    errs = [stats.norm.ppf(.975) * np.sqrt(p_c * (1 - p_c) / n_c),
            stats.norm.ppf(.975) * np.sqrt(p_t * (1 - p_t) / n_t)]
    plt.bar(["Control (A)", "Treatment (B)"], rates, yerr=errs, capsize=8,
            color=["#999999", "#4C72B0"])
    plt.ylabel("Conversion rate"); plt.title(f"A/B conversion (lift {lift:+.1%}, p={pval:.3f})")
    plt.tight_layout(); plt.savefig(out / "conversion.png", dpi=110); plt.close()

    # interactive view (GitHub-Pages ready): bars + Bayesian posteriors + power curve
    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "Conversion rate (95% CI)", "Bayesian posteriors",
        f"Power curve (@ {pa['design_mde']:+.0%} MDE)"))
    for lbl, rate, n, col in [("Control (A)", p_c, n_c, "#999999"),
                              ("Treatment (B)", p_t, n_t, "#4C72B0")]:
        se = np.sqrt(rate * (1 - rate) / n)
        fig.add_bar(x=[lbl], y=[rate], marker_color=col, showlegend=False,
                    error_y=dict(type="data", array=[1.96 * se]), row=1, col=1)
    for lbl, s, col in [("Control (A)", bay["sc"], "#999999"),
                        ("Treatment (B)", bay["st"], "#4C72B0")]:
        h, e = np.histogram(s, bins=120, density=True)
        fig.add_scatter(x=(e[:-1] + e[1:]) / 2, y=h, name=lbl, fill="tozeroy",
                        line=dict(color=col), row=1, col=2)
    ns = np.linspace(500, max(2.2 * n_c, 1.2 * pa["n_for_design"]), 80)
    fig.add_scatter(x=ns, y=[pa["power_at"](pa["design_mde"], n) for n in ns],
                    name="power", line=dict(color="#DD8452", width=2.5), row=1, col=3)
    fig.add_hline(y=pa["target_power"], line=dict(dash="dash", color="#bbb"), row=1, col=3)
    fig.add_vline(x=n_c, line=dict(dash="dot", color="#4C72B0"), row=1, col=3,
                  annotation_text=f"this test ({n_c:,}/arm)", annotation_font_size=9)
    fig.update_xaxes(title="users per arm", row=1, col=3)
    fig.update_yaxes(title="power", range=[0, 1], row=1, col=3)
    fig.update_layout(template="plotly_white", height=460, showlegend=True,
                      title=f"A/B test - lift {lift:+.1%} · p={pval:.3f} · "
                            f"P(B>A)={bay['p_t_gt_c']:.0%} · needs ~{pa['n_for_design']:,}/arm "
                            f"for {pa['design_mde']:+.0%} MDE")
    fig.write_html(out / "ab_test.html", include_plotlyjs="cdn")

    print(f"Done. lift={lift:+.1%}, p={pval:.4f}, significant={sig}, "
          f"P(B>A)={bay['p_t_gt_c']:.1%}, MDE@n={pa['mde_at_n']:+.1%}, "
          f"n/arm for {pa['design_mde']:+.0%}={pa['n_for_design']:,}")
    print("See results/RESULTS.md + results/conversion.png + results/ab_test.html")


if __name__ == "__main__":
    main()
