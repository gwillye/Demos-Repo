"""
A/B Test Analyzer - demo
------------------------
Simulates a marketing A/B test (control vs. treatment) and runs a proper
statistical analysis from BOTH schools: the frequentist read (two-proportion
z-test, 95% confidence interval, chi-square cross-check, Welch t-test on
revenue-per-converter) AND the Bayesian read (Beta-Binomial posterior with
P(treatment > control) and a credible interval) - because a p-value answers
"how surprising is this data if there were no effect?" while the team actually
wants "how likely is B better, and by how much?".

Real-world context: every growth / marketing / product team runs experiments
(new landing page, email subject, checkout flow). The hard part is reading them
correctly - is the lift real or noise? This is the stats layer that turns "the
variant looks better" into a defensible ship / no-ship decision.

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

    # interactive view (GitHub-Pages ready): frequentist bars + Bayesian posteriors
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Conversion rate (95% CI)", "Bayesian posteriors"))
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
    fig.update_layout(template="plotly_white", height=460,
                      title=f"A/B test - lift {lift:+.1%} · p={pval:.3f} · "
                            f"P(B>A)={bay['p_t_gt_c']:.0%}")
    fig.write_html(out / "ab_test.html", include_plotlyjs="cdn")

    print(f"Done. lift={lift:+.1%}, p={pval:.4f}, significant={sig}, "
          f"P(B>A)={bay['p_t_gt_c']:.1%}")
    print("See results/RESULTS.md + results/conversion.png + results/ab_test.html")


if __name__ == "__main__":
    main()
