# A/B Test Analyzer - Results

| Group | Users | Conversions | Rate |
|---|---|---|---|
| Control (A)   | 6,000 | 593 | 9.883% |
| Treatment (B) | 6,000 | 735 | 12.250% |

## Conversion
- **Relative lift: +23.9%** (absolute +2.367%)
- 95% CI for the difference: **[+1.245%, +3.488%]**
- Two-proportion z-test: z = 4.13, **p = 0.0000**
- Chi-square cross-check: chi2 = 17.07, p = 0.0000
- **Statistically significant at alpha=0.05? YES**

## Revenue per converter (secondary metric)
- Control mean R$ 43.42 vs Treatment R$ 45.96
- Welch t-test: t = 3.86, p = 0.0001

## Bayesian read (Beta-Binomial, flat prior)
- **P(treatment > control) = 100.0%** - the decision-useful number a
  p-value can't give you directly.
- Expected absolute lift **+2.367%**, 95% credible interval
  **[+1.242%, +3.491%]**.
- Reading both schools together: the frequentist test says *"this isn't noise"*; the
  Bayesian posterior says *"and here's how confident we are that B wins, and by how much."*

## Power / sample size (design the test, not just read it)
- To detect a **+10% relative lift** at **80% power**
  (alpha=0.05), you'd need **~14,946 users per arm**.
- This test (6,000/arm) is powered to detect down to a **+16.0%**
  relative lift at 80% power - anything smaller it would likely **miss**.
- Post-hoc power for the *observed* +23.9% lift: **99%**.
- Takeaway: the observed effect was large enough to detect, but a true lift below
  ~+16.0% would have left this test **underpowered** - size the next
  experiment for the smallest lift that's worth shipping.

## Verdict
Ship B - the lift is real and unlikely to be noise.
