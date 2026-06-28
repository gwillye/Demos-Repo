# A/B Test Analyzer

Read a marketing experiment correctly: conversion **lift**, **statistical
significance**, confidence intervals, and a ship / no-ship **verdict**.

## Real-world context
Every growth, marketing and product team runs A/B tests - new landing page,
email subject line, checkout flow. The hard part isn't running the test, it's
reading it: *is the variant's lift real, or just noise?* This demo is the stats
layer that turns "B looks better" into a defensible decision, the same analysis
used in experimentation / conversion-rate-optimization work.

## What it does
1. Simulates a control vs. treatment test (conversions + revenue per converter).
2. **Two-proportion z-test** for conversion, with a **chi-square** cross-check.
3. **95% confidence interval** for the absolute difference + relative lift.
4. **Welch t-test** on the secondary metric (revenue per converter).
5. **Bayesian Beta-Binomial** read: **P(treatment > control)** + a 95% credible
   interval on the lift - the decision-useful number a p-value can't give directly.
6. Prints a plain-English **verdict** at alpha = 0.05.

## Frequentist *and* Bayesian, on purpose
A p-value answers *"how surprising is this data if there were no effect?"* - **not**
*"how likely is B better?"*. So the demo reports both: the z-test/CI/chi-square say the
lift **isn't noise**, and the Bayesian posterior says **how confident** we are that B wins
**and by how much** (`P(B>A)` + credible interval). On this seed the two CIs land on top of
each other - a clean sanity check that the two schools agree.

## Results (reproducible, seed = 11)
- Detects a **significant conversion lift** (+23.9%, p < 0.001), `P(B>A) = 100%`, and a
  ship recommendation; revenue-per-converter checked separately.
- Output: [`results/RESULTS.md`](results/RESULTS.md), `results/conversion.png`
  (rates with 95% CI error bars), and **`results/ab_test.html`** - an interactive
  **Plotly** view (conversion bars + Bayesian posteriors), GitHub-Pages ready.

## Run
```bash
pip install -r requirements.txt
python ab_test_demo.py
```

## Using your own test
Replace `simulate()` with your real numbers: pass conversion counts (`x_c, n_c,
x_t, n_t`) and revenue arrays into the test functions - the math is unchanged.

## Stack
Python · NumPy · SciPy · matplotlib · Plotly
