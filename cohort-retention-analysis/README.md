# Cohort Retention Analysis

Build the **cohort retention matrix** (the triangular heatmap every growth team
lives by) from a signup + activity log, plus the average retention curve.

## Real-world context
Retention is the most important growth metric there is. Grouping users by signup
**cohort** and tracking what fraction stays active in month 0, 1, 2, ... reveals
whether the product is getting stickier over time and exactly where users drop
off. It's the backbone of lifecycle marketing, growth and subscription analytics.

## What it does
1. Builds a seeded signup + monthly-activity log (12 cohorts, 12-month horizon),
   with later cohorts retaining slightly better (an "improving product").
2. Computes the **% retained** per (cohort, months-since-signup) matrix.
3. Reports the **average retention curve** and renders the classic cohort heatmap.

## Results (reproducible, seed = 5)
- Month-1 retention ~**68%**, decaying to ~**8%** by month 6 - a realistic curve;
  the analysis recovers the planted "later cohorts retain better" trend.
- Output: [`results/RESULTS.md`](results/RESULTS.md), `results/cohort_heatmap.png`,
  `results/retention_curve.png`.

## Run
```bash
pip install -r requirements.txt
python cohort_demo.py
```

## Using your own data
Pass `build_matrix()` a DataFrame with `cohort` (signup period) and `month_since`
(one row per active user-month) - derive it from your events table.

## Stack
Python · pandas · NumPy · matplotlib
