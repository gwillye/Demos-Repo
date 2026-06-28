# Customer Churn Prediction

A compact, end-to-end **machine-learning demo** that predicts customer churn and
explains the drivers behind it.

## Real-world context
In marketing and CRM analytics, knowing *which* customers are likely to leave
lets a business spend its retention budget where it matters — targeted offers,
proactive support, win-back campaigns. This demo reproduces that customer-analytics
workflow end to end: data → model → evaluation → business-readable drivers.

## What it does
1. Builds a seeded, telecom-style customer dataset (tenure, monthly charges,
   contract type, support calls, add-on services, senior flag).
2. Trains and compares two models in clean scikit-learn pipelines:
   **Logistic Regression** (interpretable baseline) and **Random Forest**.
3. Evaluates with **ROC-AUC** + a full precision/recall/F1 report.
4. Surfaces the **feature importances** so the result is actionable, not a black box.

## Results (reproducible, seed = 42)
- Random Forest **ROC-AUC ≈ 0.82**; logistic regression close behind.
- Top churn drivers: **tenure** (longer = loyal), **contract type**
  (month-to-month churns most), **support calls**, **monthly charges**.
- Full output: [`results/RESULTS.md`](results/RESULTS.md),
  `results/roc_curves.png`, `results/feature_importance.png`.

## Run
```bash
pip install -r requirements.txt
python churn_demo.py
```

## Using your own data
The data is synthetic *only for reproducibility*. To run on real customers,
replace `make_data()` with a loader that returns the same feature columns plus a
binary `churn` column — the rest of the pipeline is unchanged.

## Stack
Python · pandas · NumPy · scikit-learn · matplotlib
