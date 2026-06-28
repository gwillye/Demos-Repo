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
5. **Cost-based decision threshold** — turns the churn *scores* into a *who-to-contact*
   retention plan (offer cost vs. value saved) and finds the ROI-optimal cutoff.

## From a score to a decision
A churn model is only worth its retention budget if you act on it well. With an offer
cost of **R$ 25**, a **30%** save rate and **R$ 200** saved per retained customer, the
break-even churn probability is **~0.42** — so the demo's optimal policy contacts only
customers above that. Blasting the *whole* base loses **~R$ 20k** on the test set; the
**targeted** policy turns that into a positive return. ROC-AUC tells you the model
*ranks* well; this tells you **who to actually contact** — the decision the business pays for.

## Results (reproducible, seed = 42)
- Random Forest **ROC-AUC ≈ 0.82**; logistic regression close behind.
- Top churn drivers: **tenure** (longer = loyal), **contract type**
  (month-to-month churns most), **support calls**, **monthly charges**.
- Cost-based policy: contact churn-prob **≥ 0.42** → ~**R$ 20k** better than contacting everyone.
- Full output: [`results/RESULTS.md`](results/RESULTS.md),
  `results/roc_curves.png`, `results/feature_importance.png`, and
  **`results/churn.html`** - an interactive **Plotly** view (ROC curves +
  feature importance + a campaign-value-vs-threshold curve), GitHub-Pages ready.

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
Python · pandas · NumPy · scikit-learn · matplotlib · Plotly
