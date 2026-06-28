# EDA Dashboard (pandas + matplotlib)

A self-contained **data-visualization demo**: it generates a synthetic sales dataset and renders a compact 2×2 **exploratory dashboard** plus key stats.

## Real-world context
The first step of any analytics/BI engagement is exploratory data analysis — seeing distributions, category breakdowns, correlations and trends before modeling. This demo reproduces that "first look" as a single shareable panel.

## What it does
1. Builds a seeded, e-commerce-style dataset (category, price, units, revenue, satisfaction, month).
2. Renders four panels: **revenue distribution**, **revenue by category**, **correlation heatmap**, **monthly trend**.
3. Writes key stats (total revenue, top category, avg satisfaction, price↔satisfaction correlation).

## Results (reproducible, seed = 42)
See [`results/RESULTS.md`](results/RESULTS.md) and `results/dashboard.png`.

## Run
```bash
pip install -r requirements.txt
python eda_demo.py
```
Swap `make_data()` for a `pd.read_csv(...)` to explore your own dataset.

## Stack
Python · pandas · NumPy · matplotlib
