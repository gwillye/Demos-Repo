# 💰 Customer Lifetime Value (CLV / LTV)

Predict how much **future** value each customer will bring — the metric behind retention budgets, VIP programs and acquisition caps. A naive "sum of past spend" rewards customers who already **churned**; this builds a predictive CLV that discounts them, then **validates it honestly on a future holdout window**.

## Method
From a **calibration window** (first 365 days) per customer:
- **frequency** (purchase rate), **recency**, **monetary** (avg order value),
- **p(alive)** — a BG/NBD-style decay: a customer silent for many of *their own* purchase-cycles is probably gone,
- `predicted CLV = rate × horizon × p(alive) × avg_order_value × margin`.

Then it **scores against the next 180 days** (data the model never saw).

## Results (reproducible, seed = 7)
- **Top decile captures 41% of actual future revenue** — i.e. the top 10% of customers by predicted CLV deliver **~4× their share** of next-period revenue. This *gains/lift* view is how LTV models are judged in practice.
- Predicted CLV vs actual future spend correlation ≈ **0.31**.

> Honest note: individual-customer LTV is inherently noisy (one person's next 6 months is high-variance), so the **ranking/decile lift is the reliable, decision-grade signal** — not a high per-customer R². The self-check asserts the decile lift, not an inflated correlation.

## Run
```bash
pip install -r requirements.txt
python clv.py
```

## Notes
- Synthetic transactions with latent purchase-rate/churn/AOV are simulated so the holdout validation is reproducible. Swap `simulate()` for your real transaction log (`customer, timestamp, value`) to use it.

## 🛠️ Stack
Python · pandas · NumPy
