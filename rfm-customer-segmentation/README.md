# RFM Customer Segmentation

Turn a raw order history into actionable marketing segments with **RFM**
(Recency, Frequency, Monetary) scoring + a KMeans cross-check.

## Real-world context
RFM is the workhorse of CRM and retail marketing. Instead of blasting every
customer with the same campaign, you score each one on how *recently* and how
*often* they buy and how *much* they spend, then act per segment: reward
**Champions**, win back **At Risk**, re-activate **Hibernating**. This is the
exact customer-analytics workflow used in BI / growth / lifecycle-marketing roles.

## What it does
1. Builds a seeded e-commerce transaction log (loyal / occasional / lapsed buyers).
2. Aggregates to per-customer **Recency, Frequency, Monetary**.
3. Scores each dimension into quintiles (1-5) and maps the R/F/M matrix to named
   segments (Champions, Loyal, At Risk, Hibernating, ...).
4. Cross-checks with **KMeans** clustering (k=4) on standardized RFM.

## Results (reproducible, seed = 7)
- ~1,500 customers segmented; **Champions** and **Hibernating/Lost** are the
  largest groups, with clean, interpretable mean-RFM profiles per segment.
- Output: [`results/RESULTS.md`](results/RESULTS.md), `results/segments.png`,
  `results/rfm_scatter.png`, and **`results/segments.html`** - an interactive
  **Plotly** view (segment sizes + an RFM map you can hover/zoom), GitHub-Pages ready.

## Run
```bash
pip install -r requirements.txt
python rfm_demo.py
```

## Using your own data
The transactions are synthetic *only for reproducibility*. To run on real orders,
pass a DataFrame with `customer_id`, `order_date`, `amount` into `compute_rfm()`.

## Stack
Python · pandas · NumPy · scikit-learn · matplotlib · Plotly
