# Market Basket Analysis

Mine **association rules** ("customers who buy X also buy Y") from shopping
baskets, ranked by **lift** - a lightweight Apriori with no heavy dependencies.

## Real-world context
Association-rule mining powers cross-sell, bundle offers, product placement and
recommendation. The textbook result - `{diapers} -> {beer}` - is exactly this
analysis. In retail / e-commerce / marketing it turns a pile of receipts into
"put these two near each other" and "recommend Y at checkout".

## What it does
1. Builds a seeded basket log over 15 products, with a few **planted**
   co-purchase relationships.
2. Counts item and pair **support**, then derives every rule's **confidence**
   and **lift**.
3. Filters by minimum support/confidence and ranks by lift.
4. Renders an item x item **co-purchase lift heatmap**.

## Results (reproducible, seed = 3)
- The planted pairs (bread+butter, coffee+sugar, beer+chips, diapers+beer,
  pasta+tomato_sauce, wine+cheese) **surface at the top** (lift ~2x), confirming
  the miner recovers real structure rather than noise.
- Output: [`results/RESULTS.md`](results/RESULTS.md), `results/cooccurrence.png`.

## Run
```bash
pip install -r requirements.txt
python market_basket_demo.py
```

## Using your own data
Pass `mine_rules()` a list of baskets (each a set/list of item names) - e.g. one
per order from a transactions table grouped by order id.

## Stack
Python · pandas · NumPy · matplotlib
