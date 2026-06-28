# Recommender — Item-Item Collaborative Filtering

A self-contained **recommender-systems demo**: it predicts user–item ratings with item-item collaborative filtering and beats a sensible baseline.

## Real-world context
Recommendation drives engagement and cross-sell in e-commerce, streaming and content. Item-item collaborative filtering ("users who liked this also liked…") is the classic, explainable workhorse behind many production recommenders.

## What it does
1. Builds a sparse synthetic ratings matrix (200 users × 60 items) with latent structure.
2. Holds out 20% of observed ratings; computes **item-item cosine similarity** on mean-centered ratings.
3. Predicts held-out ratings and compares **RMSE** against an item-mean baseline.
4. Generates **top-N recommendations** for a sample user.

## Results (reproducible, seed = 42)
See [`results/RESULTS.md`](results/RESULTS.md) and `results/rmse.png`.

## Run
```bash
pip install -r requirements.txt
python recommender_demo.py
```
Replace `make_ratings()` with a loader for a real user×item matrix to use your own data.

## Stack
Python · NumPy · matplotlib
