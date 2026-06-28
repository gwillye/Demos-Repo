# K-Means from Scratch

A self-contained **algorithms demo**: Lloyd's K-Means implemented in pure NumPy and validated against scikit-learn's implementation.

## Real-world context
Clustering powers segmentation (customers, products, regions). Implementing K-Means from scratch shows the mechanics behind the library call — assignment, centroid update, convergence and inertia.

## What it does
1. Generates 4 Gaussian blobs (`make_blobs`).
2. Runs **Lloyd's algorithm** in NumPy (assign → update → check convergence).
3. Compares the final **inertia** to scikit-learn's `KMeans` (should match within a few %).

## Results (reproducible, seed = 42)
See [`results/RESULTS.md`](results/RESULTS.md) and `results/clusters.png`.

## Run
```bash
pip install -r requirements.txt
python kmeans_demo.py
```

## Stack
Python · NumPy · scikit-learn (validation) · matplotlib
