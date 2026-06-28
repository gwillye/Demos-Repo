# PCA — Dimensionality Reduction (digits)

A self-contained **unsupervised / visualization demo**: PCA projects 64-dimensional digit vectors to 2D and shows how few components retain most of the information.

## Real-world context
High-dimensional data (images, embeddings, survey items) is hard to see and expensive to model. PCA is the standard first move — compress to a handful of components for visualization, denoising and faster downstream models.

## What it does
1. Loads `load_digits` (bundled with scikit-learn) and standardizes it.
2. Projects to **2 principal components** and scatters them colored by digit.
3. Computes the **cumulative explained variance** and how many components reach 90%.

## Results (reproducible, seed = 42)
See [`results/RESULTS.md`](results/RESULTS.md), `results/pca_scatter.png`, `results/scree.png`.

## Run
```bash
pip install -r requirements.txt
python pca_demo.py
```

## Stack
Python · scikit-learn · NumPy · matplotlib
