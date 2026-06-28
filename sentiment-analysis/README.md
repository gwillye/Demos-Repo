# Sentiment Analysis (TF-IDF + Logistic Regression)

A compact, self-contained **NLP demo** that classifies product-review sentiment and surfaces the words driving each prediction.

## Real-world context
Review / NPS / social-comment mining is core to marketing analytics — turning unstructured customer text into a positive-vs-negative signal and, crucially, *why*. This demo reproduces that workflow end to end: text → TF-IDF features → model → evaluation → interpretable top words.

## What it does
1. Generates a seeded, synthetic review corpus (2,000 reviews, balanced).
2. Trains a **TF-IDF (1–2 gram) + Logistic Regression** pipeline.
3. Evaluates with **ROC-AUC** + a precision/recall/F1 report.
4. Extracts the **most informative positive/negative words** from the model coefficients.

## Results (reproducible, seed = 42)
See [`results/RESULTS.md`](results/RESULTS.md) and `results/roc_curve.png`.

## Run
```bash
pip install -r requirements.txt
python sentiment_demo.py
```

## Using your own data
The corpus is synthetic *only for reproducibility*. To run on real reviews, replace `make_data()` with a loader returning a list of texts + a binary label — the rest of the pipeline is unchanged.

## Stack
Python · scikit-learn · NumPy · matplotlib
