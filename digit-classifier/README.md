# Handwritten Digit Classifier (SVM)

A self-contained **machine-learning / computer-vision demo**: an SVM classifies 8×8 handwritten digits and reports accuracy, a confusion matrix and sample predictions.

## Real-world context
Digit/character recognition is the classic entry point to image classification (and the lighter cousin of the CNN-on-MNIST project). This demo uses the dataset **bundled with scikit-learn**, so it runs offline with no downloads.

## What it does
1. Loads `load_digits` (1,797 8×8 images, 10 classes).
2. Trains an **SVM (RBF kernel)** in a stratified train/test split.
3. Reports **accuracy** + precision/recall/F1, a **confusion matrix**, and a sample-predictions grid (misses in red).

## Results (reproducible, seed = 42)
See [`results/RESULTS.md`](results/RESULTS.md), `results/confusion_matrix.png`, `results/samples.png`.

## Run
```bash
pip install -r requirements.txt
python digits_demo.py
```

## Stack
Python · scikit-learn · NumPy · matplotlib
