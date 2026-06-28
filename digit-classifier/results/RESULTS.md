# Digit Classifier — Results

Dataset: scikit-learn `load_digits` — **1797** 8x8 images, 10 classes. Model: **SVM (RBF, gamma=0.001)**.

- **Test accuracy: 0.989**

```
              precision    recall  f1-score   support

           0      1.000     1.000     1.000        45
           1      0.958     1.000     0.979        46
           2      1.000     1.000     1.000        44
           3      1.000     1.000     1.000        46
           4      1.000     1.000     1.000        45
           5      1.000     0.978     0.989        46
           6      1.000     0.978     0.989        45
           7      0.978     1.000     0.989        45
           8      0.976     0.953     0.965        43
           9      0.978     0.978     0.978        45

    accuracy                          0.989       450
   macro avg      0.989     0.989     0.989       450
weighted avg      0.989     0.989     0.989       450

```

See `confusion_matrix.png` and `samples.png`.
