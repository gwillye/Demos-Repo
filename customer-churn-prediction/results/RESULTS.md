# Customer Churn Prediction - Results

Synthetic dataset: **4000 customers**, churn rate **8.8%**.

## LogisticRegression

- ROC-AUC: **0.820**

```
              precision    recall  f1-score   support

           0      0.914     0.999     0.954       912
           1      0.667     0.023     0.044        88

    accuracy                          0.913      1000
   macro avg      0.790     0.511     0.499      1000
weighted avg      0.892     0.913     0.874      1000
```

## RandomForest

- ROC-AUC: **0.774**

```
              precision    recall  f1-score   support

           0      0.915     0.981     0.947       912
           1      0.227     0.057     0.091        88

    accuracy                          0.900      1000
   macro avg      0.571     0.519     0.519      1000
weighted avg      0.855     0.900     0.872      1000
```

## Feature importance (RandomForest)

- `monthly_charges`: 0.380
- `tenure_months`: 0.323
- `num_services`: 0.119
- `support_calls`: 0.106
- `contract`: 0.045
- `senior`: 0.027
