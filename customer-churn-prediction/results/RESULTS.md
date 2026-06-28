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

## Cost-based retention decision (turn the model into ROI)

Assumptions: offer cost **R$ 25**/customer · retention success **30%** · value saved per retained customer **R$ 200**.

- **Optimal policy: contact customers with churn prob ≥ 0.42** → **37 of 1000** test customers.
- **Expected net value: R$ 256** vs R$ -19,716 from blasting everyone (**R$ 19,972 better**).
- ROC-AUC says the model *ranks* well; this says **who to actually contact** to maximise return — the decision the business is paying for.
