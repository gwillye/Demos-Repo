# Customer Churn Prediction, resultados

Trabalhei com um dataset sintético de 4000 clientes, onde a taxa de churn é de 8.8%. É um caso bem desbalanceado, então vale olhar mais para recall e ROC-AUC do que para a acurácia crua.

## LogisticRegression

- ROC-AUC: 0.820

```
              precision    recall  f1-score   support

           0      0.914     0.999     0.954       912
           1      0.667     0.023     0.044        88

    accuracy                          0.913      1000
   macro avg      0.790     0.511     0.499      1000
weighted avg      0.892     0.913     0.874      1000
```

## RandomForest

- ROC-AUC: 0.774

```
              precision    recall  f1-score   support

           0      0.915     0.981     0.947       912
           1      0.227     0.057     0.091        88

    accuracy                          0.900      1000
   macro avg      0.571     0.519     0.519      1000
weighted avg      0.855     0.900     0.872      1000
```

## Importância das variáveis (RandomForest)

- `monthly_charges`: 0.380
- `tenure_months`: 0.323
- `num_services`: 0.119
- `support_calls`: 0.106
- `contract`: 0.045
- `senior`: 0.027

## Decisão de retenção baseada em custo (transformar o modelo em ROI)

Premissas: custo da oferta de R$ 25 por cliente, taxa de sucesso da retenção de 30% e valor salvo por cliente retido de R$ 200.

- Política ótima: contatar clientes com probabilidade de churn maior ou igual a 0.42, o que dá 37 de 1000 clientes do conjunto de teste.
- Valor líquido esperado: R$ 256, contra R$ -19.716 de disparar para todo mundo (R$ 19.972 melhor).
- O ROC-AUC diz que o modelo ordena bem; este cálculo diz quem de fato contatar para maximizar o retorno, que é a decisão pela qual o negócio está pagando.
