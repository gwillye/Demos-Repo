# RFM Customer Segmentation, resultados

Parti de 12.046 pedidos sintéticos espalhados por 1500 clientes, usando 2025-01-01 como data de referência para calcular a recência. A partir do RFM (recency, frequency, monetary) montei segmentos por regra e também rodei um KMeans para comparar.

## Tamanho dos segmentos (matriz RFM)

```
segment
Champions              505
Hibernating / Lost     467
Loyal                  248
Potential Loyalists    112
At Risk                 76
Others                  57
New / Promising         35
```

## Perfil dos segmentos (médias de R/F/M)

```
                     recency  frequency  monetary
segment                                          
At Risk                201.1        5.7     650.1
Champions               36.4       16.2    2903.5
Hibernating / Lost     335.1        2.1     167.2
Loyal                  106.1        6.7     885.7
New / Promising         61.6        3.5     293.0
Others                 261.1        3.7     416.3
Potential Loyalists    121.2        3.7     300.1
```

## Perfil dos clusters do KMeans (médias de R/F/M)

```
         recency  frequency  monetary
cluster                              
0          145.8        4.6     492.5
1           33.7       20.3    3983.2
2          383.3        2.1     167.7
3           43.0       12.6    1959.7
```

Dá para notar que os clusters do KMeans batem bem com a leitura dos segmentos por regra: o cluster 1 (recência baixa, frequência alta, valor alto) é claramente o grupo de Champions, e o cluster 2 (recência altíssima, pouca compra) corresponde aos Hibernating / Lost.
