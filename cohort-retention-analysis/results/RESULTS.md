# Cohort Retention Analysis, resultados

Analisei 5000 usuários distribuídos em 12 cohorts mensais, acompanhando cada um por um horizonte de 12 meses. A ideia é ver quanta gente continua ativa conforme os meses passam, agrupando pelo mês de entrada.

## Retenção por cohort (% retida, do mês 0 ao N)

```
month_since     0     1     2     3     4     5     6     7    8    9    10   11
cohort                                                                          
0            100.0  71.4  50.2  43.0  26.7  20.9  13.4  13.0  7.5  7.0  4.6  4.0
1            100.0  68.3  49.0  37.1  26.4  18.8  15.0  12.2  7.9  4.3  4.1  0.0
2            100.0  70.2  53.8  38.1  24.2  18.3  17.6  15.6  7.1  5.1  0.0  0.0
3            100.0  73.0  57.1  39.9  30.3  21.5  13.9  12.6  7.6  0.0  0.0  0.0
4            100.0  75.8  55.9  41.4  29.8  20.8  14.5  10.4  0.0  0.0  0.0  0.0
5            100.0  70.6  58.1  41.2  28.9  23.8  19.5   0.0  0.0  0.0  0.0  0.0
6            100.0  75.8  55.3  42.5  28.0  22.8   0.0   0.0  0.0  0.0  0.0  0.0
7            100.0  75.7  55.2  45.0  31.4   0.0   0.0   0.0  0.0  0.0  0.0  0.0
8            100.0  76.9  60.7  41.4   0.0   0.0   0.0   0.0  0.0  0.0  0.0  0.0
9            100.0  76.6  60.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0  0.0
10           100.0  77.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0  0.0
11           100.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  0.0  0.0  0.0  0.0
```

Os zeros à direita são apenas meses que ainda não aconteceram para os cohorts mais recentes, não quedas reais de retenção.

## Curva média de retenção

- Mês 1: 67.6%  |  Mês 3: 30.8%  |  Mês 6: 7.8%

Os cohorts mais novos retêm um pouco melhor. O "produto" sintético foi desenhado para melhorar com o tempo, e a análise consegue recuperar essa tendência.
