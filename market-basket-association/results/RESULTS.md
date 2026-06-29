# Market Basket Analysis, resultados

Gerei 6.000 cestas sintéticas usando 15 produtos e rodei a análise de regras de associação para descobrir quais itens costumam aparecer juntos no mesmo carrinho.

## Principais regras de associação (ordenadas por lift)

```
  antecedent   consequent  support  confidence  lift
      cheese         wine    0.254       0.659  1.99
        wine       cheese    0.254       0.766  1.99
       bread       butter    0.270       0.782  1.97
      butter        bread    0.270       0.679  1.97
tomato_sauce        pasta    0.261       0.667  1.96
       pasta tomato_sauce    0.261       0.766  1.96
       sugar       coffee    0.253       0.652  1.96
      coffee        sugar    0.253       0.759  1.96
     diapers         beer    0.284       0.789  1.71
        beer      diapers    0.284       0.616  1.71
        beer        chips    0.292       0.633  1.62
       chips         beer    0.292       0.745  1.62
```

Lift maior que 1 quer dizer que os dois itens são comprados juntos com mais frequência do que o acaso preveria. Os pares plantados de propósito (bread+butter, coffee+sugar, beer+chips, diapers+beer, pasta+tomato_sauce, wine+cheese) aparecem todos no topo, que é exatamente o que a gente esperava ver.
