# Market Basket Analysis - Results

6,000 synthetic baskets over 15 products.

## Top association rules (by lift)

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

Lift > 1 means the two items are bought together more than chance would predict. The planted pairs (bread+butter, coffee+sugar, beer+chips, diapers+beer, pasta+tomato_sauce, wine+cheese) surface at the top.
