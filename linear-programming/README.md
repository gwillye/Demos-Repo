# Linear Programming — Production-Mix Optimization

A self-contained **operations-research demo**: maximize profit under resource constraints with `scipy.optimize.linprog`.

## Real-world context
Operations and supply-chain teams constantly allocate limited resources (labor, materials, budget) to maximize an objective. Linear programming gives the provably optimal plan — the backbone of production planning, scheduling and logistics.

## What it does
1. Defines a production-mix problem (2 products, profit objective, labor/material/demand constraints).
2. Solves it with the **HiGHS** LP solver via `scipy.optimize.linprog`.
3. Reports the optimal allocation and maximum profit.

## Results (reproducible)
See [`results/RESULTS.md`](results/RESULTS.md) and `results/allocation.png`.

## Run
```bash
pip install -r requirements.txt
python lp_demo.py
```

## Stack
Python · SciPy · matplotlib
