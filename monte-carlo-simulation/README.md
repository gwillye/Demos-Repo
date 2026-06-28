# Monte Carlo Simulation (π + portfolio risk)

A self-contained **simulation / quantitative demo**: Monte Carlo to estimate π, then to simulate portfolio risk and compute Value-at-Risk.

## Real-world context
Monte Carlo is the workhorse for risk and uncertainty — pricing, VaR, capacity planning, A/B power. This demo shows both the intuition (estimating π by sampling) and a practical finance use (distribution of portfolio outcomes + 95% VaR).

## What it does
1. **Estimates π** by sampling 2M points in the unit square.
2. **Simulates 20,000 portfolio paths** (geometric returns, μ=8%/yr, σ=20%/yr, 252 days).
3. Reports the **mean final value** and the **95% Value-at-Risk**, and plots sample paths.

## Results (reproducible, seed = 42)
See [`results/RESULTS.md`](results/RESULTS.md) and `results/portfolio_paths.png`.

## Run
```bash
pip install -r requirements.txt
python montecarlo_demo.py
```

## Stack
Python · NumPy · matplotlib
