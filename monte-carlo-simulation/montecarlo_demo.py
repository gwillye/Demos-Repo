"""Monte Carlo simulation — (1) estimate pi, (2) portfolio risk simulation (VaR).

Self-contained (numpy + matplotlib), seeded. Saves results/RESULTS.md and
results/portfolio_paths.png.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
rng = np.random.default_rng(SEED)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

# (1) estimate pi
n = 2_000_000
pts = rng.random((n, 2))
inside = ((pts ** 2).sum(axis=1) <= 1).sum()
pi_est = 4 * inside / n

# (2) portfolio Monte Carlo
S0, mu, sigma, days, sims = 10_000.0, 0.08, 0.20, 252, 20_000
daily = rng.normal(mu / days, sigma / np.sqrt(days), size=(sims, days))
paths = S0 * np.cumprod(1 + daily, axis=1)
final = paths[:, -1]
mean_v = final.mean()
var5 = np.percentile(final, 5)           # 5th percentile of value
loss_var = S0 - var5                      # 95% VaR (loss vs initial)

plt.figure(figsize=(9, 4))
for i in range(60):
    plt.plot(paths[i], color="#4c78a8", alpha=0.15)
plt.axhline(S0, color="gray", ls="--")
plt.title(f"Portfolio Monte Carlo — mean {mean_v:,.0f}, 95% VaR {loss_var:,.0f}")
plt.xlabel("trading day")
plt.ylabel("value")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "portfolio_paths.png"), dpi=110)
plt.close()

with open(os.path.join(OUT, "RESULTS.md"), "w", encoding="utf-8") as f:
    f.write("# Monte Carlo Simulation — Results\n\n")
    f.write(f"## 1. Estimate pi\n{n:,} random points → **pi ≈ {pi_est:.4f}** "
            f"(error {abs(pi_est - np.pi):.4f}).\n\n")
    f.write("## 2. Portfolio risk\n")
    f.write(f"Initial {S0:,.0f}, μ=8%/yr, σ=20%/yr, {days} days, {sims:,} simulations.\n\n")
    f.write(f"- Mean final value: **{mean_v:,.0f}**\n")
    f.write(f"- 5th percentile: **{var5:,.0f}**\n")
    f.write(f"- **95% Value-at-Risk: {loss_var:,.0f}** (potential loss vs initial)\n")

print(f"pi={pi_est:.4f} mean_final={mean_v:.0f} VaR95={loss_var:.0f}")
print("self-check:", "OK" if abs(pi_est - np.pi) < 0.01 else "FAIL")
