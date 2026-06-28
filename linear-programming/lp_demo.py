"""Linear programming demo — production-mix optimization with scipy.optimize.linprog.

Maximizes profit subject to labor/material/demand constraints. Self-contained
(scipy + matplotlib). Saves results/RESULTS.md and results/allocation.png.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

# Two products A, B. Maximize profit 40*A + 30*B.
# Constraints: labor 2A + 1B <= 100 ; material 1A + 1B <= 80 ; demand A <= 40
products = ["A", "B"]
profit = np.array([40, 30])
A_ub = [[2, 1], [1, 1], [1, 0]]
b_ub = [100, 80, 40]

res = linprog(c=-profit, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None), (0, None)], method="highs")
alloc = res.x
total = -res.fun

plt.figure(figsize=(5, 4))
plt.bar(products, alloc, color="#4c78a8")
plt.ylabel("units to produce")
plt.title(f"Optimal production mix — profit {total:.0f}")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "allocation.png"), dpi=110)
plt.close()

with open(os.path.join(OUT, "RESULTS.md"), "w", encoding="utf-8") as f:
    f.write("# Linear Programming — Production Mix\n\n")
    f.write("Maximize profit `40A + 30B` s.t. labor `2A+B<=100`, material `A+B<=80`, demand `A<=40`.\n\n")
    f.write(f"- Optimal: **A = {alloc[0]:.0f}**, **B = {alloc[1]:.0f}**\n")
    f.write(f"- **Maximum profit: {total:.0f}**\n")
    f.write(f"- Solver status: {res.message}\n")

print(f"A={alloc[0]:.1f} B={alloc[1]:.1f} profit={total:.1f} success={res.success}")
print("self-check:", "OK" if res.success and total > 0 else "FAIL")
