# 📈 Marketing Mix Modeling (MMM)

Estimate **how much each marketing channel contributes to sales** and its **ROI**, so budget can move toward what actually works — the core question behind every media-mix / growth-marketing decision.

## The problem
You spend across TV, Search, Social and Email and see total weekly sales. But spend effects **carry over** (a campaign keeps working for weeks — *adstock*) and channels overlap, so you can't just read ROI off a spreadsheet. MMM untangles it with a regression on adstock-transformed spend.

## What this does
1. Generates realistic **synthetic** weekly data (160 weeks) with per-channel **adstock carryover** and a **known ground truth** for each channel's effect, plus noise.
2. Applies the adstock transform and fits a **linear MMM** (`scikit-learn`).
3. Recovers each channel's **estimated effect**, **contribution to sales**, and **ROI**.
4. **Self-check:** confirms the model recovers the true channel ranking (R² > 0.90).
5. Exports an interactive **Plotly** chart (`mmm_contributions.html`).

## Results (reproducible, seed = 7)
| channel | est. β | true β | contribution | ROI |
|---|---|---|---|---|
| Email | 7.63 | 8.0 | 1,221,553 | **8.48** |
| TV | 2.99 | 3.0 | 1,027,783 | 7.42 |
| Search | 5.15 | 5.0 | 866,220 | 6.43 |
| Social | 2.43 | 2.0 | 586,577 | 3.46 |

**R² = 0.921.** The model recovers the true per-channel effects and ranks ROI correctly — **Email** is the most efficient channel, **Social** the least. Actionable read: shift marginal budget from Social toward Email/Search.

## Run
```bash
pip install -r requirements.txt
python mmm.py        # prints the table + writes mmm_contributions.html
```

## Notes
- Synthetic data with a known ground truth is used **on purpose** here — it lets the README *prove* the method recovers the real effects. Swap `make_data()` for your real weekly spend + sales to use it for real.
- Adstock decay is fixed per channel for clarity; in practice you'd grid-search/optimize it.

## 🛠️ Stack
Python · pandas · NumPy · scikit-learn · Plotly
