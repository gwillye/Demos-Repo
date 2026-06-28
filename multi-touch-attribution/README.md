# 🎯 Multi-Touch Attribution

A customer rarely converts on the first click — they touch **Search → Social → Email → buy**. So which channel deserves the credit (and the next budget dollar)? This compares the standard attribution models, including a **Markov-chain removal-effect** model that estimates each channel's *causal* contribution rather than just where it sat in the funnel.

## Models compared
- **First-touch** — all credit to the first channel.
- **Last-touch** — all credit to the last channel (the common-but-misleading default).
- **Linear** — equal credit across the path.
- **Markov removal-effect** — model the journey as a Markov chain (start → channels → convert/null); a channel's credit = how much the conversion probability **drops when you remove it**. This captures causal/assist value the heuristics miss.

## What it does
Generates 6,000 synthetic journeys with a **known dominant channel** (Search), runs all four models, and a **self-check** confirms credits are normalized, non-negative, and that the Markov model **recovers the true driver**.

## Results (reproducible, seed = 7)
| channel | first | last | linear | **markov** |
|---|---|---|---|---|
| **Search** | 34.1% | 33.3% | 33.6% | **27.7%** |
| Social | 21.9% | 22.7% | 22.5% | 22.8% |
| Display | 15.2% | 15.1% | 15.1% | 17.7% |
| Email | 17.2% | 16.6% | 17.1% | 17.7% |
| Video | 11.7% | 12.4% | 11.7% | 14.1% |

Search is the driver under every model, but **Markov spreads credit more realistically** — it lifts assist channels (Display/Video) that the last-touch model under-credits, which is exactly the budget-reallocation insight attribution is for.

## Run
```bash
pip install -r requirements.txt
python attribution.py
```

## Notes
- Synthetic journeys with a known ground truth are used on purpose so the README can *prove* the Markov model recovers the real driver. Swap `gen_journeys()` for your real touchpoint logs (path + converted flag) to use it for real.

## 🛠️ Stack
Python · NumPy (standard Markov absorption math, no ML framework needed)
