# A/B Test Analyzer - Results

| Group | Users | Conversions | Rate |
|---|---|---|---|
| Control (A)   | 6,000 | 593 | 9.883% |
| Treatment (B) | 6,000 | 735 | 12.250% |

## Conversion
- **Relative lift: +23.9%** (absolute +2.367%)
- 95% CI for the difference: **[+1.245%, +3.488%]**
- Two-proportion z-test: z = 4.13, **p = 0.0000**
- Chi-square cross-check: chi2 = 17.07, p = 0.0000
- **Statistically significant at alpha=0.05? YES**

## Revenue per converter (secondary metric)
- Control mean R$ 43.42 vs Treatment R$ 45.96
- Welch t-test: t = 3.86, p = 0.0001

## Verdict
Ship B - the lift is real and unlikely to be noise.
