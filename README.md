# Demos-Repo: small, runnable data science and ML demos

This is a growing collection of self-contained demos that actually run. Each one is a focused project drawn from my work in data science, ML, NLP, analytics and marketing. Every demo runs end to end on synthetic or bundled data (no secrets), prints real metrics, and saves its outputs.

I built this as a portfolio of runnable ideas rather than a dump of coursework. Each folder has its own README with a real-world framing, a `requirements.txt`, and a `results/` folder holding the numbers and plots it produced.

## Demos

| Demo | Problem | Stack | Real-world angle |
|---|---|---|---|
| [customer-churn-prediction](customer-churn-prediction/) | Predict churn (LogReg vs RandomForest) | scikit-learn | CRM / retention (marketing) |
| [rfm-customer-segmentation](rfm-customer-segmentation/) | RFM + KMeans customer segments | pandas, scikit-learn | CRM / retail targeting |
| [ab-test-analyzer](ab-test-analyzer/) | A/B test: z-test + chi² + CI + lift + ship verdict | numpy, scipy | growth / CRO experimentation |
| [recommender-cf](recommender-cf/) | Collaborative-filtering recommender | numpy, scikit-learn | e-commerce / content reco |
| [sentiment-analysis](sentiment-analysis/) | Text sentiment classification | NLP, scikit-learn | social / brand monitoring |
| [text-summarizer](text-summarizer/) | Extractive text summarization | NLP | content ops |
| [digit-classifier](digit-classifier/) | Image digit classification | scikit-learn / CNN | computer-vision baseline |
| [pca-digits-viz](pca-digits-viz/) | Dimensionality reduction + viz | scikit-learn, matplotlib | EDA / explainability |
| [eda-dashboard](eda-dashboard/) | Exploratory data analysis dashboard | pandas, plotly | analytics reporting |
| [time-series-forecasting](time-series-forecasting/) | Forecast a seasonal series | statsmodels / sklearn | demand / finance |
| [astar-pathfinding](astar-pathfinding/) | A* search on a grid | python (algorithms) | classic CS / optimization |
| [market-basket-association](market-basket-association/) | Association rules (Apriori: support/confidence/lift) | python | cross-sell / recommendation |
| [kmeans-from-scratch](kmeans-from-scratch/) | K-means clustering implemented from scratch | numpy | unsupervised / fundamentals |
| [monte-carlo-simulation](monte-carlo-simulation/) | Monte Carlo estimation / risk simulation | numpy | finance / risk |
| [linear-programming](linear-programming/) | Linear-programming optimization | scipy | operations research |
| [cohort-retention-analysis](cohort-retention-analysis/) | Cohort retention matrix + curve | pandas | growth / lifecycle |
| [marketing-mix-modeling](marketing-mix-modeling/) | Adstock MMM: channel contribution + ROI | scikit-learn, plotly | marketing / media spend |
| [multi-touch-attribution](multi-touch-attribution/) | Multi-touch attribution across channels | pandas | marketing / attribution |
| [sales-demand-forecasting](sales-demand-forecasting/) | GBR vs seasonal-naive demand forecast | scikit-learn, plotly | retail / demand planning |
| [uplift-modeling](uplift-modeling/) | T-learner uplift for campaign targeting (Qini) | scikit-learn | marketing / causal targeting |
| [customer-lifetime-value](customer-lifetime-value/) | Predictive CLV/LTV (RFM + p(alive)), decile-lift validated | scikit-learn | marketing / retention value |

There are 21 demos here so far, and the repo grows over time.

## Running any demo

```bash
cd <demo-folder>
pip install -r requirements.txt
python <main_script>.py   # see the folder's README
```

## Data and privacy

All demos use synthetic or public/bundled data, so there are no secrets and no real personal data. They exist to show method and reasoning, reproducibly.

---

Part of [@gwillye](https://github.com/gwillye)'s portfolio. Author: Gabriel Willye, Data Scientist / AI.
