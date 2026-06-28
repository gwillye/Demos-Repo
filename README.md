# 🧪 Demos-Repo — small, runnable data-science & ML demos

A growing collection of **self-contained, actually-run** demos — each one a focused project inspired by my work in **data science, ML, NLP, analytics and marketing**. Every demo runs end-to-end (synthetic or bundled data, no secrets), prints real metrics, and saves its outputs.

> Built as a portfolio of *runnable* ideas — not coursework dumps. Each folder has its own README with a real-world framing, `requirements.txt`, and a `results/` with the numbers/plots it produced.

## 📦 Demos

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

*(more demos are being added — this repo grows over time.)*

## ▶️ Running any demo
```bash
cd <demo-folder>
pip install -r requirements.txt
python <main_script>.py   # see the folder's README
```

## 🔒 Data & privacy
All demos use **synthetic or public/bundled** data — **no secrets, no real personal data**. They exist to show method and reasoning, reproducibly.

---
*Part of [@gwillye](https://github.com/gwillye)'s portfolio. Author: Gabriel Willye — Data Scientist / AI.*
