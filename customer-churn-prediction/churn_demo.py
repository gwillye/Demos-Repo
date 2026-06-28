"""
Customer Churn Prediction - demo
--------------------------------
Generates a seeded, telecom-style customer dataset and trains an ML pipeline to
predict churn (Logistic Regression vs. Random Forest), then reports ROC-AUC,
a classification report, and feature importance.

Real-world context: in marketing / CRM analytics, predicting which customers are
about to churn lets the business target retention offers at the right people --
the same customer-analytics work used in BI/Data Science roles.

The data is SYNTHETIC (seeded for reproducibility); to use real data, replace
`make_data()` with your own loader returning the same columns + a `churn` label.

Run:  python churn_demo.py
Out:  results/RESULTS.md, results/roc_curves.png, results/feature_importance.png,
      results/churn.html (interactive)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RNG = np.random.default_rng(42)
N = 4000


def make_data(n):
    """Synthetic telecom-style customers with a realistic churn relationship."""
    tenure = RNG.integers(1, 72, n)                      # months as a customer
    monthly = RNG.normal(70, 25, n).clip(15, 150)        # monthly charges (R$)
    contract = RNG.choice([0, 1, 2], n, p=[.55, .25, .20])  # 0=monthly 1=1yr 2=2yr
    support = RNG.poisson(1.2, n)                         # support calls last quarter
    services = RNG.integers(1, 8, n)                      # add-on services
    senior = RNG.choice([0, 1], n, p=[.84, .16])
    # Churn is driven by short tenure, high charges, monthly contract, many support
    # calls, fewer services (logit -> probability).
    logit = (-2.0
             - 0.045 * tenure
             + 0.012 * (monthly - 70)
             + 1.10 * (contract == 0)
             - 0.40 * (contract == 2)
             + 0.35 * support
             - 0.08 * services
             + 0.30 * senior)
    p = 1 / (1 + np.exp(-logit))
    churn = (RNG.random(n) < p).astype(int)
    return pd.DataFrame({
        "tenure_months": tenure, "monthly_charges": monthly.round(2),
        "contract": contract, "support_calls": support,
        "num_services": services, "senior": senior, "churn": churn,
    })


def main():
    df = make_data(N)
    X, y = df.drop(columns="churn"), df["churn"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=.25, random_state=42, stratify=y)

    models = {
        "LogisticRegression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000))]),
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=300, random_state=42, n_jobs=-1))]),
    }

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    lines = ["# Customer Churn Prediction - Results\n\n",
             f"Synthetic dataset: **{N} customers**, churn rate **{y.mean():.1%}**.\n"]
    roc = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        proba = m.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, proba)
        roc[name] = (roc_curve(yte, proba), auc)
        lines.append(f"\n## {name}\n\n- ROC-AUC: **{auc:.3f}**\n\n```\n")
        lines.append(classification_report(yte, m.predict(Xte), digits=3))
        lines.append("```\n")

    rf = models["RandomForest"].named_steps["clf"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    lines.append("\n## Feature importance (RandomForest)\n\n")
    lines += [f"- `{k}`: {v:.3f}\n" for k, v in imp.items()]

    plt.figure(figsize=(6, 5))
    for name, ((fpr, tpr, _), auc) in roc.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=.4)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC - churn models"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "roc_curves.png", dpi=110); plt.close()

    plt.figure(figsize=(6, 4)); imp.sort_values().plot.barh()
    plt.title("Feature importance (RandomForest)"); plt.tight_layout()
    plt.savefig(out / "feature_importance.png", dpi=110); plt.close()

    # interactive view (GitHub-Pages ready): ROC curves + feature importance
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("ROC curve", "Feature importance (RandomForest)"))
    for (name, ((fpr, tpr, _), auc)), col in zip(roc.items(), ["#EF553B", "#00CC96"]):
        fig.add_scatter(x=fpr, y=tpr, name=f"{name} (AUC {auc:.3f})",
                        line=dict(color=col, width=2.5), row=1, col=1)
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="#bbb"),
                    showlegend=False, row=1, col=1)
    imp_asc = imp.sort_values()
    fig.add_bar(x=imp_asc.values, y=imp_asc.index, orientation="h",
                marker_color="#636EFA", showlegend=False, row=1, col=2)
    fig.update_xaxes(title="False positive rate", row=1, col=1)
    fig.update_yaxes(title="True positive rate", row=1, col=1)
    fig.update_layout(template="plotly_white", height=470, legend=dict(x=.45, y=.1),
                      title=f"Customer churn - ROC & drivers (churn rate {y.mean():.1%})")
    fig.write_html(out / "churn.html", include_plotlyjs="cdn")

    (out / "RESULTS.md").write_text("".join(lines), encoding="utf-8")
    best = max(a for (_, a) in roc.values())
    print(f"Done. Best ROC-AUC: {best:.3f}")
    print("See results/RESULTS.md + results/*.png + results/churn.html")


if __name__ == "__main__":
    main()
