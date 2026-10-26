"""Train, tune and evaluate the churn model, then save it with its decision threshold.

Run from the repo root:  python src/train.py
Outputs:
  models/churn_model.joblib     fitted pipeline + tuned threshold
  outputs/metrics.json          test-set metrics for baseline and tuned models
  images/*.png                  charts used in the README
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import MODEL_PATH, ROOT, SEED, build_model, load_data  # noqa: E402

IMG = ROOT / "images"
OUT = ROOT / "outputs"
TARGET_RECALL = 0.80  # catch at least 80% of churners

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
})


def evaluate(y_true, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": round(accuracy_score(y_true, pred), 3),
        "precision": round(precision_score(y_true, pred), 3),
        "recall": round(recall_score(y_true, pred), 3),
        "f1": round(f1_score(y_true, pred), 3),
        "roc_auc": round(roc_auc_score(y_true, proba), 3),
    }


def pick_threshold(y_true, proba, target_recall=TARGET_RECALL) -> float:
    """Highest threshold that still reaches the target recall (keeps precision as high as possible)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    ok = recall[:-1] >= target_recall
    return float(thresholds[ok].max()) if ok.any() else 0.5


def lift_table(y_true, proba) -> pd.DataFrame:
    """Share of all churners captured when contacting the top X% riskiest customers."""
    order = np.argsort(-proba)
    y_sorted = np.asarray(y_true)[order]
    rows = []
    for pct in (10, 20, 30, 40, 50):
        n = int(len(y_sorted) * pct / 100)
        captured = y_sorted[:n].sum() / y_sorted.sum()
        rows.append({"top_pct_contacted": pct,
                     "pct_of_churners_captured": round(captured * 100, 1),
                     "lift": round(captured / (pct / 100), 2)})
    return pd.DataFrame(rows)


def plot_threshold(y_true, proba, threshold) -> None:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(thresholds, precision[:-1], label="Precision", color="#264653")
    ax.plot(thresholds, recall[:-1], label="Recall (churners caught)", color="#e76f51")
    ax.axvline(0.5, ls=":", color="grey")
    ax.text(0.505, 0.05, "default 0.5", color="grey", fontsize=9)
    ax.axvline(threshold, ls="--", color="#2a9d8f")
    ax.text(threshold - 0.01, 0.12, f"tuned {threshold:.2f}", color="#2a9d8f", fontsize=9, ha="right")
    ax.set(title="Choosing the decision threshold (test set)",
           xlabel="Predicted churn probability threshold", ylabel="Score", ylim=(0, 1.02))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMG / "threshold_tradeoff.png")
    plt.close(fig)


def plot_lift(lift: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(lift["top_pct_contacted"].astype(str) + "%", lift["pct_of_churners_captured"],
           color="#2a9d8f", label="Model ranking")
    ax.plot(lift["top_pct_contacted"].astype(str) + "%", lift["top_pct_contacted"],
            color="grey", ls="--", marker="o", label="Random outreach")
    for i, v in enumerate(lift["pct_of_churners_captured"]):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set(title="Churners captured by contacting the riskiest customers",
           xlabel="Share of customers contacted (highest risk first)",
           ylabel="% of all churners captured", ylim=(0, 105))
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(IMG / "lift_chart.png")
    plt.close(fig)


def plot_importance(model, X_test, y_test) -> pd.Series:
    result = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                                    n_repeats=10, random_state=SEED)
    imp = pd.Series(result.importances_mean, index=X_test.columns).sort_values()
    top = imp.tail(10)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(top.index, top.values, color="#264653")
    ax.set(title="What drives churn predictions (permutation importance)",
           xlabel="Drop in ROC-AUC when the feature is shuffled")
    fig.tight_layout()
    fig.savefig(IMG / "feature_importance.png")
    plt.close(fig)
    return imp.sort_values(ascending=False)


def one_standard_error(cv_results) -> int:
    """Pick the most regularized model whose CV score is within 1 std of the best (1-SE rule)."""
    res = pd.DataFrame(cv_results)
    best = res["mean_test_score"].idxmax()
    cutoff = res.loc[best, "mean_test_score"] - res.loc[best, "std_test_score"]
    ok = res[res["mean_test_score"] >= cutoff]
    ok = ok.assign(C=ok["param_classifier__C"].astype(float))
    return int(ok.sort_values(["C", "mean_test_score"], ascending=[True, False]).index[0])


def plot_churn_rates(X: pd.DataFrame, y: pd.Series) -> dict:
    """Observed churn rate by the factors the model relies on most."""
    df = X.assign(churn=y)
    df["tenure_band"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 72],
                               labels=["0–12 mo", "13–24 mo", "25–48 mo", "49–72 mo"])
    panels = [("Contract", "Contract type"), ("tenure_band", "Tenure"),
              ("InternetService", "Internet service"), ("PaymentMethod", "Payment method")]
    overall = y.mean() * 100
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    rates = {}
    for ax, (col, title) in zip(axes, panels):
        r = df.groupby(col, observed=True)["churn"].mean() * 100
        rates[col] = r.round(1).to_dict()
        labels = [str(i).replace(" (automatic)", "\n(auto)").replace(" check", "\ncheck") for i in r.index]
        ax.bar(labels, r.values, color=["#e76f51" if v > overall else "#2a9d8f" for v in r.values])
        ax.axhline(overall, ls="--", color="grey", linewidth=1)
        for i, v in enumerate(r.values):
            ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Churn rate (%)")
    axes[-1].text(len(r) - 0.5, overall + 1.5, f"overall {overall:.1f}%", color="grey",
                  fontsize=8, ha="right")
    fig.suptitle("Who churns? Observed churn rate by key factor", fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMG / "churn_drivers.png")
    plt.close(fig)
    return rates


def main() -> None:
    IMG.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=SEED, stratify=y)

    # 1. Baseline: the original notebook model (default settings, 0.5 threshold)
    baseline = build_model().fit(X_train, y_train)
    base_metrics = evaluate(y_test, baseline.predict_proba(X_test)[:, 1], 0.5)

    # 2. Hyperparameter tuning with 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = GridSearchCV(
        build_model(),
        param_grid={"classifier__C": [0.01, 0.1, 1.0, 10.0],
                    "classifier__class_weight": [None, "balanced"]},
        scoring="roc_auc", cv=cv, n_jobs=-1, refit=one_standard_error)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    best_params = {k.split("__")[1]: v for k, v in grid.best_params_.items()}
    best_cv_auc = grid.cv_results_["mean_test_score"][grid.best_index_]

    # 3. Pick the decision threshold on out-of-fold training predictions (no test leakage)
    oof = cross_val_predict(build_model(**best_params),
                            X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = pick_threshold(y_train, oof)

    # 4. Evaluate on the untouched test set
    proba = best.predict_proba(X_test)[:, 1]
    tuned_metrics = evaluate(y_test, proba, threshold)
    lift = lift_table(y_test, proba)

    plot_threshold(y_test, proba, threshold)
    plot_lift(lift)
    importance = plot_importance(best, X_test, y_test)
    rates = plot_churn_rates(X, y)

    joblib.dump({"model": best, "threshold": threshold}, MODEL_PATH)
    metrics = {
        "baseline_logistic_regression": base_metrics,
        "tuned_logistic_regression": tuned_metrics,
        "best_params": best_params,
        "cv_roc_auc": round(float(best_cv_auc), 3),
        "lift": lift.to_dict(orient="records"),
        "top_features": importance.head(5).round(4).to_dict(),
        "churn_rates_pct": rates,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    print("Best params:", best_params, "| CV ROC-AUC:", metrics["cv_roc_auc"])
    print(pd.DataFrame({"baseline": base_metrics, "tuned": tuned_metrics}))
    print(lift.to_string(index=False))
    print("\nTop features:\n", importance.head(8).round(4).to_string())
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
