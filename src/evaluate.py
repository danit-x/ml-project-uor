"""Shared evaluation and model comparisons (proposal sections 4.10 and 4.12).

Owner: Dharmasiri S.O.R. Positive class 1 means phishing.
Never tune on the test set; use cv_evaluate() on training data while developing.
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from . import config

CV_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
REPORT_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def cv_splitter() -> StratifiedKFold:
    """Reproducible CV folds shared by all four models."""
    return StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )


def cv_evaluate(unfitted_model, X_train, y_train) -> dict:
    """Mean and standard deviation of training-only cross-validation scores."""
    scores = cross_validate(
        unfitted_model, X_train, y_train,
        cv=cv_splitter(), scoring=CV_METRICS,
    )
    out = {}
    for metric in CV_METRICS:
        out[f"{metric}_mean"] = float(scores[f"test_{metric}"].mean())
        out[f"{metric}_std"] = float(scores[f"test_{metric}"].std())
    return out


def evaluate(model, X_test, y_test) -> dict:
    """Final held-out metrics and confusion counts for phishing (class 1)."""
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def compare(results: dict) -> pd.DataFrame:
    """Return one row per (feature_set, model) with a common metric layout."""
    table = pd.DataFrame(results).T
    table.index.names = ["feature_set", "model"]
    return table.reset_index()


def feature_delta_table(results) -> pd.DataFrame:
    """Original versus engineered scores, with engineered minus original deltas.

    Accepts the results dictionary used in main.py, or compare(results).
    Missing models in either set are retained with NaN rather than fabricated.
    """
    table = compare(results) if isinstance(results, dict) else results.copy()
    necessary = {"feature_set", "model", *REPORT_METRICS}
    missing = necessary.difference(table.columns)
    if missing:
        raise ValueError(f"Missing comparison columns: {sorted(missing)}")
    orig = table.loc[table["feature_set"] == "original"].set_index("model")[REPORT_METRICS]
    eng = table.loc[table["feature_set"] == "engineered"].set_index("model")[REPORT_METRICS]
    both = orig.add_suffix("_original").join(
        eng.add_suffix("_engineered"), how="outer"
    )
    for metric in REPORT_METRICS:
        both[f"delta_{metric}"] = (
            both[f"{metric}_engineered"] - both[f"{metric}_original"]
        )
    return both.reset_index()


def _save_figure(fig, filename):
    if filename:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURES_DIR / filename, dpi=150, bbox_inches="tight")


def plot_confusion_matrix(model, X_test, y_test, title, filename=None):
    """Single-model confusion matrix (rows: true label, columns: prediction)."""
    disp = ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, labels=[0, 1],
        display_labels=["legitimate", "phishing"], cmap="Blues",
    )
    disp.ax_.set_title(title)
    _save_figure(disp.figure_, filename)
    return disp


def plot_confusion_matrices(models, X_test, y_test, title, filename=None):
    """Compare the confusion matrices for a dictionary of fitted models.

    All models must have been fitted on the feature columns in X_test.
    """
    if not models:
        raise ValueError("Pass at least one fitted model")
    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4))
    axes = [axes] if len(models) == 1 else axes
    for ax, (name, model) in zip(axes, models.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, model.predict(X_test), labels=[0, 1],
            display_labels=["legitimate", "phishing"],
            cmap="Blues", colorbar=False, ax=ax,
        )
        ax.set_title(name)
    fig.suptitle(title)
    fig.tight_layout()
    _save_figure(fig, filename)
    return fig


def plot_roc_curves(models, X_test, y_test, title, filename=None):
    """One ROC plot for all fitted models on the SAME held-out test rows."""
    if not models:
        raise ValueError("Pass at least one fitted model")
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in models.items():
        scores = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, scores, pos_label=1)
        auc = roc_auc_score(y_test, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_figure(fig, filename)
    return fig


def plot_feature_comparison(results, metric="f1", filename=None):
    """Grouped bar chart of original vs engineered results for each model."""
    if metric not in REPORT_METRICS:
        raise ValueError(f"metric must be one of {REPORT_METRICS}")
    table = compare(results) if isinstance(results, dict) else results.copy()
    pivot = table.pivot(index="model", columns="feature_set", values=metric)
    for needed in ("original", "engineered"):
        if needed not in pivot.columns:
            raise ValueError(f"Missing feature set: {needed}")
    ax = pivot[["original", "engineered"]].plot.bar(figsize=(9, 5), rot=15)
    ax.set(ylabel=metric.upper(), xlabel="Model",
           title=f"Before vs after feature engineering: {metric.upper()}")
    ax.legend(title="Feature set")
    ax.figure.tight_layout()
    _save_figure(ax.figure, filename)
    return ax.figure
