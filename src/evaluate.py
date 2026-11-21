"""Common evaluation + cross-model comparison for URLSentra (4.10, 4.12).

Owner: Dharmasiri S.O.R. In the normalized target, 1 means phishing.
Training-CV methods must not be passed held-out test rows during development.
"""
from collections.abc import Mapping

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
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


def cv_splitter() -> StratifiedKFold:
    """Shared stratified CV folds; reuse for all development experiments."""
    return StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )


def cv_evaluate(unfitted_model, X_train, y_train) -> dict:
    """Cross-validated training-only metrics (mean and standard deviation)."""
    scores = cross_validate(
        clone(unfitted_model), X_train, y_train,
        cv=cv_splitter(), scoring=CV_METRICS,
        error_score="raise",
    )
    out = {}
    for metric in CV_METRICS:
        out[f"{metric}_mean"] = float(scores[f"test_{metric}"].mean())
        out[f"{metric}_std"] = float(scores[f"test_{metric}"].std())
    return out


def compare_training_models(feature_sets: Mapping, models: Mapping) -> pd.DataFrame:
    """Run identical CV for unfitted models on TRAINING data only.

    feature_sets = {'original': (X_train_o, y_train),
                    'engineered': (X_train_e, y_train)}.
    models = {'SVM': svm.build(), 'Random Forest': forest.build(), ...}.

    Caller must supply only training rows, e.g. those returned as X_train and
    y_train by data.split(). Model selection must use this CV table, NOT the
    held-out test set. Returns model, feature_set and mean/std for each metric.
    """
    if not feature_sets or not models:
        raise ValueError("Provide at least one feature set and one model")
    rows = []
    for feature_set, pair in feature_sets.items():
        X_train, y_train = pair
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must contain the same row count")
        for model_name, unfitted_model in models.items():
            result = cv_evaluate(unfitted_model, X_train, y_train)
            rows.append({"feature_set": feature_set, "model": model_name, **result})
    return pd.DataFrame(rows)


def evaluate(model, X_test, y_test) -> dict:
    """Final held-out test metrics; call only after model selection is frozen."""
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
    """Keep main.py's original interface for final metric comparison."""
    table = pd.DataFrame(results).T
    table.index.names = ["feature_set", "model"]
    return table.reset_index()


def feature_delta_table(results) -> pd.DataFrame:
    """Engineered minus original: CV data (mean columns) or final-score data."""
    frame = compare(results) if isinstance(results, dict) else results.copy()
    if not {"model", "feature_set"}.issubset(frame.columns):
        raise ValueError("Expected model and feature_set columns")
    metrics = [
        metric for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]
        if metric in frame.columns
    ]
    if not metrics:
        metrics = [
            f"{metric}_mean" for metric in CV_METRICS
            if f"{metric}_mean" in frame.columns
        ]
    if not metrics:
        raise ValueError("No supported score columns found")
    orig = frame.loc[frame.feature_set == "original"].set_index("model")[metrics]
    eng = frame.loc[frame.feature_set == "engineered"].set_index("model")[metrics]
    merged = orig.add_suffix("_original").join(
        eng.add_suffix("_engineered"), how="outer"
    )
    for metric in metrics:
        merged[f"delta_{metric}"] = (
            merged[f"{metric}_engineered"] - merged[f"{metric}_original"]
        )
    return merged.reset_index()


def _save_figure(fig, filename):
    if filename:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURES_DIR / filename, dpi=150, bbox_inches="tight")


def plot_confusion_matrix(model, X_test, y_test, title, filename=None):
    """Compatible single-model confusion matrix (caller chooses validation/test)."""
    disp = ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, labels=[0, 1],
        display_labels=["legitimate", "phishing"],
    )
    disp.ax_.set_title(title)
    _save_figure(disp.figure_, filename)
    return disp


def plot_confusion_matrices(models, X_valid, y_valid, title, filename=None):
    """Multi-model confusion plot on a training-derived validation fold."""
    if not models:
        raise ValueError("Pass fitted models")
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4))
    if len(models) == 1:
        axes = [axes]
    for ax, (model_name, model) in zip(axes, models.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_valid, model.predict(X_valid), labels=[0, 1],
            display_labels=["legitimate", "phishing"],
            cmap="Blues", colorbar=False, ax=ax,
        )
        ax.set_title(model_name)
    fig.suptitle(title)
    fig.tight_layout()
    _save_figure(fig, filename)
    return fig


def plot_roc_curves(models, X_valid, y_valid, title, filename=None):
    """Plot ROC for fitted models on SAME validation rows (NOT ordinary test)."""
    if not models:
        raise ValueError("Pass fitted models")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for model_name, model in models.items():
        scores = model.predict_proba(X_valid)[:, 1]
        fpr, tpr, _ = roc_curve(y_valid, scores, pos_label=1)
        auc = roc_auc_score(y_valid, scores)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_figure(fig, filename)
    return fig


def plot_feature_comparison(results, metric="f1", filename=None):
    """Original vs engineered bar chart for a selected metric.

    For training CV results use '<metric>_mean'; for final-score data use
    '<metric>'. Caller is responsible for supplying the correct results.
    """
    if metric not in CV_METRICS:
        raise ValueError(f"metric must be one of {CV_METRICS}")
    frame = compare(results) if isinstance(results, dict) else results.copy()
    score_col = f"{metric}_mean" if f"{metric}_mean" in frame.columns else metric
    if score_col not in frame.columns:
        raise ValueError(f"Missing comparison metric: {metric}")
    pivot = frame.pivot(index="model", columns="feature_set", values=score_col)
    if not {"original", "engineered"}.issubset(pivot.columns):
        raise ValueError("Need both original and engineered feature sets")
    ax = pivot[["original", "engineered"]].plot.bar(figsize=(8.5, 5), rot=15)
    ax.set(ylabel=f"{metric.upper()} score", xlabel="Model",
           title=f"Original vs engineered features ({metric.upper()})")
    ax.legend(title="Feature set")
    ax.figure.tight_layout()
    _save_figure(ax.figure, filename)
    return ax.figure
