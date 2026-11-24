"""Common evaluation + cross-model comparison for URLSentra (4.10, 4.12).

Owner: Dharmasiri S.O.R. In the normalized target, 1 means phishing.
Training-CV methods must not be passed held-out test rows during development.
"""
import warnings
from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate

from . import config, data

CV_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


class UngroupedCVWarning(UserWarning):
    """Training CV ran without URL groups (legacy protocol, not the approved one)."""


def cv_splitter(grouped: bool = False):
    """Shared seeded CV splitter.

    grouped=True (the approved protocol): StratifiedGroupKFold with
    config.CV_FOLDS folds; its split() needs the URL group ids.
    grouped=False: the earlier StratifiedKFold, kept for the historical
    notebook sections, cv_folds(groups=None) and the inner calibration folds
    of risk.calibration_report() (ungrouped; disclosed as a limitation).
    Prefer cv_folds(), which also caches.
    """
    cls = StratifiedGroupKFold if grouped else StratifiedKFold
    return cls(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


_FOLD_CACHE = {}


def _aligned_groups(X, groups) -> np.ndarray:
    """Group ids in X's row order. A Series is aligned on X.index."""
    if isinstance(groups, pd.Series) and isinstance(X, (pd.DataFrame, pd.Series)):
        aligned = groups.reindex(X.index)
        if aligned.isna().any():
            raise ValueError("groups has no id for some rows of X")
        return aligned.to_numpy()
    groups = np.asarray(groups)
    if len(groups) != len(X):
        raise ValueError(f"groups has {len(groups)} entries for {len(X)} rows")
    return groups


def cv_folds(X, y, groups=None) -> list:
    """Training CV folds as a list of (train_positions, validation_positions).

    Input: training rows only, their target, and their URL group ids
    (data.group_ids(df).loc[X_train.index]). Output: config.CV_FOLDS folds
    from StratifiedGroupKFold, so no URL group is in both parts of a fold.
    The same (rows, y, groups) always give the same folds, and the list is
    cached, so every model compared on these rows sees identical folds.
    Pass the list as cv= to GridSearchCV, cross_validate or
    CalibratedClassifierCV; the groups are already applied.

    groups=None runs the legacy ungrouped StratifiedKFold and emits an
    UngroupedCVWarning: near-duplicate URLs can then cross folds, so such
    results are not valid under the approved protocol.
    """
    if groups is None:
        warnings.warn(
            "Training CV without URL groups: near-duplicate URLs can cross folds. "
            "Pass groups=data.group_ids(df).loc[X_train.index].",
            UngroupedCVWarning, stacklevel=2,
        )
        return list(cv_splitter(grouped=False).split(np.zeros(len(y)), np.asarray(y)))
    group_arr = _aligned_groups(X, groups)
    y_arr = np.asarray(y)
    key = data.fingerprint(pd.Series(y_arr), pd.Series(group_arr))
    if key not in _FOLD_CACHE:
        if len(_FOLD_CACHE) >= 4:
            _FOLD_CACHE.pop(next(iter(_FOLD_CACHE)))
        splits = cv_splitter(grouped=True).split(np.zeros(len(y_arr)), y_arr, group_arr)
        _FOLD_CACHE[key] = [(train, valid) for train, valid in splits]
    return _FOLD_CACHE[key]


def cv_evaluate(unfitted_model, X_train, y_train, groups=None, n_jobs=None) -> dict:
    """Cross-validated training-only metrics (mean and standard deviation).

    groups: URL group ids of the training rows (approved protocol). Without
    them the legacy ungrouped folds are used, with an UngroupedCVWarning.
    n_jobs: folds fitted in parallel (None = one after another). Leave None
    for models that already use every core (Random Forest).
    """
    scores = cross_validate(
        clone(unfitted_model), X_train, y_train,
        cv=cv_folds(X_train, y_train, groups), scoring=CV_METRICS,
        error_score="raise", n_jobs=n_jobs,
    )
    out = {}
    for metric in CV_METRICS:
        out[f"{metric}_mean"] = float(scores[f"test_{metric}"].mean())
        out[f"{metric}_std"] = float(scores[f"test_{metric}"].std())
    return out


def compare_training_models(feature_sets: Mapping, models: Mapping, groups=None,
                            n_jobs: Mapping = None) -> pd.DataFrame:
    """Run identical CV for unfitted models on TRAINING data only.

    feature_sets = {'url_base': (X_train_b, y_train),
                    'url_engineered': (X_train_e, y_train), ...}.
    models = {'Logistic Regression': logreg.build(), 'Random Forest': forest.build(), ...}
    (main.py wraps each in features.FeatureSetSelector so selection is refitted per fold).
    groups = URL group ids of those training rows (approved grouped CV);
    every feature set must hold the same rows, so all get identical folds.

    Caller must supply only training rows, e.g. those returned as X_train and
    y_train by data.split(). Model selection must use this CV table, NOT the
    held-out test set. Returns model, feature_set and mean/std for each metric.
    n_jobs = optional {model_name: n_jobs} passed to cv_evaluate (default None).
    """
    if not feature_sets or not models:
        raise ValueError("Provide at least one feature set and one model")
    n_jobs = n_jobs or {}
    rows = []
    for feature_set, pair in feature_sets.items():
        X_train, y_train = pair
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must contain the same row count")
        for model_name, unfitted_model in models.items():
            result = cv_evaluate(unfitted_model, X_train, y_train, groups=groups,
                                 n_jobs=n_jobs.get(model_name))
            rows.append({"feature_set": feature_set, "model": model_name, **result})
    return pd.DataFrame(rows)


def evaluate(model, X_test, y_test) -> dict:
    """Final held-out test metrics; call only after model selection is frozen."""
    return prediction_metrics(y_test, model.predict(X_test), model.predict_proba(X_test)[:, 1])


def prediction_metrics(y_test, y_pred, y_score) -> dict:
    """evaluate()'s metrics from predictions (1 = phishing) and phishing scores.

    Used by evaluate() and for fixed rules that are not sklearn models (the
    URL-format reference baseline in main.py), so both report the same metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def rule_cv_metrics(y_train, y_pred, folds) -> dict:
    """CV-style mean/std of a FIXED rule's metrics over the validation folds.

    The rule learns nothing, so there is nothing to fit per fold; scoring it
    on the same validation folds as the models makes its numbers directly
    comparable with cv_evaluate() output (same keys).
    """
    y_true, y_hat = np.asarray(y_train), np.asarray(y_pred)
    per_fold = [prediction_metrics(y_true[v], y_hat[v], y_hat[v]) for _, v in folds]
    out = {}
    for metric in CV_METRICS:
        values = np.array([fold[metric] for fold in per_fold])
        out[f"{metric}_mean"] = float(values.mean())
        out[f"{metric}_std"] = float(values.std())
    return out


def compare(results: dict) -> pd.DataFrame:
    """Keep main.py's original interface for final metric comparison."""
    table = pd.DataFrame(results).T
    table.index.names = ["feature_set", "model"]
    return table.reset_index()


def feature_delta_table(results, base="original", other="engineered") -> pd.DataFrame:
    """`other` minus `base` feature set: CV data (mean columns) or final-score data.

    Defaults keep notebook 03's original/engineered naming; main.py passes
    base="url_base", other="url_engineered".
    """
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
    orig = frame.loc[frame.feature_set == base].set_index("model")[metrics]
    eng = frame.loc[frame.feature_set == other].set_index("model")[metrics]
    merged = orig.add_suffix(f"_{base}").join(
        eng.add_suffix(f"_{other}"), how="outer"
    )
    for metric in metrics:
        merged[f"delta_{metric}"] = (
            merged[f"{metric}_{other}"] - merged[f"{metric}_{base}"]
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


def plot_feature_comparison(results, metric="f1", filename=None,
                            base="original", other="engineered"):
    """`base` vs `other` feature set bar chart for a selected metric.

    For training CV results use '<metric>_mean'; for final-score data use
    '<metric>'. Caller is responsible for supplying the correct results.
    Defaults keep notebook 03's original naming; pass base="url_base",
    other="url_engineered" for the agreed feature sets.
    """
    if metric not in CV_METRICS:
        raise ValueError(f"metric must be one of {CV_METRICS}")
    frame = compare(results) if isinstance(results, dict) else results.copy()
    score_col = f"{metric}_mean" if f"{metric}_mean" in frame.columns else metric
    if score_col not in frame.columns:
        raise ValueError(f"Missing comparison metric: {metric}")
    pivot = frame.pivot(index="model", columns="feature_set", values=score_col)
    if not {base, other}.issubset(pivot.columns):
        raise ValueError(f"Need both {base} and {other} feature sets")
    ax = pivot[[base, other]].plot.bar(figsize=(8.5, 5), rot=15)
    ax.set(ylabel=f"{metric.upper()} score", xlabel="Model",
           title=f"{base} vs {other} features ({metric.upper()})")
    ax.legend(title="Feature set")
    ax.figure.tight_layout()
    _save_figure(ax.figure, filename)
    return ax.figure
