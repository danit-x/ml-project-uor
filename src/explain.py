"""Explainability part: which features drive the predictions.

Owner: Thanthrige D.P  (proposal section 4.13)

Tree models (Decision Tree, Random Forest) are explained with SHAP
TreeExplainer: exact Shapley values, in units of phishing probability.
Linear models (Logistic Regression, linear SVM) are explained by their own
coefficients: in a standardised linear model, coefficient x standardised value
is that feature's contribution to the decision score (log-odds for LR), which
is what shap.LinearExplainer computes for independent features. No extra
library or background sample is needed for them.

SHAP is imported inside the functions so the rest of the project runs even
if shap is not installed.
"""
import numpy as np
import pandas as pd

from . import config


def _phishing_values(shap_values):
    """SHAP values for the phishing class, whatever shape the shap version returns."""
    if isinstance(shap_values, list):  # older shap: one array per class
        return shap_values[1]
    if getattr(shap_values, "ndim", 2) == 3:  # newer shap: (rows, features, classes)
        return shap_values[:, :, 1]
    return shap_values


def shap_values_tree(model, X_sample: pd.DataFrame):
    """SHAP values for a tree model (Decision Tree / Random Forest).

    Pass a sample (e.g. 500-2,000 rows), not the full test set, to keep it fast
    (about 30 s per 1,000 rows for the tuned 500-tree forest).
    """
    import shap

    explainer = shap.TreeExplainer(model)
    return _phishing_values(explainer.shap_values(X_sample))


def global_importance(model, X_sample: pd.DataFrame) -> pd.DataFrame:
    """Features ranked by mean absolute SHAP value."""
    values = shap_values_tree(model, X_sample)
    return rank_importance(X_sample.columns, values, "mean_abs_shap")


def summary_plot(model, X_sample: pd.DataFrame, filename="shap_summary.png", values=None):
    """Save the SHAP summary (beeswarm) plot to reports/figures.

    Pass precomputed `values` (from shap_values_tree) to avoid computing them twice.
    """
    import matplotlib.pyplot as plt
    import shap

    if values is None:
        values = shap_values_tree(model, X_sample)
    shap.summary_plot(values, X_sample, show=False, max_display=15)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def rank_importance(columns, values, name) -> pd.DataFrame:
    importance = pd.DataFrame({"feature": list(columns), name: np.abs(values).mean(axis=0)})
    return importance.sort_values(name, ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- linear models

def _linear_parts(pipeline):
    """(scaler, coefficients) of a fitted scaler + linear-model pipeline.

    Handles LogisticRegression (coef_) and a CalibratedClassifierCV around a
    LinearSVC (coefficients averaged over its CV fold models, as in svm.py).
    """
    scaler, final = pipeline[:-1], pipeline[-1]
    if hasattr(final, "coef_"):
        coef = final.coef_
    elif hasattr(final, "calibrated_classifiers_"):
        coef = np.mean([c.estimator.coef_ for c in final.calibrated_classifiers_], axis=0)
    else:
        raise TypeError(f"{type(final).__name__} has no linear coefficients")
    return scaler, np.ravel(coef)


def linear_contributions(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Per-row, per-feature contribution to the decision score of a linear pipeline.

    contribution = coefficient x standardised value. Positive pushes towards
    phishing. Rows and columns match X.
    """
    scaler, coef = _linear_parts(pipeline)
    z = scaler.transform(X)
    return pd.DataFrame(z * coef, index=X.index, columns=X.columns)


def linear_importance(pipeline, X_sample: pd.DataFrame) -> pd.DataFrame:
    """Features ranked by mean absolute contribution (the linear-model counterpart of global_importance)."""
    contrib = linear_contributions(pipeline, X_sample)
    table = rank_importance(X_sample.columns, contrib.values, "mean_abs_contribution")
    _, coef = _linear_parts(pipeline)
    table["coefficient"] = table["feature"].map(dict(zip(X_sample.columns, coef)))
    return table


# ---------------------------------------------------------------- single URLs

def contributions(model, X: pd.DataFrame) -> pd.DataFrame:
    """Per-feature contributions for any of the project's models (tree -> SHAP, linear -> coef x z)."""
    if hasattr(model, "estimators_") or hasattr(model, "tree_"):
        return pd.DataFrame(shap_values_tree(model, X), index=X.index, columns=X.columns)
    return linear_contributions(model, X)


def explain_predictions(model, X: pd.DataFrame, scores, top=3, contrib=None) -> pd.DataFrame:
    """Risk score + risk level + predicted class + top contributing features, one row per URL.

    Inputs: the fitted model that is explained, the rows X, the risk scores for
    those rows (risk.risk_score of the model that serves the score, e.g. the
    calibrated forest), how many features to list, and optionally precomputed
    contributions (contributions(model, X)).
    The calibrated score is explained by the uncalibrated forest's SHAP values:
    sigmoid calibration is monotonic, so it changes the scale of the score but
    not which features push it up or down.
    """
    from . import risk

    if contrib is None:
        contrib = contributions(model, X)
    scores = np.asarray(scores)
    out = pd.DataFrame(index=X.index)
    out["risk_score"] = scores
    out["risk_level"] = np.asarray(risk.risk_levels(scores))
    out["predicted"] = np.where(scores >= 0.5, "phishing", "legitimate")
    order = np.argsort(-np.abs(contrib.values), axis=1)[:, :top]
    cols = np.asarray(X.columns)
    for k in range(top):
        idx = order[:, k]
        rows = np.arange(len(X))
        out[f"feature_{k + 1}"] = cols[idx]
        out[f"value_{k + 1}"] = X.values[rows, idx]
        out[f"contribution_{k + 1}"] = contrib.values[rows, idx]
    return out


def explain_one(model, x_row: pd.DataFrame, score: float, top=5) -> pd.DataFrame:
    """Readable explanation of one URL: its top features, values and contributions.

    x_row is a one-row DataFrame. Positive contribution = pushes towards phishing.
    """
    from . import risk

    contrib = contributions(model, x_row).iloc[0]
    order = contrib.abs().sort_values(ascending=False).index[:top]
    table = pd.DataFrame({
        "feature": order,
        "value": x_row.iloc[0][order].values,
        "contribution": contrib[order].values,
        "pushes_towards": np.where(contrib[order].values > 0, "phishing", "legitimate"),
    })
    table.attrs["risk_score"] = float(score)
    table.attrs["risk_level"] = risk.risk_level(float(score))
    return table
