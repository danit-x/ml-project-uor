"""Explainability part: which features drive the predictions.

Owner: Thanthrige D.P  (proposal section 4.13)

SHAP is imported inside the functions so the rest of the project runs even
if shap is not installed.
"""
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

    Pass a sample (e.g. 2,000 rows), not the full test set, to keep it fast.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    return _phishing_values(explainer.shap_values(X_sample))


def global_importance(model, X_sample: pd.DataFrame) -> pd.DataFrame:
    """Features ranked by mean absolute SHAP value."""
    values = shap_values_tree(model, X_sample)
    importance = pd.DataFrame(
        {"feature": X_sample.columns, "mean_abs_shap": abs(values).mean(axis=0)}
    )
    return importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def summary_plot(model, X_sample: pd.DataFrame, filename="shap_summary.png"):
    """Save the SHAP summary (beeswarm) plot to reports/figures."""
    import matplotlib.pyplot as plt
    import shap

    values = shap_values_tree(model, X_sample)
    shap.summary_plot(values, X_sample, show=False)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


# TODO (owner):
# - explanation of single predictions (one phishing URL, one legitimate URL)
# - present risk score + predicted class + top contributing features together
# - explanations for the non-tree models (LR coefficients, or shap.LinearExplainer)
