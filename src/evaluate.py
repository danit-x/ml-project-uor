"""Evaluation part: common metrics, model comparison, before/after comparison.

Owner: Dharmasiri S.O.R  (proposal sections 4.10, 4.12)

Everyone evaluates their model with evaluate() so results are comparable.
Positive class = phishing (1).
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
)

from . import config


def evaluate(model, X_test, y_test) -> dict:
    """Return all planned metrics for one fitted model."""
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compare(results: dict) -> pd.DataFrame:
    """Turn {(feature_set, model_name): metrics} into one comparison table."""
    table = pd.DataFrame(results).T
    table.index.names = ["feature_set", "model"]
    return table.reset_index()


def plot_confusion_matrix(model, X_test, y_test, title, filename=None):
    """Draw a confusion matrix; save it to reports/figures if filename is given."""
    disp = ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["legitimate", "phishing"]
    )
    disp.ax_.set_title(title)
    if filename:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        disp.figure_.savefig(config.FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    return disp


# TODO (owner):
# - ROC curves of the four models on one plot
# - before vs after feature engineering table / chart from compare()
# - short written interpretation of the differences
