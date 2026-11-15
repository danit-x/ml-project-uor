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
from sklearn.model_selection import StratifiedKFold, cross_validate

from . import config

CV_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def cv_splitter() -> StratifiedKFold:
    """The shared CV folds. Pass as cv= to GridSearchCV / RandomizedSearchCV."""
    return StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )


def cv_evaluate(unfitted_model, X_train, y_train) -> dict:
    """Cross-validated metrics on the training set (mean and std per metric).

    Use this while developing and tuning. The test set (evaluate()) is only
    for the final reported numbers.
    """
    scores = cross_validate(
        unfitted_model, X_train, y_train, cv=cv_splitter(), scoring=CV_METRICS
    )
    out = {}
    for m in CV_METRICS:
        out[f"{m}_mean"] = scores[f"test_{m}"].mean()
        out[f"{m}_std"] = scores[f"test_{m}"].std()
    return out


def evaluate(model, X_test, y_test) -> dict:
    """Return all planned metrics for one fitted model (final test-set report)."""
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
