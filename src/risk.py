"""Risk scoring part: phishing probability as a risk score, calibration.

Owner: Thanthrige D.P  (proposal section 4.11)

The risk score is the model's estimated probability that a URL is phishing
(predict_proba[:, 1], 1 = phishing). For that number to be read as a risk
it should be calibrated: of all URLs scored about 0.8, about 80 % should be
phishing. calibration_report() measures that on the training set with
out-of-fold predictions, so the test set is never used to pick a method.
"""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import cross_val_predict

from . import config

# Risk bands on the 0-1 score. Low < LOW_MAX <= Medium < HIGH_MIN <= High.
# Justified in notebook 04 phase 4 with band_table() on out-of-fold training
# scores: the observed phishing rate in each band has to match its name
# (Low mostly legitimate, High mostly phishing, Medium genuinely uncertain).
LOW_MAX = 0.30
HIGH_MIN = 0.70
LEVELS = ["Low", "Medium", "High"]

CALIBRATION_METHODS = ["none", "sigmoid", "isotonic"]

# Chosen with calibration_report() on the Random Forest (notebook 04 phase 4,
# 20k sample, out-of-fold on training rows): sigmoid gave the lowest Brier score
# and log loss with ROC-AUC unchanged. The uncalibrated forest is underconfident
# (many correct URLs scored between 0.05 and 0.95), which sigmoid fixes.
CALIBRATION = "sigmoid"


def risk_score(model, X) -> np.ndarray:
    """Estimated probability (0-1) that each URL is phishing."""
    return model.predict_proba(X)[:, 1]


def risk_level(score: float) -> str:
    """Band name for one score."""
    if score < LOW_MAX:
        return "Low"
    if score < HIGH_MIN:
        return "Medium"
    return "High"


def risk_levels(scores) -> pd.Categorical:
    """Band names for an array of scores (vectorised risk_level)."""
    bins = [-np.inf, LOW_MAX, HIGH_MIN, np.inf]
    return pd.cut(np.asarray(scores), bins=bins, labels=LEVELS, right=False)


def calibrate(unfitted_model, X_train, y_train, method=CALIBRATION):
    """Fit a probability-calibrated version of a model on training data only.

    Uses the shared CV folds: each fold's model is calibrated on the rows it
    did not see, and the fold models are averaged. Never pass the test set.
    """
    from .evaluate import cv_splitter

    calibrated = CalibratedClassifierCV(unfitted_model, method=method, cv=cv_splitter())
    return calibrated.fit(X_train, y_train)


def _maybe_calibrated(unfitted_model, method):
    """The model itself for method 'none', else an unfitted calibrated wrapper."""
    if method == "none":
        return unfitted_model
    from .evaluate import cv_splitter

    return CalibratedClassifierCV(unfitted_model, method=method, cv=cv_splitter())


def oof_scores(unfitted_model, X_train, y_train) -> np.ndarray:
    """Out-of-fold phishing probabilities on the training set (shared CV folds).

    Every row is scored by a model that did not see it, so these scores show
    how the risk score behaves on new URLs without touching the test set.
    """
    from .evaluate import cv_splitter

    proba = cross_val_predict(
        unfitted_model, X_train, y_train, cv=cv_splitter(), method="predict_proba"
    )
    return proba[:, 1]


def score_metrics(y, scores) -> dict:
    """Probability-quality metrics for one set of scores.

    Brier score: mean squared error of the probability (0 is perfect, lower
    is better). Log loss punishes confident mistakes hardest. ROC-AUC only
    checks the ranking, so calibration should leave it almost unchanged.
    """
    clipped = np.clip(scores, 1e-15, 1 - 1e-15)
    return {
        "brier": brier_score_loss(y, scores),
        "log_loss": log_loss(y, clipped, labels=[0, 1]),
        "roc_auc": roc_auc_score(y, scores),
    }


def calibration_report(unfitted_model, X_train, y_train, methods=CALIBRATION_METHODS):
    """Compare no calibration, sigmoid (Platt) and isotonic on training data.

    Inputs: an unfitted model (e.g. forest.build()), training features/target.
    Output: (table with one row per method, {method: out-of-fold scores}).
    Each method is scored out-of-fold with the shared CV folds; a calibrated
    model calibrates inside each training fold, so this is nested and the
    held-out fold is never used to fit the calibration.
    Keep calibration only if it lowers Brier score / log loss.
    """
    scores, rows = {}, {}
    for m in methods:
        scores[m] = oof_scores(_maybe_calibrated(unfitted_model, m), X_train, y_train)
        rows[m] = score_metrics(y_train, scores[m])
    return pd.DataFrame(rows).T, scores


def plot_calibration(y, scores: dict, filename=None, n_bins=10, title="Calibration curve"):
    """Reliability diagram: mean predicted risk vs observed phishing rate per bin.

    A perfectly calibrated score lies on the diagonal. Bins are quantile-based
    so every point has enough URLs behind it (scores pile up near 0 and 1).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="perfectly calibrated")
    for name, s in scores.items():
        frac_pos, mean_pred = calibration_curve(y, s, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", ms=4, label=f"{name} (Brier {brier_score_loss(y, s):.4f})")
    ax.set_xlabel("mean predicted risk score")
    ax.set_ylabel("observed share of phishing URLs")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    if filename:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    return fig


def band_table(y, scores) -> pd.DataFrame:
    """How URLs fall into the Low / Medium / High bands, and how often each band is phishing.

    Used to justify LOW_MAX / HIGH_MIN: a band is only useful if its observed
    phishing rate matches its name.
    """
    levels = risk_levels(scores)
    y = np.asarray(y)
    rows = []
    for lvl in LEVELS:
        mask = np.asarray(levels == lvl)
        rows.append({
            "band": lvl,
            "score_range": {"Low": f"[0, {LOW_MAX})", "Medium": f"[{LOW_MAX}, {HIGH_MIN})",
                            "High": f"[{HIGH_MIN}, 1]"}[lvl],
            "n_urls": int(mask.sum()),
            "share_of_urls": mask.mean(),
            "phishing_rate": y[mask].mean() if mask.any() else np.nan,
            "n_phishing": int(y[mask].sum()),
            "n_legit": int((1 - y[mask]).sum()),
        })
    return pd.DataFrame(rows)
