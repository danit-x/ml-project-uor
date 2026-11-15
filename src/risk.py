"""Risk scoring part: phishing probability as a risk score, calibration.

Owner: Thanthrige D.P  (proposal section 4.11)
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from . import config

# Risk bands on the 0-1 score. TODO (owner): justify or adjust the thresholds.
LOW_MAX = 0.30
HIGH_MIN = 0.70


def risk_score(model, X) -> np.ndarray:
    """Estimated probability (0-1) that each URL is phishing."""
    return model.predict_proba(X)[:, 1]


def risk_level(score: float) -> str:
    if score < LOW_MAX:
        return "Low"
    if score < HIGH_MIN:
        return "Medium"
    return "High"


def calibrate(unfitted_model, X_train, y_train, method="isotonic"):
    """Fit a probability-calibrated version of a model.

    TODO (owner): compare calibration curves / Brier score before and after,
    and only keep calibration where it actually helps.
    """
    calibrated = CalibratedClassifierCV(unfitted_model, method=method, cv=3)
    return calibrated.fit(X_train, y_train)
