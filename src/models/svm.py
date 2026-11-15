"""Support Vector Machine.  Owner: Dharmasiri S.O.R  (proposal section 4.9)

A kernel SVM is very slow on ~235,000 rows, so the baseline is a linear SVM.
It is wrapped in CalibratedClassifierCV so that predict_proba() exists, which
ROC-AUC and the risk score need.
"""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .. import config


def build():
    """Unfitted model. TODO (owner): tune C; optionally try an RBF kernel on a sample."""
    svm = LinearSVC(random_state=config.RANDOM_STATE)
    return make_pipeline(StandardScaler(), CalibratedClassifierCV(svm, cv=3))


def train(X, y):
    return build().fit(X, y)
