"""Random Forest.  Owner: Thanthrige D.P  (proposal section 4.9)"""
from sklearn.ensemble import RandomForestClassifier

from .. import config


def build():
    """Unfitted model. TODO (owner): tune hyperparameters (n_estimators, max_depth, ...)."""
    return RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )


def train(X, y):
    return build().fit(X, y)
