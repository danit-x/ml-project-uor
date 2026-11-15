"""Logistic Regression.  Owner: Madushanka D.M.C  (proposal section 4.9)"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import config


def build():
    """Unfitted model. TODO (owner): tune hyperparameters (C, penalty, ...)."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
    )


def train(X, y):
    return build().fit(X, y)
