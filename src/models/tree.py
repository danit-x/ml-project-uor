"""Decision Tree.  Owner: Madushan S.C  (proposal section 4.9)"""
from sklearn.tree import DecisionTreeClassifier

from .. import config


def build():
    """Unfitted model. TODO (owner): tune hyperparameters (max_depth, min_samples_leaf, ...)."""
    return DecisionTreeClassifier(random_state=config.RANDOM_STATE)


def train(X, y):
    return build().fit(X, y)
