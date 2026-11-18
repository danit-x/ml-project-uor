"""Logistic Regression.  Owner: Madushanka D.M.C  (proposal section 4.9)

build() returns the pipeline with the hyperparameters chosen by tune() on the
training split (5-fold CV from evaluate.cv_splitter(); see notebook 01). They
are written in as fixed values so main.py stays fast and reproducible.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import config, evaluate

# liblinear supports both L1 (l1_ratio=1) and L2 (l1_ratio=0) for a binary target,
# so one solver covers the whole grid and the comparison is like for like.
SOLVER = "liblinear"
MAX_ITER = 1000

# Search space for tune(). C = inverse regularisation strength (small C = simpler model);
# l1_ratio 0 = L2 (shrinks all weights), 1 = L1 (can set weights to exactly 0).
# L1 stops at C = 1: on the full training split the classes are almost linearly
# separable, and a single liblinear L1 fit with C = 10 did not finish in 30 minutes
# (C <= 1 fits take about 2 s). Weak L1 regularisation is also the least useful part
# of the grid, because the point of L1 is to force weights to zero.
PARAM_GRID = [
    {
        "logisticregression__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "logisticregression__l1_ratio": [0.0],
    },
    {
        "logisticregression__C": [0.001, 0.01, 0.1, 1.0],
        "logisticregression__l1_ratio": [1.0],
    },
]
# Metric that picks the best candidate. F1 on the phishing class balances
# missed phishing (recall) against false alarms (precision); all of
# evaluate.CV_METRICS are still recorded for every candidate.
REFIT_METRIC = "f1"

# Chosen by tune() on the full training split, original features (see notebook 01
# for the CV table). C = 10 and C = 100 tied on mean CV F1; GridSearchCV keeps the
# first tied candidate, which is also the stronger regularisation (C = 10).
# All candidates score above 0.998, so the differences between them are tiny.
BEST_C = 10.0
BEST_L1_RATIO = 0.0


def _pipeline(C: float, l1_ratio: float):
    """Scaler + LogisticRegression; the scaler is fitted inside CV folds only."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            l1_ratio=l1_ratio,
            solver=SOLVER,
            max_iter=MAX_ITER,
            random_state=config.RANDOM_STATE,
        ),
    )


def build():
    """Unfitted model with the tuned hyperparameters (BEST_C, BEST_L1_RATIO)."""
    return _pipeline(C=BEST_C, l1_ratio=BEST_L1_RATIO)


def train(X, y):
    return build().fit(X, y)


def tune(X_train, y_train, n_jobs=-1) -> GridSearchCV:
    """Grid-search C and the penalty type with the shared CV folds.

    Inputs: training features and target only (never the test set).
    Output: the fitted GridSearchCV; use .best_params_ and .cv_results_.
    Copy the best values into BEST_C / BEST_L1_RATIO by hand so build() is fixed.
    """
    search = GridSearchCV(
        build(),
        PARAM_GRID,
        scoring=evaluate.CV_METRICS,
        refit=REFIT_METRIC,
        cv=evaluate.cv_splitter(),
        n_jobs=n_jobs,
    )
    return search.fit(X_train, y_train)
