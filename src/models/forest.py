"""Random Forest.  Owner: Thanthrige D.P  (proposal section 4.9)

An ensemble of decision trees, each trained on a bootstrap sample of the rows
and a random subset of features at every split. Averaging many decorrelated
trees lowers the variance of a single tree, and the averaged leaf frequencies
give the phishing probability used as the risk score (section 4.11).

Hyperparameters in build() were chosen with tune() (RandomizedSearchCV on the
training set only, shared CV folds); see notebook 04 for the search results.
"""
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from .. import config

# Search space for tune(). Kept small on purpose (about 20 candidates).
PARAM_DISTRIBUTIONS = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", 0.3, 0.5],
    "class_weight": [None, "balanced"],
}


def build():
    """Unfitted model with the tuned hyperparameters.

    Trees need no scaling, so no Pipeline is required (nothing is fitted
    before the model).

    Chosen by tune() on the 20,000-row sample (original features, CV F1).
    Six candidates tied for best CV F1 (0.99993, std 0.00015); the next group
    was one URL worse over all folds, i.e. within noise. Among the tied ones
    this is the cheapest (100 trees, ~0.5 s per fit) and the most regularised
    (min_samples_leaf=8 gives smoother probabilities for the risk score).
    Provisional: revisit after the leakage check (phase 2).
    """
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=30,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight=None,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )


def train(X, y):
    """Fit build() on (X, y) and return the fitted model."""
    return build().fit(X, y)


def tune(X_train, y_train, n_iter=20, scoring="f1"):
    """Randomized hyperparameter search on the training set only.

    Inputs: training features and target (1 = phishing), number of random
    candidates, and the metric used to choose the best one.
    Output: (fitted RandomizedSearchCV, seconds taken). Every candidate is
    scored on all shared CV metrics; `scoring` decides which one picks the winner.

    Why randomized and not grid: 4*4*4*4*2 = 512 combinations would take hours;
    20 random candidates cover the space well when only a few settings matter.
    Uses evaluate.cv_splitter() so the folds match every other member's.
    """
    from ..evaluate import CV_METRICS, cv_splitter

    search = RandomizedSearchCV(
        RandomForestClassifier(n_jobs=-1, random_state=config.RANDOM_STATE),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring=CV_METRICS,
        refit=scoring,
        cv=cv_splitter(),
        random_state=config.RANDOM_STATE,
        n_jobs=1,  # the forest itself already uses all cores
    )
    start = time.perf_counter()
    search.fit(X_train, y_train)
    return search, time.perf_counter() - start


def search_table(search, top=10):
    """Top candidates of a tune() result as a small rounded table."""
    import pandas as pd

    res = pd.DataFrame(search.cv_results_)
    cols = ["params"] + [f"mean_test_{m}" for m in ("f1", "roc_auc", "recall", "precision")]
    cols += ["std_test_f1", "mean_fit_time"]
    return res.sort_values(f"rank_test_{search.refit}")[cols].head(top).round(4)
