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


# Bounded grouped search used for the progress evaluation (tune_models.py part
# "rf_subset"). A full-data RandomizedSearchCV (100 forest fits on 150k rows) was
# judged too expensive, so: a reproducible group-preserving subset of the training
# rows (data.group_subset, SUBSET_ROWS), the same grouped 5-fold CV and url_engineered
# features, and 2 x 2 x 2 = 8 candidates on the settings that change a forest's
# capacity. n_estimators is fixed, not tuned (more trees only lower variance and cost
# time); class_weight is left at None (classes are about 43 / 57 %, not imbalanced).
SUBSET_ROWS = 50_000
SUBSET_N_ESTIMATORS = 300
SUBSET_PARAM_GRID = {
    "max_depth": [None, 20],
    "min_samples_leaf": [1, 4],
    "max_features": ["sqrt", 0.5],
}

# Frozen 2026-11-24 from tune_models.py part "rf_subset" (reports/tuning/rf_subset_search.csv):
# 50,000 training rows in 49,229 whole URL groups (phishing share 0.4270 vs 0.4271 on the
# full training split), url_engineered (23 features), grouped 5-fold CV on the subset,
# phishing F1, no test rows; 8 candidates, 40 forest fits, 4.9 min. Mean CV F1 (std):
#   depth 20,   leaf 1, max_features 0.5:  0.99119 (0.00181), 660 leaves/tree  <- best and chosen
#   depth None, leaf 1, max_features 0.5:  0.99096 (0.00149), 714 leaves/tree  (within 1 SE)
#   depth None, leaf 1, sqrt:              0.99026 (0.00165), 1,214 leaves/tree
#   leaf 4 (any): 0.9825 - 0.9894 (larger leaves lose about 0.002 - 0.009 F1)
# Rule fixed before the run: fewest mean leaves per tree among candidates with mean
# F1 >= best - SE(best) (SE 0.00090). These are SUBSET CV numbers (tuning evidence);
# the forest is trained on the full training split by main.py, whose grouped CV on the
# full split is in reports/cv_results.csv. Superseded: 500 trees, depth 20, sqrt (20k
# sample, ungrouped CV, all original features incl. page content).
BEST_PARAMS = {
    "n_estimators": SUBSET_N_ESTIMATORS,
    "max_depth": 20,
    "min_samples_leaf": 1,
    "max_features": 0.5,
    "class_weight": None,
}
TUNING_NOTE = (
    "Bounded grouped 5-fold CV on a 50,000-row group-preserving training subset, url_engineered, "
    "one-SE rule on leaves per tree: depth 20, leaf 1, max_features 0.5, 300 trees, "
    "subset CV F1 = 0.99119 (std 0.00181)."
)


def build():
    """Unfitted model with the tuned hyperparameters (BEST_PARAMS).

    Trees need no scaling, so no Pipeline is required (nothing is fitted
    before the model).
    """
    return RandomForestClassifier(
        **BEST_PARAMS,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )


def train(X, y):
    """Fit build() on (X, y) and return the fitted model."""
    return build().fit(X, y)


def tune(X_train, y_train, n_iter=20, scoring="f1", groups=None):
    """Randomized hyperparameter search on the training set only.

    Inputs: training features and target (1 = phishing), number of random
    candidates, and the metric used to choose the best one.
    Output: (fitted RandomizedSearchCV, seconds taken). Every candidate is
    scored on all shared CV metrics; `scoring` decides which one picks the winner.

    Why randomized and not grid: 4*4*4*4*2 = 512 combinations would take hours;
    20 random candidates cover the space well when only a few settings matter.
    Uses evaluate.cv_folds() so the folds match every other member's; pass
    the training rows' URL group ids as groups (approved grouped CV).
    """
    from ..evaluate import CV_METRICS, cv_folds

    search = RandomizedSearchCV(
        RandomForestClassifier(n_jobs=-1, random_state=config.RANDOM_STATE),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring=CV_METRICS,
        refit=scoring,
        cv=cv_folds(X_train, y_train, groups),
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
