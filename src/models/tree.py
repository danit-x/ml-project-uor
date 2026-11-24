"""Decision Tree training and cross-validation (proposal section 4.9).

Owner: Madushan S.C.  In the processed target, 1 = phishing (positive class).

build() holds the configuration main.py evaluates. tune() is the training-only
grouped grid search that chooses it; copy its best_params_ into the BEST_*
constants so main.py stays fast and reproducible (CLAUDE.md "Tuning").
"""
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from .. import config, evaluate

# max_depth 20 was added on 2026-11-24 after the complexity review
# (tune_models.py part "dt": a max_depth curve 8..32 with leaves of 1); see the
# BEST_* comment below. The other values are the original grid.
PARAM_GRID = {
    "max_depth": [4, 8, 16, 20, None],
    "min_samples_leaf": [1, 10, 50],
    "class_weight": [None, "balanced"],
}

# Phishing (class 1) is the positive label in every threshold metric; F1 picks
# the winner because it balances missed phishing against false alarms.
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
    "recall": make_scorer(recall_score, pos_label=1, zero_division=0),
    "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
    "roc_auc": "roc_auc",
}
REFIT_METRIC = "f1"

# Configuration evaluated by main.py (complexity review of 2026-11-24, tune_models.py
# part "dt", reports/tuning/dt_complexity.csv). Full training split (188,297 rows of
# the grouped split), url_engineered set (23 features), grouped 5-fold CV, phishing
# F1. No test rows used. Fold-tree sizes are means over the 5 fitted fold trees.
#   max_depth=None (first grid winner): F1 0.99008 (std 0.00030), depth 37, 1,564 leaves
#   max_depth=20 (depth curve):         F1 0.99053 (std 0.00042), depth 20, 1,299 leaves
#   max_depth=24:                       F1 0.99034 (std 0.00029), depth 24, 1,459 leaves
#   max_depth=16 / 8 / 4:               F1 0.98720 / 0.87218 / 0.81524
# Rule fixed before the run: CART one-standard-error rule (fewest leaves with mean
# F1 >= best - SE). It picks max_depth=20, which is also the highest mean: the
# unlimited tree's extra 17 levels add 265 leaves and no CV gain. The difference
# to max_depth=None (0.00045) is small (about 1.5 std), so the honest summary is
# "as good and simpler". Depth 20 was found by a depth curve outside the first
# grid; choosing the best of 8 extra configurations adds a little optimism to its
# CV score, which is one more reason the held-out test is reported separately.
# A tree small enough to read (depth 4: 15 leaves) loses about 0.17 F1, so for the
# demonstration the tree is explained by its top levels and feature importances,
# not by reading all 1,299 leaves.
BEST_MAX_DEPTH = 20
BEST_MIN_SAMPLES_LEAF = 1
BEST_CLASS_WEIGHT = None
TUNING_NOTE = (
    "Grouped 5-fold CV on the full training split, url_engineered features, "
    "complexity review + one-SE rule: max_depth=20, phishing F1 = 0.99053 (std 0.00042)."
)


def build():
    """Unfitted tree with the frozen configuration (BEST_* constants)."""
    return DecisionTreeClassifier(
        max_depth=BEST_MAX_DEPTH,
        min_samples_leaf=BEST_MIN_SAMPLES_LEAF,
        class_weight=BEST_CLASS_WEIGHT,
        random_state=config.RANDOM_STATE,
    )


def tune(X, y, groups=None, n_jobs=-1):
    """Grid-search PARAM_GRID with grouped CV on TRAINING rows only.

    Inputs: training features, target (1 = phishing) and the URL group ids
    of the same rows (data.group_ids(df).loc[X_train.index]). The folds come
    from evaluate.cv_folds(), so no near-duplicate URL group is split between
    a fitting and a validation fold, and the folds equal those used for the
    other models' CV on the same rows. All SCORING metrics are recorded; the
    best candidate is chosen by phishing F1 (REFIT_METRIC).
    Output: the fitted GridSearchCV (.best_params_, .cv_results_).
    groups=None falls back to ungrouped folds with an UngroupedCVWarning.
    """
    search = GridSearchCV(
        build(),
        PARAM_GRID,
        scoring=SCORING,
        refit=REFIT_METRIC,
        cv=evaluate.cv_folds(X, y, groups),
        n_jobs=n_jobs,
        error_score="raise",
    )
    return search.fit(X, y)


def is_frozen_config(params) -> bool:
    """True if params (e.g. search.best_params_) equal the build() configuration."""
    current = build().get_params()
    return all(current[key] == value for key, value in params.items())


def train(X, y):
    """Fit the frozen configuration (build()) and return the fitted tree."""
    return build().fit(X, y)
