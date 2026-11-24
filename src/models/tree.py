"""Decision Tree training and cross-validation (proposal section 4.9)."""
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from .. import config, evaluate

PARAM_GRID = {
    "max_depth": [4, 8, 16, None],
    "min_samples_leaf": [1, 10, 50],
    "class_weight": [None, "balanced"],
}


def build():
    """Return a reproducible, regularized starting model for the shared runner."""
    return DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=10, random_state=config.RANDOM_STATE
    )


def tune(X, y):
    """Tune phishing F1 using shared stratified folds on the training set."""
    search = GridSearchCV(
        build(),
        PARAM_GRID,
        scoring=make_scorer(f1_score, zero_division=0),
        cv=evaluate.cv_splitter(),
        n_jobs=-1,
        error_score="raise",
    )
    return search.fit(X, y)


def train(X, y):
    """Fit the fixed starting configuration for the shared runner."""
    return build().fit(X, y)
