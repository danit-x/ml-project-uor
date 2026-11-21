"""Support Vector Machine (proposal section 4.9).

Owner: Dharmasiri S.O.R.
Use a linear SVM for this large tabular dataset. The estimator and its scaler
are calibrated within CV folds so probabilities can be used for ROC-AUC and
phishing risk scoring without fitting preprocessing on held-out folds.
"""
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .. import config, evaluate

# These are safe reproducible baseline values, NOT claimed as tuned optima.
# After tune() runs on the agreed training data, replace them with its best
# settings and record the experiment in notebook 03.
BEST_C = 1.0
BEST_CLASS_WEIGHT = None
REFIT_METRIC = "f1"
PARAM_GRID = {
    "estimator__linearsvc__C": [0.01, 0.1, 1.0, 10.0],
    "estimator__linearsvc__class_weight": [None, "balanced"],
}


def _pipeline(C: float, class_weight=None):
    """Calibrated LinearSVC with the scaler INSIDE the calibrated estimator.

    Scaling is fitted only on the training partition of each inner fold.
    LinearSVC is preferable to a kernel SVC for ~235k rows and ~50 features.
    """
    base = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=C,
            class_weight=class_weight,
            dual=False,
            max_iter=5000,
            tol=1e-3,
            random_state=config.RANDOM_STATE,
        ),
    )
    return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)


def build():
    """Return an unfitted SVM with fixed, reproducible parameters."""
    return _pipeline(C=BEST_C, class_weight=BEST_CLASS_WEIGHT)


def train(X, y):
    """Fit the configured SVM and return a model supporting predict_proba()."""
    return build().fit(X, y)


def tune(X_train, y_train, n_jobs=1):
    """Grid-search C and class weighting using shared, stratified training CV.

    Pass TRAINING data only. Start on a training subset (not on test rows).
    Each candidate includes probability calibration; this is more costly
    than tuning LinearSVC alone but matches the final predictor.
    """
    search = GridSearchCV(
        build(),
        PARAM_GRID,
        cv=evaluate.cv_splitter(),
        scoring=evaluate.CV_METRICS,
        refit=REFIT_METRIC,
        n_jobs=n_jobs,
        error_score="raise",
    )
    return search.fit(X_train, y_train)


def tuning_table(search, top=10):
    """Show candidates ordered by mean phishing-class CV F1 (not test metrics)."""
    results = pd.DataFrame(search.cv_results_)
    cols = [
        "params", "mean_test_f1", "std_test_f1",
        "mean_test_recall", "mean_test_precision",
        "mean_test_roc_auc", "mean_fit_time",
    ]
    return results.sort_values("rank_test_f1")[cols].head(top).reset_index(drop=True)
