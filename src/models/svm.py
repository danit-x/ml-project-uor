"""Linear SVM for URLSentra phishing classification (section 4.9).

Owner: Dharmasiri S.O.R.  In the processed target, 1 = phishing.
"""
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .. import config, evaluate

# Provisional best C = 0.1, selected using training-only
# 5-fold CV on 2000 training rows (original features).
# Retune after final feature engineering is merged.
BEST_C = 0.1
BEST_CLASS_WEIGHT = None
REFIT_METRIC = "f1"
PARAM_GRID = {
    "estimator__linearsvc__C": [0.01, 0.1, 1.0, 10.0],
}


def configured(C=BEST_C, class_weight=BEST_CLASS_WEIGHT):
    """Unfitted calibrated linear SVM.

    StandardScaler is INSIDE CalibratedClassifierCV to ensure each internal
    calibration fold fits its own scaler, not a scaler on the entire dataset.
    Calibration provides predict_proba() for common evaluation and risk scores.
    """
    estimator = make_pipeline(
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
    return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=3)


def build():
    """Unfitted estimator for main.py; parameters remain fixed/reproducible."""
    return configured(C=BEST_C, class_weight=BEST_CLASS_WEIGHT)


def train(X, y):
    """Fit the configured estimator; support predict() and predict_proba()."""
    return build().fit(X, y)


def tune(X_train, y_train, n_jobs=1, groups=None):
    """Tune C using shared training CV; never pass test rows.

    C is tuned on a modest TRAINING SUBSET first; the scoring protocol stays
    identical to the other models. Expensive nested calibration is intentional.
    groups: URL group ids of the training rows (approved grouped CV,
    evaluate.cv_folds). Without them the legacy ungrouped folds are used and
    an UngroupedCVWarning is emitted; BEST_C above came from such a run, so
    the SVM stays PROVISIONAL until it is retuned with groups.
    """
    search = GridSearchCV(
        build(),
        param_grid=PARAM_GRID,
        scoring=evaluate.CV_METRICS,
        refit=REFIT_METRIC,
        cv=evaluate.cv_folds(X_train, y_train, groups),
        n_jobs=n_jobs,
        error_score="raise",
    )
    return search.fit(X_train, y_train)


def tuning_table(search):
    """Return measured training CV scores, sorted by phishing-class mean F1."""
    frame = pd.DataFrame(search.cv_results_)
    columns = [
        "params", "mean_test_f1", "std_test_f1", "mean_test_recall",
        "mean_test_precision", "mean_test_roc_auc", "mean_fit_time",
    ]
    return frame.sort_values("rank_test_f1")[columns].reset_index(drop=True)
