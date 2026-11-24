"""Logistic Regression.  Owner: Madushanka D.M.C  (proposal section 4.9)

build() returns the pipeline with the hyperparameters chosen by grouped_search()
and select_one_se() on the training split (tune_models.py part "lr_followup": the
shared grouped 5-fold CV from evaluate.cv_folds(), url_engineered; see BEST_C
below and notebook 01). They are written in as fixed values so main.py stays fast
and reproducible.
"""
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import config, evaluate

# Default solver of _pipeline(). The search uses one solver per penalty: lbfgs for
# L2 (fast, smooth), liblinear for L1 (lbfgs cannot do L1). build() uses BEST_SOLVER.
SOLVER = "liblinear"
# Fixed iteration cap for every solver and candidate. A fit that reaches it (or raises
# sklearn's ConvergenceWarning) counts as NOT converged and cannot win the search.
MAX_ITER = 1000

# Bounded search space (approved 2026-11-24, replaces the earlier 10-candidate liblinear
# grid whose full-data run did not finish). C = inverse regularisation strength
# (small C = stronger regularisation, simpler model); l1_ratio 0 = L2 (shrinks all
# weights), 1 = L1 (can set weights to exactly 0). L1 stops at C = 1: the classes are
# almost linearly separable, and weak L1 fits with liblinear did not finish in
# 30 minutes. On the full training split even L1 C = 0.1 did not
# converge, so this space was not completed (see FOLLOWUP_SPACE below).
SEARCH_SPACE = [
    {"solver": "lbfgs", "l1_ratio": 0.0, "C": [0.001, 0.01, 0.1, 1.0, 10.0]},
    {"solver": "liblinear", "l1_ratio": 1.0, "C": [0.001, 0.01, 0.1, 1.0]},
]
# The same space in GridSearchCV form, used by tune().
PARAM_GRID = [
    {
        "logisticregression__C": block["C"],
        "logisticregression__l1_ratio": [block["l1_ratio"]],
        "logisticregression__solver": [block["solver"]],
    }
    for block in SEARCH_SPACE
]
# Bounded follow-up search (2026-11-24, tune_models.py part "lr_followup"). The full
# SEARCH_SPACE run was stopped after 36 of 45 fits: L1/liblinear C = 0.1 took 33 min
# for one fold and did not converge, and the remaining L1 fits did not finish in a
# further 55 min. All 25 L2/lbfgs fits had converged, with C = 10 (the upper edge)
# the best. The follow-up therefore re-runs L2/lbfgs only and extends C above 10 to
# check whether the edge was a real optimum. L1 is not re-run (log kept in
# reports/tuning/lr_full_run_INCOMPLETE_2026-11-24.log).
# Round 1 (C up to 100, 40 fits, all converged) put the best at the upper edge again
# (C = 100) with shrinking gains (10 -> 30: +0.00105 F1, 30 -> 100: +0.00022), so one
# pre-registered edge check added C = 300 and 1000 (round 2 = this list, re-run in
# full). The search stops after round 2 whatever the result.
FOLLOWUP_SPACE = [
    {"solver": "lbfgs", "l1_ratio": 0.0,
     "C": [0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]},
]
# Worker processes for grouped_search(): 2 because the machine has 8 GB RAM.
SEARCH_N_JOBS = 2
# Metric that picks the best candidate. F1 on the phishing class balances
# missed phishing (recall) against false alarms (precision); all of
# evaluate.CV_METRICS are still recorded for every candidate.
REFIT_METRIC = "f1"

# Frozen 2026-11-24 from tune_models.py part "lr_followup" round 2 (approved protocol:
# full training split of the group-aware split, 188,297 rows, url_engineered 23 features,
# the shared grouped 5-fold CV, phishing F1; no test rows). 50 fits, all converged
# (n_iter <= 223 of MAX_ITER 1000). Mean grouped-CV F1 (std over 5 folds):
#   C = 1000: 0.99674 (0.00026)   C = 300: 0.99658 (0.00021)   C = 100: 0.99647 (0.00024)
#   C = 30:   0.99625 (0.00020)   C = 10:  0.99520 (0.00037)   C = 1:   0.96997 (0.00561)
# Rule fixed before the run (one-SE): best = 0.99674, SE 0.00013 -> threshold 0.99661;
# the smallest C at or above it is C = 1000 (C = 300 misses by 0.00003).
# LIMITATION: C = 1000 is the upper edge of the searched range. F1 still rises with C,
# but each step from C = 30 on adds only about one SE, so the curve has flattened; the
# search was stopped there as pre-registered. Why weak regularisation wins: the URL
# features nearly separate the classes because every legitimate URL is a bare
# https://www homepage (collection format, see reports/format_baseline.csv); a weakly
# penalised linear model can reproduce that boundary with large weights.
# Superseded: C = 10 from the 2026-11-17 ungrouped tuning on 50 original features
# (liblinear). The full grouped SEARCH_SPACE run was stopped (L1 did not converge).
BEST_C = 1000.0
BEST_L1_RATIO = 0.0
BEST_SOLVER = "lbfgs"
TUNING_NOTE = (
    "Grouped 5-fold CV on the full training split, url_engineered, L2/lbfgs, one-SE rule: "
    "C = 1000 (upper edge of the searched range), phishing F1 = 0.99674 (std 0.00026)."
)


def _pipeline(C: float, l1_ratio: float, solver: str = SOLVER):
    """Scaler + LogisticRegression; the scaler is fitted inside CV folds only."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            l1_ratio=l1_ratio,
            solver=solver,
            max_iter=MAX_ITER,
            random_state=config.RANDOM_STATE,
        ),
    )


def build():
    """Unfitted model with the tuned hyperparameters (BEST_C, BEST_L1_RATIO, BEST_SOLVER)."""
    return _pipeline(C=BEST_C, l1_ratio=BEST_L1_RATIO, solver=BEST_SOLVER)


def train(X, y):
    return build().fit(X, y)


def tune(X_train, y_train, n_jobs=-1, groups=None) -> GridSearchCV:
    """Grid-search C and the penalty type with the shared CV folds.

    Inputs: training features and target only (never the test set), and the
    URL group ids of those rows (approved grouped CV, evaluate.cv_folds()).
    Output: the fitted GridSearchCV; use .best_params_ and .cv_results_.
    Copy the best values into BEST_C / BEST_L1_RATIO / BEST_SOLVER by hand so
    build() is fixed. Not used for the frozen 2026-11-24 values: those come from
    grouped_search() + select_one_se() (tune_models.py), which also record
    convergence per fit.
    """
    search = GridSearchCV(
        build(),
        PARAM_GRID,
        scoring=evaluate.CV_METRICS,
        refit=REFIT_METRIC,
        cv=evaluate.cv_folds(X_train, y_train, groups),
        n_jobs=n_jobs,
    )
    return search.fit(X_train, y_train)


def search_candidates(space=None) -> list:
    """A search space (default SEARCH_SPACE) as a flat list of
    {"penalty", "solver", "l1_ratio", "C"} dicts."""
    return [
        {"penalty": "l1" if block["l1_ratio"] == 1.0 else "l2", "solver": block["solver"],
         "l1_ratio": block["l1_ratio"], "C": float(c)}
        for block in (SEARCH_SPACE if space is None else space) for c in block["C"]
    ]


def _fit_fold(cand: dict, fold: int, X, y, fit_idx, val_idx) -> dict:
    """Fit one candidate on one fold; return its scores and convergence record."""
    row = {**cand, "fold": fold, "max_iter": MAX_ITER}
    start = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = _pipeline(cand["C"], cand["l1_ratio"], cand["solver"]).fit(X[fit_idx], y[fit_idx])
        lr = model[-1]
        n_iter = int(np.max(lr.n_iter_))
        n_warn = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        proba = model.predict_proba(X[val_idx])[:, 1]
        pred = model.predict(X[val_idx])
        yv = y[val_idx]
        row.update({
            "fit_ok": True, "error": "",
            "n_iter": n_iter, "convergence_warnings": n_warn,
            "converged": bool(n_warn == 0 and n_iter < MAX_ITER),
            "n_nonzero_coef": int(np.count_nonzero(lr.coef_)),
            "accuracy": accuracy_score(yv, pred), "precision": precision_score(yv, pred),
            "recall": recall_score(yv, pred), "f1": f1_score(yv, pred),
            "roc_auc": roc_auc_score(yv, proba),
        })
    except Exception as exc:  # a failed fit is recorded, never silently dropped
        row.update({"fit_ok": False, "error": repr(exc), "converged": False})
    row["fit_seconds"] = time.perf_counter() - start
    return row


def grouped_search(X_train, y_train, folds, n_jobs=SEARCH_N_JOBS, log=print,
                   space=None, on_row=None, time_budget=None) -> pd.DataFrame:
    """Bounded grouped-CV search of a search space, one row per (candidate, fold).

    Inputs: training features and target only, and the shared grouped folds
    (evaluate.cv_folds(X_train, y_train, groups)); space defaults to SEARCH_SPACE.
    Every fit records its scores, n_iter_ and ConvergenceWarnings, so
    non-converged candidates can be excluded.
    Progress: one log line per finished fit (n_jobs worker processes).
    on_row(row): called after every finished fit (e.g. to append it to a CSV, so
    an interrupted run keeps its evidence). time_budget: seconds; once exceeded,
    no further results are collected and the unfinished fits are abandoned
    (they are missing from the output, so their candidates cannot be valid).
    """
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train).astype(int)
    tasks = [(c, k, fit, val) for c in search_candidates(space) for k, (fit, val) in enumerate(folds)]
    total, rows, start = len(tasks), [], time.perf_counter()
    jobs = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        delayed(_fit_fold)(c, k, X, y, fit, val) for c, k, fit, val in tasks)
    for done, row in enumerate(jobs, 1):
        rows.append(row)
        if on_row is not None:
            on_row(row)
        status = (f"f1 {row['f1']:.5f} n_iter {row['n_iter']:>4} "
                  f"{'converged' if row['converged'] else 'NOT CONVERGED'}"
                  if row["fit_ok"] else f"FAILED {row['error']}")
        log(f"[{done:>2}/{total}] {row['penalty']}/{row['solver']:<9} C={row['C']:<6g} "
            f"fold {row['fold']} | {status} | {row['fit_seconds']:.1f} s "
            f"| elapsed {time.perf_counter() - start:.0f} s")
        if time_budget is not None and time.perf_counter() - start > time_budget and done < total:
            log(f"TIME BUDGET of {time_budget:.0f} s exceeded: stopping with {done}/{total} fits")
            jobs.close()
            break
    return pd.DataFrame(rows).sort_values(["penalty", "C", "fold"]).reset_index(drop=True)


def select_one_se(per_fold: pd.DataFrame, n_folds: int = config.CV_FOLDS):
    """One-standard-error choice among VALID candidates (converged in every fold).

    A candidate is valid only if all n_folds folds are present, fitted and
    converged, so a candidate cut short by an interrupted run is never valid.

    best = valid candidate with the highest mean fold F1; SE = sample std of its
    fold F1 / sqrt(n_folds). Eligible = valid candidates with mean F1 >= best - SE.
    Chosen = the eligible candidate with the smallest C (strongest regularisation);
    a tie on C goes to the higher mean F1. Rule fixed before the run.
    Output: (one row per candidate, chosen row as dict, best row as dict).
    """
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = []
    for (penalty, solver, C), g in per_fold.groupby(["penalty", "solver", "C"], sort=False):
        ok = g[g["fit_ok"]]
        row = {"penalty": penalty, "solver": solver, "C": C, "max_iter": MAX_ITER,
               "n_folds": len(g), "n_fit_ok": len(ok), "n_converged": int(g["converged"].sum()),
               "valid": bool(len(g) == n_folds and len(ok) == len(g) and g["converged"].all()),
               "n_iter_max": ok["n_iter"].max() if len(ok) else np.nan,
               "convergence_warnings": ok["convergence_warnings"].sum() if len(ok) else np.nan,
               "n_nonzero_coef_mean": ok["n_nonzero_coef"].mean() if len(ok) else np.nan,
               "fit_seconds_mean": g["fit_seconds"].mean()}
        for m in metrics:
            vals = ok[m].to_numpy(dtype=float) if len(ok) else np.array([np.nan])
            row[f"{m}_mean"] = vals.mean()
            row[f"{m}_std"] = vals.std()                       # population std, as sklearn's std_test_*
        f1 = ok["f1"].to_numpy(dtype=float)
        row["f1_se"] = f1.std(ddof=1) / np.sqrt(len(f1)) if len(f1) > 1 else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    valid = table[table["valid"]]
    if valid.empty:
        table["within_1se_of_best"] = False
        table["chosen"] = False
        return table.sort_values("f1_mean", ascending=False), None, None
    best = valid.loc[valid["f1_mean"].idxmax()]
    eligible = valid[valid["f1_mean"] >= best["f1_mean"] - best["f1_se"]]
    chosen = eligible.sort_values(["C", "f1_mean"], ascending=[True, False]).iloc[0]
    table["within_1se_of_best"] = table["valid"] & (table["f1_mean"] >= best["f1_mean"] - best["f1_se"])
    table["is_best_mean_f1"] = (table["penalty"] == best["penalty"]) & (table["C"] == best["C"])
    table["chosen"] = (table["penalty"] == chosen["penalty"]) & (table["C"] == chosen["C"])
    table = table.sort_values("f1_mean", ascending=False).reset_index(drop=True)
    return table, chosen.to_dict(), best.to_dict()
