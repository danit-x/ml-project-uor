"""Training-only grouped tuning of the three main models (progress evaluation).

    python tune_models.py                    # full training split (about 1.5 h, RF dominates)
    python tune_models.py --sample 20000     # quick check of the script itself
    python tune_models.py --parts fs dt      # only some parts

Protocol, identical for every part:
- rows: the TRAINING rows of the group-aware split (data.split_index); the
  test rows are never loaded into any search;
- features: the primary url_engineered set (features.build_feature_sets on
  the training rows);
- folds: evaluate.cv_folds(X_train, y_train, groups), the same five grouped,
  stratified, seeded folds for every model (cached on rows + labels + groups);
- positive class: phishing (target 1); the winner is chosen by phishing F1.

Parts (each writes CSVs to reports/tuning/, suffix _sample_N for --sample runs):
  fs  per-fold feature-selection check: is the redundancy filter, refitted on
      each fold's fitting rows, the same as the filter fitted on all training
      rows? Plus how far each decision is from the 0.98 threshold.
  dt  tree.tune() on tree.PARAM_GRID, then a complexity review: depth, leaves
      and nodes of the fitted fold trees for every grid configuration and for a
      max_depth curve (leaf 1, no class weight). Decision rule, fixed before
      the run (CART one-standard-error rule, Breiman et al. 1984; Hastie et al.,
      ESL 7.10): among all evaluated configurations, take the one with the
      fewest leaves whose mean CV F1 >= best mean - SE(best), where
      SE = sample std of the best's 5 fold scores / sqrt(5).
  lr  logreg.grouped_search() on logreg.SEARCH_SPACE (bounded: L2/lbfgs C 0.001-10,
      L1/liblinear C 0.001-1; 9 candidates, 45 fits, 2 workers, fixed max_iter).
      Every fit records n_iter_ and ConvergenceWarnings (lr_grouped_folds.csv);
      a candidate is valid only if it converged in all folds. Decision rule,
      fixed before the run: smallest C whose mean CV F1 >= best valid - SE(best).
  rf  forest.tune(n_iter=20) on forest.PARAM_DISTRIBUTIONS (20 of 512, 100 fits).

Bounded parts used for the progress evaluation (not in the default run):
  lr_followup  logreg.FOLLOWUP_SPACE (L2/lbfgs only) after the full "lr" run was
      stopped at 36/45 fits (L1/liblinear did not finish). Same rows/folds/rule as
      "lr"; every fit is appended to lr_followup_folds_partial.csv immediately.
  rf_subset   8 RF candidates (forest.SUBSET_PARAM_GRID, n_estimators fixed) on a
      group-preserving subset of the training rows (data.group_subset,
      forest.SUBSET_ROWS); grouped 5-fold CV on that subset; one-SE rule on mean
      leaves per tree. Subset CV numbers are tuning evidence only; the chosen
      forest is then trained on the full training split by main.py.
  Both stop collecting results once --time-budget seconds have passed and keep
  what finished (python tune_models.py --parts lr_followup rf_subset).

The chosen values are then copied by hand into the BEST_* constants of each
model module (CLAUDE.md "Tuning"), and main.py uses only those fixed values.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_validate

from src import config, data, evaluate, features
from src.models import forest, logreg, tree

OUT_DIR = config.RESULTS_DIR / "tuning"
DT_DEPTH_CURVE = [8, 12, 16, 20, 24, 28, 32, None]
PARTS = ("fs", "dt", "lr", "rf", "lr_followup", "rf_subset")
DEFAULT_PARTS = ("fs", "dt", "lr", "rf")


def _name(stem, sample, ext="csv"):
    return OUT_DIR / (f"{stem}_sample_{sample}.{ext}" if sample else f"{stem}.{ext}")


def prepare(sample=None):
    """Training rows, url_engineered features, labels, groups and the shared folds."""
    df = data.load_clean_data()
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=config.RANDOM_STATE).reset_index(drop=True)
    df = features.add_features(df)
    groups = data.group_ids(df)
    train_idx, _ = data.split_index(df)        # the test rows are not used below
    train_df = df.loc[train_idx]
    sets = features.build_feature_sets(train_df)
    cols = sets[features.PRIMARY_FEATURE_SET]
    y = data.get_target(df).loc[train_idx]
    g = groups.loc[train_idx]
    folds = evaluate.cv_folds(train_df[cols], y, g)
    return train_df, sets, cols, y, g, folds


def _search_summary(search, seconds, n_candidates, grid_note):
    res = pd.DataFrame(search.cv_results_)
    best = res.loc[search.best_index_]
    k = config.CV_FOLDS
    folds_f1 = np.array([best[f"split{i}_test_f1"] for i in range(k)])
    se = folds_f1.std(ddof=1) / np.sqrt(k)
    within = int((res["mean_test_f1"] >= best["mean_test_f1"] - se).sum())
    return {
        "best_params": {key: (None if v is None else v) for key, v in search.best_params_.items()},
        "mean_cv_f1": float(best["mean_test_f1"]),
        "std_cv_f1": float(best["std_test_f1"]),
        "se_cv_f1": float(se),
        "mean_cv_roc_auc": float(best["mean_test_roc_auc"]),
        "mean_cv_precision": float(best["mean_test_precision"]),
        "mean_cv_recall": float(best["mean_test_recall"]),
        "mean_cv_accuracy": float(best["mean_test_accuracy"]),
        "candidates_within_1se_of_best": within,
        "n_candidates": n_candidates,
        "n_fits": n_candidates * k,
        "search_space": grid_note,
        "seconds": round(seconds, 1),
    }


def _save_search(search, stem, sample):
    res = pd.DataFrame(search.cv_results_).drop(columns=["params"])
    res.sort_values("rank_test_f1").to_csv(_name(stem, sample), index=False)


def part_fs(train_df, sets, folds, sample):
    report = features.fold_selection_report(train_df, folds, sets)
    report.assign(added=report["added"].map(";".join), removed=report["removed"].map(";".join)) \
        .to_csv(_name("feature_selection_folds", sample), index=False)
    margin = features.correlation_margin(train_df, sets)
    margin.to_csv(_name("feature_selection_margin", sample), index=False)
    print(report[["fold", "feature_set", "n_selected", "same_as_outer"]].to_string(index=False))
    print(margin.round(5).to_string(index=False))
    return {
        "folds_x_sets_checked": len(report),
        "all_same_as_outer": bool(report["same_as_outer"].all()),
        "max_abs_r_between_kept": {
            r.feature_set: round(r.abs_r, 5)
            for r in margin.itertuples() if r.kind.startswith("max")
        },
        "min_abs_r_of_removed": {
            name: (round(float(m.abs_r.min()), 5) if len(m) else None)
            for name, m in margin[~margin.kind.str.startswith("max")].groupby("feature_set")
        },
    }


def _tree_complexity(params, X, y, folds):
    """Grouped CV of one tree configuration plus the size of its fold trees."""
    model = clone(tree.build()).set_params(**params)
    cv = cross_validate(model, X, y, cv=folds, scoring=tree.SCORING,
                        return_estimator=True, n_jobs=-1, error_score="raise")
    est = cv["estimator"]
    f1 = cv["test_f1"]
    return {
        "max_depth": params["max_depth"], "min_samples_leaf": params["min_samples_leaf"],
        "class_weight": params["class_weight"],
        "f1_mean": f1.mean(), "f1_std": f1.std(), "f1_se": f1.std(ddof=1) / np.sqrt(len(f1)),
        "roc_auc_mean": cv["test_roc_auc"].mean(), "precision_mean": cv["test_precision"].mean(),
        "recall_mean": cv["test_recall"].mean(), "accuracy_mean": cv["test_accuracy"].mean(),
        "depth_mean": np.mean([e.get_depth() for e in est]),
        "leaves_mean": np.mean([e.get_n_leaves() for e in est]),
        "nodes_mean": np.mean([e.tree_.node_count for e in est]),
        "fit_seconds_mean": cv["fit_time"].mean(),
    }


def part_dt(X, y, g, folds, sample):
    start = time.perf_counter()
    search = tree.tune(X, y, groups=g)
    seconds = time.perf_counter() - start
    _save_search(search, "dt_grouped_search", sample)
    n_grid = len(search.cv_results_["params"])
    summary = _search_summary(search, seconds, n_grid, "tree.PARAM_GRID (full grid)")

    configs = [dict(p) for p in search.cv_results_["params"]]
    for depth in DT_DEPTH_CURVE:
        p = {"max_depth": depth, "min_samples_leaf": 1, "class_weight": None}
        if p not in configs:
            configs.append(p)
    rows = []
    for p in configs:
        row = _tree_complexity(p, X, y, folds)
        row["source"] = "grid" if p in search.cv_results_["params"] else "depth curve"
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)
    best = table.iloc[0]
    eligible = table[table["f1_mean"] >= best["f1_mean"] - best["f1_se"]]
    chosen = eligible.sort_values(["leaves_mean", "f1_mean"], ascending=[True, False]).iloc[0]
    table["within_1se_of_best"] = table["f1_mean"] >= best["f1_mean"] - best["f1_se"]
    table.to_csv(_name("dt_complexity", sample), index=False)
    show = ["source", "max_depth", "min_samples_leaf", "class_weight", "f1_mean", "f1_std",
            "depth_mean", "leaves_mean", "nodes_mean", "within_1se_of_best"]
    print(table[show].head(15).round(5).to_string(index=False))
    summary["complexity_rule"] = "fewest leaves with mean F1 >= best - SE(best)"
    summary["one_se_choice"] = {
        "max_depth": None if pd.isna(chosen["max_depth"]) else int(chosen["max_depth"]),
        "min_samples_leaf": int(chosen["min_samples_leaf"]),
        "class_weight": None if pd.isna(chosen["class_weight"]) else chosen["class_weight"],
        "f1_mean": float(chosen["f1_mean"]), "f1_std": float(chosen["f1_std"]),
        "depth_mean": float(chosen["depth_mean"]), "leaves_mean": float(chosen["leaves_mean"]),
    }
    summary["best_tree_size"] = {
        "depth_mean": float(best["depth_mean"]), "leaves_mean": float(best["leaves_mean"]),
        "nodes_mean": float(best["nodes_mean"]),
    }
    return summary


def part_lr(X, y, folds, sample):
    start = time.perf_counter()
    per_fold = logreg.grouped_search(X, y, folds, log=lambda s: print(s, flush=True))
    seconds = time.perf_counter() - start
    per_fold.to_csv(_name("lr_grouped_folds", sample), index=False)
    table, chosen, best = logreg.select_one_se(per_fold)
    table.to_csv(_name("lr_grouped_search", sample), index=False)
    show = ["penalty", "solver", "C", "valid", "n_converged", "n_iter_max", "f1_mean", "f1_std",
            "f1_se", "roc_auc_mean", "n_nonzero_coef_mean", "within_1se_of_best", "chosen"]
    print(table[show].round(5).to_string(index=False))

    def _pick(row):
        if row is None:
            return None
        keys = ["penalty", "solver", "C", "f1_mean", "f1_std", "f1_se", "roc_auc_mean",
                "precision_mean", "recall_mean", "accuracy_mean", "n_converged", "n_iter_max",
                "n_nonzero_coef_mean"]
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in ((k, row[k]) for k in keys)}

    return {
        "search_space": logreg.SEARCH_SPACE, "max_iter": logreg.MAX_ITER,
        "n_jobs": logreg.SEARCH_N_JOBS, "n_candidates": int(len(table)),
        "n_fits": int(len(per_fold)), "n_fits_ok": int(per_fold["fit_ok"].sum()),
        "n_fits_converged": int(per_fold["converged"].sum()),
        "n_fits_not_converged": int((per_fold["fit_ok"] & ~per_fold["converged"]).sum()),
        "n_fits_failed": int((~per_fold["fit_ok"]).sum()),
        "n_valid_candidates": int(table["valid"].sum()),
        "selection_rule": "valid = converged in every fold; best = highest mean F1 among valid; "
                          "chosen = smallest C with mean F1 >= best - SE(best), "
                          "SE = sample std of best's fold F1 / sqrt(k); tie on C -> higher mean F1",
        "best_mean_f1": _pick(best),
        "one_se_choice": _pick(chosen),
        "chosen_converged_all_folds": None if chosen is None
        else bool(chosen["n_converged"] == chosen["n_folds"]),
        "seconds": round(seconds, 1),
    }


def _csv_appender(path):
    """on_row callback that appends each finished fit to a CSV (header once)."""
    if path.exists():
        path.unlink()

    def append(row):
        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
    return append


def part_lr_followup(X, y, folds, sample, time_budget):
    """Bounded L2/lbfgs re-run (logreg.FOLLOWUP_SPACE) after the stopped full search.

    Same rows, features and folds as part "lr". Each finished fit is appended to
    lr_followup_folds_partial.csv at once, so an interrupted run keeps its
    evidence. The one-SE rule (logreg.select_one_se) is applied to the valid
    (converged in all 5 folds) candidates only.
    """
    partial = _name("lr_followup_folds_partial", sample)
    start = time.perf_counter()
    per_fold = logreg.grouped_search(
        X, y, folds, log=lambda s: print(s, flush=True), space=logreg.FOLLOWUP_SPACE,
        on_row=_csv_appender(partial), time_budget=time_budget)
    seconds = time.perf_counter() - start
    per_fold.to_csv(_name("lr_followup_folds", sample), index=False)
    table, chosen, best = logreg.select_one_se(per_fold)
    table.to_csv(_name("lr_followup_search", sample), index=False)
    show = ["penalty", "solver", "C", "valid", "n_converged", "n_iter_max", "f1_mean", "f1_std",
            "f1_se", "roc_auc_mean", "within_1se_of_best", "chosen"]
    print(table[show].round(5).to_string(index=False))

    def _pick(row):
        if row is None:
            return None
        keys = ["penalty", "solver", "C", "f1_mean", "f1_std", "f1_se", "roc_auc_mean",
                "precision_mean", "recall_mean", "accuracy_mean", "n_converged", "n_iter_max",
                "n_nonzero_coef_mean"]
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in ((k, row[k]) for k in keys)}

    n_expected = len(logreg.search_candidates(logreg.FOLLOWUP_SPACE)) * len(folds)
    c_max = max(logreg.FOLLOWUP_SPACE[0]["C"])
    return {
        "search_space": logreg.FOLLOWUP_SPACE, "max_iter": logreg.MAX_ITER,
        "n_jobs": logreg.SEARCH_N_JOBS, "time_budget_s": time_budget,
        "n_fits_expected": n_expected, "n_fits": int(len(per_fold)),
        "complete": bool(len(per_fold) == n_expected),
        "n_fits_converged": int(per_fold["converged"].sum()),
        "n_valid_candidates": int(table["valid"].sum()),
        "selection_rule": "valid = converged in every fold; best = highest mean F1 among valid; "
                          "chosen = smallest C with mean F1 >= best - SE(best)",
        "best_mean_f1": _pick(best),
        "one_se_choice": _pick(chosen),
        "best_at_upper_edge_of_C": None if best is None else bool(best["C"] == c_max),
        "earlier_full_run": "stopped after 36/45 fits; see lr_full_run_INCOMPLETE_2026-11-24.log "
                            "(L1/liblinear C=0.1 fold 1 not converged; 9 L1 fits never finished)",
        "seconds": round(seconds, 1),
    }


def part_rf_subset(train_df, cols, y, g, sample, time_budget):
    """Bounded grouped RF search on a group-preserving subset of the training rows.

    Subset: data.group_subset(g, forest.SUBSET_ROWS) (whole URL groups, seeded).
    Folds: evaluate.cv_folds on the subset's rows and groups (5 grouped folds).
    Grid: forest.SUBSET_PARAM_GRID with n_estimators = forest.SUBSET_N_ESTIMATORS.
    Every candidate also records the mean leaves and depth per tree of its fold
    forests. Rule fixed before the run (as for the tree): best = highest mean
    F1; chosen = the candidate with the fewest mean leaves per tree among those
    with mean F1 >= best - SE(best). One CSV row is appended per candidate, and
    the time budget is checked between candidates.
    """
    from itertools import product
    from sklearn.ensemble import RandomForestClassifier

    sub = data.group_subset(g, forest.SUBSET_ROWS)
    Xs, ys, gs = train_df.loc[sub, cols], y.loc[sub], g.loc[sub]
    folds = evaluate.cv_folds(Xs, ys, gs)
    print(f"RF subset: {len(sub):,} training rows, {gs.nunique():,} groups, phishing share "
          f"{ys.mean():.4f} (full training {y.mean():.4f}); folds {[len(v) for _, v in folds]}",
          flush=True)
    grid = forest.SUBSET_PARAM_GRID
    candidates = [dict(zip(grid, values)) for values in product(*grid.values())]
    partial = _name("rf_subset_search_partial", sample)
    append = _csv_appender(partial)
    rows, start = [], time.perf_counter()
    for i, params in enumerate(candidates, 1):
        if time.perf_counter() - start > time_budget:
            print(f"TIME BUDGET of {time_budget:.0f} s exceeded: stopping with {i - 1}/"
                  f"{len(candidates)} candidates", flush=True)
            break
        model = RandomForestClassifier(n_estimators=forest.SUBSET_N_ESTIMATORS, n_jobs=-1,
                                       random_state=config.RANDOM_STATE, **params)
        t0 = time.perf_counter()
        cv = cross_validate(model, Xs, ys, cv=folds, scoring=evaluate.CV_METRICS,
                            return_estimator=True, error_score="raise")
        est = cv["estimator"]
        f1 = cv["test_f1"]
        row = {**params, "n_estimators": forest.SUBSET_N_ESTIMATORS,
               "f1_mean": f1.mean(), "f1_std": f1.std(), "f1_se": f1.std(ddof=1) / np.sqrt(len(f1)),
               **{f"{m}_mean": cv[f"test_{m}"].mean() for m in evaluate.CV_METRICS if m != "f1"},
               **{f"f1_fold{k}": v for k, v in enumerate(f1)},
               "leaves_per_tree_mean": np.mean([t.get_n_leaves() for e in est for t in e.estimators_]),
               "depth_per_tree_mean": np.mean([t.get_depth() for e in est for t in e.estimators_]),
               "fit_seconds_mean": cv["fit_time"].mean(), "seconds": time.perf_counter() - t0}
        del cv, est
        rows.append(row)
        append(row)
        print(f"[{i}/{len(candidates)}] {params} | f1 {row['f1_mean']:.5f} ± {row['f1_std']:.5f} | "
              f"leaves/tree {row['leaves_per_tree_mean']:.0f} | {row['seconds']:.0f} s | "
              f"elapsed {time.perf_counter() - start:.0f} s", flush=True)
    table = pd.DataFrame(rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)
    best = table.iloc[0]
    table["within_1se_of_best"] = table["f1_mean"] >= best["f1_mean"] - best["f1_se"]
    chosen = table[table["within_1se_of_best"]].sort_values(
        ["leaves_per_tree_mean", "f1_mean"], ascending=[True, False]).iloc[0]
    table["chosen"] = table.index == chosen.name
    table.to_csv(_name("rf_subset_search", sample), index=False)
    print(table.drop(columns=[c for c in table if c.startswith("f1_fold")]).round(5).to_string(index=False))

    def _value(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        v = v.item() if hasattr(v, "item") else v
        return int(v) if isinstance(v, float) and v.is_integer() and v > 1 else v  # 20.0 -> 20, keep 0.5

    def _params(row):
        return {k: _value(row[k]) for k in grid} | {"n_estimators": forest.SUBSET_N_ESTIMATORS}

    return {
        "protocol": "group-preserving subset of the TRAINING rows (data.group_subset, seed "
                    f"{config.RANDOM_STATE}); grouped 5-fold CV on the subset; url_engineered",
        "subset_rows": int(len(sub)), "subset_groups": int(gs.nunique()),
        "subset_phishing_share": float(ys.mean()),
        "grid": grid, "n_estimators_fixed": forest.SUBSET_N_ESTIMATORS,
        "n_candidates_planned": len(candidates), "n_candidates_done": len(rows),
        "complete": len(rows) == len(candidates), "time_budget_s": time_budget,
        "selection_rule": "fewest mean leaves per tree with mean F1 >= best - SE(best)",
        "best": {**_params(best), "f1_mean": float(best["f1_mean"]), "f1_std": float(best["f1_std"]),
                 "f1_se": float(best["f1_se"]), "leaves_per_tree_mean": float(best["leaves_per_tree_mean"])},
        "one_se_choice": {**_params(chosen), "f1_mean": float(chosen["f1_mean"]),
                          "f1_std": float(chosen["f1_std"]),
                          "leaves_per_tree_mean": float(chosen["leaves_per_tree_mean"])},
        "seconds": round(time.perf_counter() - start, 1),
    }


def part_rf(X, y, g, sample):
    search, seconds = forest.tune(X, y, n_iter=20, scoring="f1", groups=g)
    _save_search(search, "rf_grouped_search", sample)
    n = len(search.cv_results_["params"])
    summary = _search_summary(
        search, seconds, n,
        "forest.PARAM_DISTRIBUTIONS, RandomizedSearchCV n_iter=20 of 512 combinations",
    )
    print(forest.search_table(search, top=10).to_string(index=False))
    return summary


def main(sample=None, parts=DEFAULT_PARTS, time_budget=1500):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    train_df, sets, cols, y, g, folds = prepare(sample)
    X = train_df[cols]
    print(f"Training rows {len(X):,} | {features.PRIMARY_FEATURE_SET}: {len(cols)} features | "
          f"folds {[len(v) for _, v in folds]} | prep {time.perf_counter() - start:.0f} s")
    summary_path = _name("tuning_summary", sample, "json")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary["protocol"] = {
        "rows": "training rows of data.split_index (group-aware); test rows unused",
        "n_train_rows": len(X), "feature_set": features.PRIMARY_FEATURE_SET,
        "features": cols, "cv": f"evaluate.cv_folds: {config.CV_FOLDS} grouped stratified folds",
        "positive_class": "phishing (1)", "refit_metric": "f1", "sample": sample,
    }
    runners = {
        "fs": lambda: part_fs(train_df, sets, folds, sample),
        "dt": lambda: part_dt(X, y, g, folds, sample),
        "lr": lambda: part_lr(X, y, folds, sample),
        "rf": lambda: part_rf(X, y, g, sample),
        "lr_followup": lambda: part_lr_followup(X, y, folds, sample, time_budget),
        "rf_subset": lambda: part_rf_subset(train_df, cols, y, g, sample, time_budget),
    }
    for part in parts:
        print(f"\n=== {part} ===", flush=True)
        summary[part] = runners[part]()
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        print(json.dumps(summary[part], indent=2, default=str), flush=True)
    print(f"\nSaved {summary_path} | total {(time.perf_counter() - start) / 60:.1f} min")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=list(DEFAULT_PARTS))
    parser.add_argument("--time-budget", type=float, default=1500,
                        help="seconds per bounded part (lr_followup, rf_subset); partial results are kept")
    args = parser.parse_args()
    main(args.sample, args.parts, args.time_budget)
