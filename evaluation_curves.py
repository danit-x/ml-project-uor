"""Learning curves and ROC curves of the three official models (diagnostics only).

    python evaluation_curves.py                  # full data: grouped-CV curves + held-out ROC (~20-30 min)
    python evaluation_curves.py --sample 20000   # quick check of the script (CV parts only, *_sample_N files)
    python evaluation_curves.py --parts test     # only the held-out ROC of the saved models (~2 min)

Nothing here changes a model, a hyperparameter, a threshold or a feature set,
and nothing here is used to select a model. Protocol:

- cv (learning curves + grouped-CV ROC): the TRAINING rows of the group-aware
  split only, the same five grouped folds as reports/cv_results.csv
  (evaluate.cv_folds), each official model as
  make_pipeline(FeatureSetSelector("url_engineered"), build()) so feature
  selection is refitted on every fitting subset. Sizes 25 %, 50 %, 100 % of
  each fold's training part (whole URL groups). The 100 % fits are the
  cv_results.csv fits, so their mean AUC / F1 are checked against that file.
- test (held-out ROC): the frozen models already in saved_models/ (no fitting),
  on the held-out rows rebuilt with the same cleaning and split as main.py;
  the row count must equal manifest.json (47,073) and the AUCs must equal
  reports/results.csv. Run once, after every choice was frozen.

Outputs (full data): reports/learning_curves.csv (mean/std), reports/learning_curves_folds.csv
(every fold and size, with the columns the refitted selector kept vs the frozen set),
reports/roc_cv_auc.csv, reports/roc_test_auc.csv and
figures/{learning_curves,roc_cv,roc_test}_url_engineered.png.
"""
import argparse
import json
import time

import joblib
import pandas as pd
from sklearn.pipeline import make_pipeline

from src import config, data, evaluate, features
from src.models import MODELS

OFFICIAL = ("Logistic Regression", "Decision Tree", "Random Forest")
PRIMARY = features.PRIMARY_FEATURE_SET
AUC_TOLERANCE = 1e-6      # same folds and seeds: agreement should be exact up to float noise


def _name(stem, sample, ext="csv"):
    return f"{stem}_sample_{sample}.{ext}" if sample else f"{stem}.{ext}"


def load_split(sample=None):
    """Clean data with engineered features, URL groups and the main.py split (train/test disjoint)."""
    df = data.load_clean_data()
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=config.RANDOM_STATE).reset_index(drop=True)
    df = features.add_features(df)
    groups = data.group_ids(df)
    train_idx, test_idx = data.split_index(df)
    if set(train_idx) & set(test_idx) or set(groups.loc[train_idx]) & set(groups.loc[test_idx]):
        raise AssertionError("train and test rows or URL groups overlap")
    return df, groups, train_idx, test_idx


def training_inputs(df, groups, train_idx):
    """X (every candidate column), y and groups of the TRAINING rows only."""
    X = df.loc[train_idx, features.candidate_columns()]
    return X, data.get_target(df).loc[train_idx], groups.loc[train_idx]


def run_cv(df, groups, train_idx, sample):
    X, y, g = training_inputs(df, groups, train_idx)
    tables, roc = [], {}
    for name in OFFICIAL:
        t0 = time.time()
        pipe = make_pipeline(features.FeatureSetSelector(PRIMARY), MODELS[name].build())
        table, roc[name] = evaluate.grouped_learning_curve(pipe, X, y, g)
        tables.append(table.assign(model=name))
        print(f"  {name}: {len(table)} grouped fits in {time.time() - t0:.0f} s")
    per_fold = pd.concat(tables, ignore_index=True)
    # Which columns the selector, refitted on each fitting subset, kept compared with the
    # set fitted on all training rows (= the frozen models' 23 columns in manifest.json).
    frozen = features.build_feature_sets(df.loc[train_idx])[PRIMARY]
    chosen = per_fold.pop("selected").str.split("|")
    per_fold["n_selected"] = chosen.map(len)
    per_fold["dropped_vs_frozen"] = chosen.map(lambda c: "|".join(x for x in frozen if x not in c))
    per_fold["added_vs_frozen"] = chosen.map(lambda c: "|".join(x for x in c if x not in frozen))
    per_fold.insert(0, "model", per_fold.pop("model"))
    per_fold.to_csv(config.RESULTS_DIR / _name("learning_curves_folds", sample), index=False)
    summary = evaluate.summarize_learning_curve(per_fold)
    summary.insert(1, "feature_set", PRIMARY)
    summary.to_csv(config.RESULTS_DIR / _name("learning_curves", sample), index=False)
    print(summary.round(5).to_string(index=False))

    # Consistency with reports/cv_results*.csv: the 100 % fits are the same fits.
    cv_file = config.RESULTS_DIR / _name("cv_results", sample)
    cv = pd.read_csv(cv_file) if cv_file.exists() else None
    rows = []
    for name in OFFICIAL:
        full = per_fold.loc[(per_fold["model"] == name) & (per_fold["train_size"] == 1.0)].sort_values("fold")
        row = {"model": name, "feature_set": PRIMARY, "protocol": "grouped 5-fold CV, training rows"}
        row |= {f"auc_fold_{k + 1}": v for k, v in enumerate(full["valid_roc_auc"])}
        row |= {"auc_mean": full["valid_roc_auc"].mean(), "auc_std": full["valid_roc_auc"].std(ddof=0),
                "valid_f1_mean": full["valid_f1"].mean()}
        if cv is not None:
            ref = cv.loc[(cv["feature_set"] == PRIMARY) & (cv["model"] == name)].iloc[0]
            row |= {"cv_results_roc_auc_mean": ref["roc_auc_mean"], "cv_results_f1_mean": ref["f1_mean"],
                    "abs_diff_auc": abs(row["auc_mean"] - ref["roc_auc_mean"]),
                    "abs_diff_f1": abs(row["valid_f1_mean"] - ref["f1_mean"])}
        rows.append(row)
    auc = pd.DataFrame(rows)
    auc.to_csv(config.RESULTS_DIR / _name("roc_cv_auc", sample), index=False)
    print(auc.drop(columns=["protocol"]).round(6).to_string(index=False))
    if cv is None:
        print(f"  {cv_file.name} not found: AUC consistency not checked")
    elif (auc[["abs_diff_auc", "abs_diff_f1"]] > AUC_TOLERANCE).any().any():
        print(f"  WARNING: grouped-CV AUC/F1 differ from {cv_file.name} by more than {AUC_TOLERANCE}")
    else:
        print(f"  grouped-CV AUC and F1 match {cv_file.name} (|diff| <= {AUC_TOLERANCE})")

    rule = format_rule_point(df, train_idx, cv_folds=evaluate.cv_folds(X, y, g))
    n = len(train_idx)
    evaluate.plot_learning_curves(
        summary, f"Learning curves, {PRIMARY}: grouped 5-fold CV on {n:,} training rows (mean ± std)",
        _name(f"learning_curves_{PRIMARY}", sample, "png"), per_fold=per_fold)
    evaluate.plot_cv_roc(
        roc, f"Grouped 5-fold CV ROC, {PRIMARY}\ntraining rows only, mean ± std over folds",
        _name(f"roc_cv_{PRIMARY}", sample, "png"), rule_point=rule)


def format_rule_point(df, rows, cv_folds=None):
    """(FPR, TPR) of the fixed rule 'not a bare https://www homepage -> phishing' on these rows.

    With cv_folds: the mean over the validation folds (as reports/format_baseline.csv).
    """
    y = data.get_target(df).loc[rows].to_numpy()
    pred = (~data.bare_homepage_mask(df.loc[rows, "URL"])).astype(int).to_numpy()
    parts = [v for _, v in cv_folds] if cv_folds is not None else [slice(None)]
    fprs = [((pred[p] == 1) & (y[p] == 0)).sum() / (y[p] == 0).sum() for p in parts]
    tprs = [((pred[p] == 1) & (y[p] == 1)).sum() / (y[p] == 1).sum() for p in parts]
    return float(sum(fprs) / len(fprs)), float(sum(tprs) / len(tprs))


def run_test(df, test_idx):
    """Held-out ROC of the saved, frozen official models (loaded, never fitted)."""
    manifest = json.loads((config.MODELS_DIR / "manifest.json").read_text())
    if manifest.get("sample"):
        raise SystemExit("saved_models/ come from a --sample run; the held-out ROC needs the full-data models")
    if len(test_idx) != manifest["n_test_rows"]:
        raise SystemExit(f"{len(test_idx)} test rows, manifest says {manifest['n_test_rows']}")
    cols = manifest["feature_sets"][PRIMARY]
    files = {name: config.MODELS_DIR / f"{name.lower().replace(' ', '_')}_{PRIMARY}.joblib" for name in OFFICIAL}
    missing = [p.name for p in files.values() if not p.exists()]
    if missing:
        raise SystemExit(f"Missing saved models {missing}; not retraining (run main.py only if agreed)")
    models = {name: joblib.load(p) for name, p in files.items()}
    X_test, y_test = df.loc[test_idx, cols], data.get_target(df).loc[test_idx]
    print(f"  held-out rows: {len(X_test):,} (manifest {manifest['n_test_rows']:,}), {len(cols)} features")
    _, aucs = evaluate.plot_test_roc(
        models, X_test, y_test,
        f"Held-out test ROC, {PRIMARY}\nfrozen saved models, {len(X_test):,} test rows, evaluated once",
        f"roc_test_{PRIMARY}.png", rule_point=format_rule_point(df, test_idx))
    results = pd.read_csv(config.RESULTS_DIR / "results.csv")
    rows = []
    for name, auc in aucs.items():
        ref = results.loc[(results["feature_set"] == PRIMARY) & (results["model"] == name), "roc_auc"].iloc[0]
        rows.append({"model": name, "feature_set": PRIMARY, "n_test_rows": len(X_test), "test_roc_auc": auc,
                     "results_csv_roc_auc": ref, "abs_diff": abs(auc - ref)})
    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS_DIR / "roc_test_auc.csv", index=False)
    print(table.round(8).to_string(index=False))
    ok = (table["abs_diff"] <= AUC_TOLERANCE).all()
    print(f"  held-out AUCs {'match' if ok else 'DO NOT match'} reports/results.csv (tolerance {AUC_TOLERANCE})")


def main(sample=None, parts=("cv", "test")):
    t0 = time.time()
    df, groups, train_idx, test_idx = load_split(sample)
    print(f"Rows {len(df):,} | train {len(train_idx):,} | test {len(test_idx):,} | shared URL groups: 0")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if "cv" in parts:
        print("Grouped-CV learning curves and ROC (training rows only) ...")
        run_cv(df, groups, train_idx, sample)
    if "test" in parts:
        if sample:
            print("Held-out ROC skipped: saved_models/ are the full-data models, not this sample's")
        else:
            print("Held-out ROC of the frozen saved models ...")
            run_test(df, test_idx)
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="random sample of N rows (quick check)")
    parser.add_argument("--parts", nargs="+", default=["cv", "test"], choices=["cv", "test"])
    args = parser.parse_args()
    main(args.sample, args.parts)
