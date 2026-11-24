"""Run the whole pipeline end to end.

    python main.py                 # uses data/raw/PhiUSIIL_Phishing_URL_Dataset.csv
    python main.py --sample 20000  # quick run on a random sample
    python main.py --data path/to/file.csv

Protocol (group decisions of 2026-11-24):
1. Near-duplicate URLs share a group (data.url_group_key); the train/test split
   is group-aware and stratified (data.split), and so is every training CV.
2. Three nested feature sets are chosen on TRAINING rows only
   (features.build_feature_sets): url_base, url_engineered (PRIMARY) and
   extended (secondary: + IsHTTPS + page content). URLSimilarityIndex is in none.
3. Development: grouped 5-fold CV on the training rows for the three main
   models (Logistic Regression, Decision Tree, Random Forest), identical folds
   for all, written to reports/cv_results.csv. The feature-set filter is a
   pipeline step (features.FeatureSetSelector), so it is refitted on each
   fold's fitting rows and the validation rows never influence it. The test
   set plays no part. Hyperparameters were chosen beforehand with
   tune_models.py (same rows, folds and features); main.py does no search.
4. Final: every model in MODELS is fitted with its frozen build() on the
   training rows and evaluated once on the held-out rows, written to
   reports/results.csv with a Brier score and a protocol column:
   "official" = the three main models on url_engineered, "secondary" = the
   three main models on url_base / extended, "provisional" = SVM (its C was
   tuned with the earlier ungrouped CV on a 2,000-row sample, so it is
   reported separately, never in the official comparison).
   reports/official_comparison.csv joins the official rows' grouped CV and
   test metrics. reports/format_baseline.csv scores the fixed rule
   "not a bare https://www homepage -> phishing" (data.bare_homepage_mask) on
   the same CV folds and test rows: a reference, not a model.
   reports/format_controlled.csv splits the held-out rows into bare https://www
   homepages (the format cue is absent there) and all other URLs, for the
   format rule and the three official models (diagnostic only).
   A --sample N run writes the *_sample_N.csv files instead.
Then calibrates the Random Forest (reports/risk_bands.csv) and writes risk
scores and SHAP explanations for the first --explain-rows test URLs (see
risk_and_explain). saved_models/manifest.json records which run wrote the
saved models; demo.py uses them to score and explain held-out test URLs.
"""
import argparse

import joblib
import pandas as pd
from sklearn.pipeline import make_pipeline

from src import config, data, evaluate, features, risk
from src.models import MODELS, forest

# Kept for notebook 04, which imports these names. The single source of the
# exclusion is now features.EXCLUDED_FEATURES (URLSimilarityIndex, label proxy).
DROPPED_FEATURES = sorted(features.EXCLUDED_FEATURES)

# The three models of the progress evaluation; they get the grouped CV table.
MAIN_MODELS = ("Logistic Regression", "Decision Tree", "Random Forest")

# Metrics at or above this are reported with a warning, not as a result (CLAUDE.md).
NEAR_PERFECT = 0.99


def without_dropped(cols):
    """cols minus DROPPED_FEATURES, order kept."""
    return [c for c in cols if c not in DROPPED_FEATURES]


def _out_name(stem, sample):
    """A sample run must never overwrite the full-dataset results."""
    return f"{stem}_sample_{sample}.csv" if sample else f"{stem}.csv"


def _role(set_name):
    return "primary" if set_name == features.PRIMARY_FEATURE_SET else "secondary"


def _protocol(set_name, model_name):
    """official / secondary / provisional label of one results row (see module docstring)."""
    if model_name not in MAIN_MODELS:
        return "provisional"
    return "official" if set_name == features.PRIMARY_FEATURE_SET else "secondary"


def run(data_path=None, sample=None, save_models=True, explain_rows=500, cv=True):
    df = data.load_clean_data(data_path)
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=config.RANDOM_STATE).reset_index(drop=True)
    print(f"Rows after cleaning: {len(df):,}")

    features.check_feature_groups(data.get_original_features(df))
    df = features.add_features(df)

    # 1. URL groups and the group-aware split (same rows for every feature set).
    groups = data.group_ids(df)
    conflicts = data.conflicting_label_groups(df)
    print(f"URL groups: {groups.nunique():,} ({int((groups.value_counts() > 1).sum()):,} with "
          f"several rows); label-conflict groups kept as label noise: "
          f"{conflicts['url_group_key'].nunique()} ({len(conflicts)} rows)")
    train_idx, test_idx = data.split_index(df)
    overlap = set(groups.loc[train_idx]) & set(groups.loc[test_idx])
    if overlap:
        raise AssertionError(f"{len(overlap)} URL groups are in both train and test")
    y = data.get_target(df)
    print(f"Train rows {len(train_idx):,} | test rows {len(test_idx):,} "
          f"({len(test_idx) / len(df):.1%}) | phishing share train "
          f"{y.loc[train_idx].mean():.4f}, test {y.loc[test_idx].mean():.4f} | shared groups: 0")

    # 2. Feature sets, chosen on training rows only.
    feature_sets = features.build_feature_sets(df.loc[train_idx])
    for set_name, cols in feature_sets.items():
        print(f"  {set_name} ({_role(set_name)}): {len(cols)} features")
    dropped = [c for c in features.ENGINEERED_FEATURES if c not in feature_sets["url_engineered"]]
    print(f"  engineered features removed as redundant (|r| >= {features.CORRELATION_THRESHOLD}): {dropped}")

    # 3. Development: grouped CV on training rows, identical folds for every model.
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_table = None
    if cv:
        cv_table = cv_comparison(df, feature_sets, train_idx, groups.loc[train_idx])
        cv_name = _out_name("cv_results", sample)
        cv_table.to_csv(config.RESULTS_DIR / cv_name, index=False)
        print(f"Saved {config.RESULTS_DIR / cv_name} (training CV, not test results)")
        show = ["feature_set", "model"] + [f"{m}_mean" for m in evaluate.CV_METRICS] + ["f1_std"]
        print(cv_table[show].round(4).to_string(index=False))
        print()

    # 4. Final held-out evaluation with the frozen build() configurations.
    results = {}
    fitted = {}
    for set_name, cols in feature_sets.items():
        X_train, X_test, y_train, y_test = data.split(df, cols)
        for model_name, module in MODELS.items():
            print(f"Training {model_name} on {set_name} features ({len(cols)}) ...")
            model = module.train(X_train, y_train)
            if set_name == features.PRIMARY_FEATURE_SET:
                # Only the primary set's models are used again below; keeping the
                # full-data forests of every set in memory would need several GB.
                fitted[(set_name, model_name)] = model
            # Brier score = how good the phishing probability is as a risk score (4.11).
            brier = risk.score_metrics(y_test, risk.risk_score(model, X_test))["brier"]
            results[(set_name, model_name)] = (
                evaluate.evaluate(model, X_test, y_test)
                | {"brier": brier, "n_features": len(cols), "role": _role(set_name),
                   "protocol": _protocol(set_name, model_name)}
            )
            if save_models:
                config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
                name = f"{model_name.lower().replace(' ', '_')}_{set_name}.joblib"
                joblib.dump(model, config.MODELS_DIR / name)

    table = evaluate.compare(results)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_name = _out_name("results", sample)
    table.to_csv(config.RESULTS_DIR / out_name, index=False)
    print(f"Saved {config.RESULTS_DIR / out_name} (held-out test, frozen configurations)")
    print()
    print(table.round(4).to_string(index=False))
    _warn_near_perfect(table)

    official = official_comparison(table, cv_table)
    official_name = _out_name("official_comparison", sample)
    official.to_csv(config.RESULTS_DIR / official_name, index=False)
    print(f"\nOFFICIAL comparison ({features.PRIMARY_FEATURE_SET}, grouped split and folds) "
          f"-> {config.RESULTS_DIR / official_name}")
    print(official.round(4).to_string(index=False))
    svm_rows = table.loc[table["protocol"] == "provisional"]
    if len(svm_rows):
        print("\nSVM (PROVISIONAL, not part of the official comparison: C from the earlier "
              "ungrouped 2,000-row tuning):")
        print(svm_rows[["feature_set", "model", "accuracy", "precision", "recall", "f1", "roc_auc"]]
              .round(4).to_string(index=False))

    baseline = format_baseline(df, train_idx, test_idx, groups.loc[train_idx])
    baseline_name = _out_name("format_baseline", sample)
    baseline.to_csv(config.RESULTS_DIR / baseline_name, index=False)
    print("\nREFERENCE (not a model): rule 'not a bare https://www homepage -> phishing' "
          f"-> {config.RESULTS_DIR / baseline_name}")
    print(baseline.round(4).to_string(index=False))

    controlled = format_controlled(df, test_idx, feature_sets, fitted)
    controlled_name = _out_name("format_controlled", sample)
    controlled.to_csv(config.RESULTS_DIR / controlled_name, index=False)
    print("\nFORMAT-CONTROLLED test breakdown (diagnostic only, frozen models) "
          f"-> {config.RESULTS_DIR / controlled_name}")
    print(controlled.round(4).to_string(index=False))

    if save_models:
        _write_manifest(sample, len(train_idx), len(test_idx), feature_sets)

    if explain_rows:
        primary = features.PRIMARY_FEATURE_SET
        risk_and_explain(df, feature_sets[primary], fitted[(primary, "Random Forest")],
                         sample, explain_rows, groups=groups, save_model=save_models)
    return table


MANIFEST = "manifest.json"


def _write_manifest(sample, n_train, n_test, feature_sets):
    """Record which run produced saved_models/ (demo.py refuses a mismatched dataset)."""
    import json
    from datetime import datetime

    from src.models import logreg, tree

    info = {
        "written": datetime.now().isoformat(timespec="seconds"),
        "sample": sample, "n_train_rows": int(n_train), "n_test_rows": int(n_test),
        "feature_sets": feature_sets, "primary_feature_set": features.PRIMARY_FEATURE_SET,
        "logreg": {"C": logreg.BEST_C, "l1_ratio": logreg.BEST_L1_RATIO, "solver": logreg.BEST_SOLVER},
        "tree": {"max_depth": tree.BEST_MAX_DEPTH, "min_samples_leaf": tree.BEST_MIN_SAMPLES_LEAF,
                 "class_weight": tree.BEST_CLASS_WEIGHT},
        "forest": forest.BEST_PARAMS,
    }
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (config.MODELS_DIR / MANIFEST).write_text(json.dumps(info, indent=2, default=str))


def format_controlled(df, test_idx, feature_sets, fitted) -> pd.DataFrame:
    """Held-out results split by URL format: bare https://www homepages vs all other URLs.

    In PhiUSIIL every legitimate URL is a bare https://www homepage, so on
    "other URLs" every row is phishing and the format alone decides. The
    bare-homepage subset is where the format cue is ABSENT: all legitimate
    URLs plus the few phishing URLs written as bare https://www homepages.
    Detecting those phishing rows needs more than the format, so the recall
    there is the most direct evidence of what a model learned beyond the
    collection format. Diagnostic only: computed with the frozen models after
    every choice was made, never used for selection. One row per (subset,
    model) for the official models and the fixed format rule.
    """
    y = data.get_target(df).loc[test_idx]
    bare = data.bare_homepage_mask(df.loc[test_idx, "URL"])
    primary = features.PRIMARY_FEATURE_SET
    X = df.loc[test_idx, feature_sets[primary]]
    preds = {"format rule (reference)": (~bare).astype(int).to_numpy()}
    for name in MAIN_MODELS:
        preds[name] = fitted[(primary, name)].predict(X)
    rows = []
    for subset, mask in (("bare https://www homepage", bare.to_numpy()),
                         ("other URLs", ~bare.to_numpy())):
        yt = y.to_numpy()[mask]
        for name, p in preds.items():
            pt = p[mask]
            tp = int(((pt == 1) & (yt == 1)).sum())
            fn = int(((pt == 0) & (yt == 1)).sum())
            fp = int(((pt == 1) & (yt == 0)).sum())
            tn = int(((pt == 0) & (yt == 0)).sum())
            rows.append({
                "subset": subset, "model": name, "n_rows": int(mask.sum()),
                "n_phishing": int(yt.sum()), "n_legitimate": int((1 - yt).sum()),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "phishing_recall": tp / (tp + fn) if tp + fn else float("nan"),
                "false_positive_rate": fp / (fp + tn) if fp + tn else float("nan"),
                "accuracy": (tp + tn) / len(yt) if len(yt) else float("nan"),
            })
    return pd.DataFrame(rows)


def cv_comparison(df, feature_sets, train_idx, train_groups) -> pd.DataFrame:
    """Grouped training CV of the MAIN_MODELS on every feature set.

    All rows are training rows; every model and feature set gets the same
    folds (evaluate.cv_folds caches them for these rows and groups).
    Each model is wrapped as FeatureSetSelector(set) -> build(), and X holds
    every candidate column, so the redundancy filter is refitted on each
    fold's fitting rows only. fold_selection_report() then says whether any
    fold chose different columns from the filter fitted on all training rows
    (which is what the final models and tune_models.py use).
    """
    y_train = data.get_target(df).loc[train_idx]
    X_candidates = df.loc[train_idx, features.candidate_columns()]
    folds = evaluate.cv_folds(X_candidates, y_train, train_groups)
    report = features.fold_selection_report(df.loc[train_idx], folds, feature_sets)
    differ = report.loc[~report["same_as_outer"]]
    print(f"Feature selection refitted inside each CV fold: {len(report) - len(differ)} of "
          f"{len(report)} (fold, set) pairs chose the same columns as the training-rows filter")
    if len(differ):
        print(differ.to_string(index=False))
    frames = []
    for set_name, cols in feature_sets.items():
        print(f"Grouped CV on {set_name} ({len(cols)} features, selected per fold) ...")
        frame = evaluate.compare_training_models(
            {set_name: (X_candidates, y_train)},
            {name: make_pipeline(features.FeatureSetSelector(set_name), MODELS[name].build())
             for name in MAIN_MODELS},
            groups=train_groups,
            # LR and DT fit on one core, so their folds run in parallel; the forest uses all cores itself.
            n_jobs={"Logistic Regression": -1, "Decision Tree": -1},
        )
        same = bool(report.loc[report["feature_set"] == set_name, "same_as_outer"].all())
        frames.append(frame.assign(
            n_features=len(cols), role=_role(set_name),
            selection="refitted per fold", fold_sets_equal_outer=same,
        ))
    return pd.concat(frames, ignore_index=True)


def official_comparison(table, cv_table=None) -> pd.DataFrame:
    """The official three-model comparison: MAIN_MODELS on the primary set only.

    One row per model: grouped training-CV mean and std of every CV metric
    (when cv_table is given) next to the held-out test metrics and confusion
    counts. SVM and the secondary feature sets are not included.
    """
    primary = features.PRIMARY_FEATURE_SET
    test = table.loc[(table["feature_set"] == primary) & table["model"].isin(MAIN_MODELS)]
    keep = ["model", "n_features", "accuracy", "precision", "recall", "f1", "roc_auc",
            "tn", "fp", "fn", "tp", "brier"]
    out = test[keep].rename(columns={m: f"test_{m}" for m in keep[2:]}).set_index("model")
    if cv_table is not None:
        cv_cols = [f"{m}_{s}" for m in evaluate.CV_METRICS for s in ("mean", "std")]
        cv = cv_table.loc[cv_table["feature_set"] == primary].set_index("model")[cv_cols]
        out = cv.add_prefix("cv_").join(out, how="right")
    out = out.reindex(list(MAIN_MODELS))
    return out.reset_index().assign(feature_set=primary)


def format_baseline(df, train_idx, test_idx, train_groups) -> pd.DataFrame:
    """Score the fixed rule 'not a bare https://www homepage -> phishing'.

    The rule (data.bare_homepage_mask) is fixed in advance and fitted on
    nothing. It is scored on the same grouped CV validation folds and on the
    held-out rows with evaluate's metrics, to show how much of a URL-only
    score the dataset's URL format alone explains. Not a model.
    """
    y = data.get_target(df)
    rule = (~data.bare_homepage_mask(df["URL"])).astype(int)
    folds = evaluate.cv_folds(df.loc[train_idx, features.candidate_columns()],
                              y.loc[train_idx], train_groups)
    cv = evaluate.rule_cv_metrics(y.loc[train_idx], rule.loc[train_idx], folds)
    test = evaluate.prediction_metrics(y.loc[test_idx], rule.loc[test_idx], rule.loc[test_idx])
    row = {"reference": "not bare https://www homepage -> phishing"}
    row |= {f"cv_{k}": v for k, v in cv.items()}
    row |= {f"test_{k}": v for k, v in test.items()}
    row["share_legitimate_bare_train"] = float(
        data.bare_homepage_mask(df.loc[train_idx, "URL"])[y.loc[train_idx] == 0].mean())
    row["share_phishing_bare_train"] = float(
        data.bare_homepage_mask(df.loc[train_idx, "URL"])[y.loc[train_idx] == 1].mean())
    return pd.DataFrame([row])


def _warn_near_perfect(table):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    high = table.loc[(table[metrics].astype(float) >= NEAR_PERFECT).any(axis=1), ["feature_set", "model"]]
    if len(high):
        print(f"\nWARNING: {len(high)} rows have a metric >= {NEAR_PERFECT}. Treat this as a "
              "prompt to check for leakage/collection bias, not as a headline result.")


def risk_and_explain(df, cols, forest_model, sample=None, n_rows=500, groups=None, save_model=True):
    """Risk scores and SHAP explanations for the Random Forest on the primary feature set (4.11, 4.13).

    The served risk score comes from the forest calibrated on training rows only
    (risk.calibrate, method risk.CALIBRATION). Explanations are SHAP values of
    the uncalibrated forest; the calibration is monotonic, so it keeps the same
    feature directions. Only the first n_rows test URLs are explained (SHAP is
    about 30 s per 1,000 rows), which also keeps the output CSV small.

    Writes: <prefix>shap_importance.csv, <prefix>risk_scores.csv (one row per
    explained test URL: URL, true label, risk score, level, top 3 features) and
    figures/shap_summary<suffix>.png. Sample runs use the gitignored
    results_sample_N_ prefix so they never overwrite the full-data files.
    """
    from src import explain

    X_train, X_test, y_train, y_test = data.split(df, cols)
    groups = data.group_ids(df) if groups is None else groups
    print(f"Calibrating Random Forest ({risk.CALIBRATION}) on training rows (grouped folds) ...")
    risk_model = risk.calibrate(forest.build(), X_train, y_train, groups=groups.loc[X_train.index])
    test_scores = risk.risk_score(risk_model, X_test)
    print("Calibrated forest, test set:",
          {k: round(v, 5) for k, v in risk.score_metrics(y_test, test_scores).items()})
    print(risk.band_table(y_test, test_scores).round(4).to_string(index=False))
    bands = risk.band_table(y_test, test_scores)
    bands.to_csv(config.RESULTS_DIR / _out_name("risk_bands", sample), index=False)
    if save_model:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(risk_model, config.MODELS_DIR / f"random_forest_calibrated_{features.PRIMARY_FEATURE_SET}.joblib")

    X_explain = X_test.iloc[:n_rows]
    print(f"SHAP for {len(X_explain)} test URLs ...")
    contrib = explain.contributions(forest_model, X_explain)
    importance = explain.rank_importance(X_explain.columns, contrib.values, "mean_abs_shap")
    per_url = explain.explain_predictions(
        forest_model, X_explain, test_scores[:n_rows], top=3, contrib=contrib
    )
    per_url.insert(0, "URL", df.loc[X_explain.index, "URL"].values)
    per_url.insert(1, "is_phishing", y_test.iloc[:n_rows].values)

    prefix = f"results_sample_{sample}_" if sample else ""
    suffix = f"_sample_{sample}" if sample else ""
    importance.to_csv(config.RESULTS_DIR / f"{prefix}shap_importance.csv", index=False)
    per_url.to_csv(config.RESULTS_DIR / f"{prefix}risk_scores.csv", index=False)
    explain.summary_plot(forest_model, X_explain, f"shap_summary{suffix}.png", values=contrib.values)
    print(f"Saved {prefix}shap_importance.csv, {prefix}risk_scores.csv, figures/shap_summary{suffix}.png")
    print(importance.head(10).round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="path to the dataset CSV")
    parser.add_argument("--sample", type=int, default=None, help="use a random sample of N rows")
    parser.add_argument("--explain-rows", type=int, default=500,
                        help="test URLs to explain with SHAP after training (0 = skip risk scoring and SHAP)")
    parser.add_argument("--skip-cv", action="store_true",
                        help="skip the grouped training-CV comparison of the three main models")
    args = parser.parse_args()
    run(args.data, args.sample, explain_rows=args.explain_rows, cv=not args.skip_cv)
