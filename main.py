"""Run the whole pipeline end to end.

    python main.py                 # uses data/raw/PhiUSIIL_Phishing_URL_Dataset.csv
    python main.py --sample 20000  # quick run on a random sample
    python main.py --data path/to/file.csv

Trains the four models on the original features and on the engineered
features, evaluates them the same way, and writes reports/results.csv
(or reports/results_sample_N.csv for a --sample run), with a Brier score per
model. Then calibrates the Random Forest and writes risk scores and SHAP
explanations for the first --explain-rows test URLs (see risk_and_explain).
"""
import argparse

import joblib

from src import config, data, evaluate, features, risk
from src.models import MODELS, forest

# Group decision: drop features that leak the label.
# URLSimilarityIndex is 100 for every legitimate URL because it measures
# similarity to the very list the legitimate URLs were taken from (nb04 phase 2:
# the rule "USI < 100 means phishing" alone gives training F1 0.996, 0 false
# positives). IsHTTPS and the page-content counts are collection bias, not
# leakage: they are kept, and nb04 also reports a URL-only result next to the
# full one. Applied here to both feature sets until features.select_features()
# (owner Madushan S.C) does the drop itself; then this list can go.
DROPPED_FEATURES = ["URLSimilarityIndex"]


def without_dropped(cols):
    """cols minus DROPPED_FEATURES, order kept."""
    return [c for c in cols if c not in DROPPED_FEATURES]


def run(data_path=None, sample=None, save_models=True, explain_rows=500):
    df = data.load_clean_data(data_path)
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=config.RANDOM_STATE).reset_index(drop=True)
    print(f"Rows after cleaning: {len(df):,}")

    original = without_dropped(data.get_original_features(df))
    df = features.add_features(df)

    # Feature selection may learn from the data, so it only sees training rows.
    # data.split() picks the same rows for any column list (same y, same seed).
    candidates = original + features.ENGINEERED_FEATURES
    X_train_all, _, _, _ = data.split(df, candidates)
    engineered = without_dropped(features.select_features(df.loc[X_train_all.index], candidates))

    feature_sets = {"original": original, "engineered": engineered}
    results = {}
    fitted = {}

    for set_name, cols in feature_sets.items():
        X_train, X_test, y_train, y_test = data.split(df, cols)
        for model_name, module in MODELS.items():
            print(f"Training {model_name} on {set_name} features ({len(cols)}) ...")
            model = module.train(X_train, y_train)
            fitted[(set_name, model_name)] = model
            # Brier score = how good the phishing probability is as a risk score (4.11).
            brier = risk.score_metrics(y_test, risk.risk_score(model, X_test))["brier"]
            results[(set_name, model_name)] = evaluate.evaluate(model, X_test, y_test) | {"brier": brier}
            if save_models:
                config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
                name = f"{model_name.lower().replace(' ', '_')}_{set_name}.joblib"
                joblib.dump(model, config.MODELS_DIR / name)

    table = evaluate.compare(results)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # A sample run must never overwrite the full-dataset results.
    out_name = f"results_sample_{sample}.csv" if sample else "results.csv"
    table.to_csv(config.RESULTS_DIR / out_name, index=False)
    print(f"Saved {config.RESULTS_DIR / out_name}")
    print()
    print(table.round(4).to_string(index=False))

    if explain_rows:
        risk_and_explain(df, original, fitted[("original", "Random Forest")], sample, explain_rows)
    return table


def risk_and_explain(df, cols, forest_model, sample=None, n_rows=500):
    """Risk scores and SHAP explanations for the Random Forest on original features (4.11, 4.13).

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
    print(f"Calibrating Random Forest ({risk.CALIBRATION}) on training rows ...")
    risk_model = risk.calibrate(forest.build(), X_train, y_train)
    test_scores = risk.risk_score(risk_model, X_test)
    print("Calibrated forest, test set:",
          {k: round(v, 5) for k, v in risk.score_metrics(y_test, test_scores).items()})
    print(risk.band_table(y_test, test_scores).round(4).to_string(index=False))

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
    args = parser.parse_args()
    run(args.data, args.sample, explain_rows=args.explain_rows)
