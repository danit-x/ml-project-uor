"""Run the whole pipeline end to end.

    python main.py                 # uses data/raw/PhiUSIIL_Phishing_URL_Dataset.csv
    python main.py --sample 20000  # quick run on a random sample
    python main.py --data path/to/file.csv

Trains the four models on the original features and on the engineered
features, evaluates them the same way, and writes reports/results.csv
(or reports/results_sample_N.csv for a --sample run).
"""
import argparse

import joblib

from src import config, data, evaluate, features
from src.models import MODELS


def run(data_path=None, sample=None, save_models=True):
    df = data.load_clean_data(data_path)
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=config.RANDOM_STATE).reset_index(drop=True)
    print(f"Rows after cleaning: {len(df):,}")

    original = data.get_original_features(df)
    df = features.add_features(df)

    # Feature selection may learn from the data, so it only sees training rows.
    # data.split() picks the same rows for any column list (same y, same seed).
    candidates = original + features.ENGINEERED_FEATURES
    X_train_all, _, _, _ = data.split(df, candidates)
    engineered = features.select_features(df.loc[X_train_all.index], candidates)

    feature_sets = {"original": original, "engineered": engineered}
    results = {}

    for set_name, cols in feature_sets.items():
        X_train, X_test, y_train, y_test = data.split(df, cols)
        for model_name, module in MODELS.items():
            print(f"Training {model_name} on {set_name} features ({len(cols)}) ...")
            model = module.train(X_train, y_train)
            results[(set_name, model_name)] = evaluate.evaluate(model, X_test, y_test)
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
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="path to the dataset CSV")
    parser.add_argument("--sample", type=int, default=None, help="use a random sample of N rows")
    args = parser.parse_args()
    run(args.data, args.sample)
