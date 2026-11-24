"""Demonstration: risk score, risk band and explanation for held-out test URLs.

    python demo.py                       # three examples from the held-out test set
    python demo.py --url "<test URL>"    # one URL, which must be a held-out test URL
    python demo.py --n 5 --top 5         # more examples / more features per URL

Needs the models written by `python main.py` (saved_models/, see manifest.json).
The demo uses the same data, cleaning, URL groups and group-aware split as
main.py, so the URLs shown were never used to fit, tune or calibrate a model.

Scope (deliberately narrow): only URLs from the PhiUSIIL held-out test rows are
scored. Several model inputs (for example TLDLegitimateProb and URLCharProb)
were computed by the dataset authors' feature extractor, which is not part of
this project, so the demo cannot score an arbitrary new URL, and the scores say
nothing about URLs collected in a different way (see the format-bias notes in
reports/format_baseline.csv and reports/format_controlled.csv).

Output per URL: calibrated Random Forest risk score (0-1), risk band
(risk.LOW_MAX / risk.HIGH_MIN), predicted class, true label, and the top SHAP
contributions of the forest (positive = pushes towards phishing).
"""
import argparse
import json
import sys

import joblib
import pandas as pd

from src import config, data, explain, features, risk

PRIMARY = features.PRIMARY_FEATURE_SET


def load_models():
    """(manifest, forest, calibrated forest) from saved_models/, with clear errors."""
    manifest_path = config.MODELS_DIR / "manifest.json"
    forest_path = config.MODELS_DIR / f"random_forest_{PRIMARY}.joblib"
    calibrated_path = config.MODELS_DIR / f"random_forest_calibrated_{PRIMARY}.joblib"
    missing = [p.name for p in (manifest_path, forest_path, calibrated_path) if not p.exists()]
    if missing:
        sys.exit(f"Missing {missing} in {config.MODELS_DIR}. Run `python main.py` first.")
    manifest = json.loads(manifest_path.read_text())
    return manifest, joblib.load(forest_path), joblib.load(calibrated_path)


def test_rows(manifest):
    """Clean data (sampled like the run that saved the models) and its held-out test rows."""
    df = data.load_clean_data()
    sample = manifest.get("sample")
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=config.RANDOM_STATE).reset_index(drop=True)
    df = features.add_features(df)
    _, test_idx = data.split_index(df)
    if len(test_idx) != manifest["n_test_rows"]:
        sys.exit("The data do not match the run that saved the models; rerun `python main.py`.")
    return df, test_idx


def pick_examples(df, test_idx, scores, n):
    """Clear phishing, clear legitimate, then the most uncertain test URLs (deterministic)."""
    y = data.get_target(df).loc[test_idx]
    s = pd.Series(scores, index=test_idx)
    chosen = [s[y == 1].idxmax(), s[y == 0].idxmin()]
    for i in (s - 0.5).abs().sort_values().index:
        if len(chosen) >= n:
            break
        if i not in chosen:
            chosen.append(i)
    return chosen[:n]


def show(df, idx, cols, forest_model, score, top):
    url = df.loc[idx, "URL"]
    truth = "phishing" if data.get_target(df).loc[idx] == 1 else "legitimate"
    table = explain.explain_one(forest_model, df.loc[[idx], cols], score, top=top)
    predicted = "phishing" if score >= 0.5 else "legitimate"
    print(f"\nURL: {url}")
    print(f"  risk score {score:.3f}  ->  band {table.attrs['risk_level']}  |  predicted {predicted}  "
          f"|  true label {truth}")
    print(f"  bare https://www homepage (the dataset's legitimate-URL format): "
          f"{bool(data.bare_homepage_mask(df.loc[[idx], 'URL']).iloc[0])}")
    print("  top contributions (SHAP, phishing probability units; + pushes towards phishing):")
    print(table.round(4).to_string(index=False))


def main(url=None, n=3, top=3):
    manifest, forest_model, calibrated = load_models()
    run = "full data" if not manifest.get("sample") else f"--sample {manifest['sample']}"
    print(f"Models from main.py run of {manifest['written']} ({run}); "
          f"{manifest['n_test_rows']:,} held-out test URLs; feature set {PRIMARY}.")
    print(f"Bands: Low < {risk.LOW_MAX} <= Medium < {risk.HIGH_MIN} <= High "
          f"(calibration: {risk.CALIBRATION}, fitted on training rows only).")
    df, test_idx = test_rows(manifest)
    cols = manifest["feature_sets"][PRIMARY]
    if url is not None:
        hits = df.index[df["URL"] == url]
        if len(hits) == 0 or hits[0] not in set(test_idx):
            sys.exit("This URL is not a held-out PhiUSIIL test URL. The demo only scores held-out "
                     "test URLs (see the module docstring for why arbitrary URLs are not supported).")
        rows = [hits[0]]
    else:
        scores_all = risk.risk_score(calibrated, df.loc[test_idx, cols])
        rows = pick_examples(df, test_idx, scores_all, n)
    scores = risk.risk_score(calibrated, df.loc[rows, cols])
    for idx, score in zip(rows, scores):
        show(df, idx, cols, forest_model, float(score), top)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None, help="a held-out test URL to explain")
    parser.add_argument("--n", type=int, default=3, help="number of example URLs (default 3)")
    parser.add_argument("--top", type=int, default=3, help="features listed per URL (default 3)")
    args = parser.parse_args()
    main(args.url, args.n, args.top)
