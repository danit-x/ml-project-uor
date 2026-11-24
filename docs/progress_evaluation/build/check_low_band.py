"""Read-only check: how the Random Forest's Low-band phishing URLs relate to LR's false negatives.

    python docs/progress_evaluation/build/check_low_band.py     # about 2 minutes (grouped split)

Loads the saved full-data models (saved_models/, written by `python main.py`), scores the held-out
test rows and prints counts only (no URLs). Writes nothing. Not part of the ML pipeline.
"""
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import joblib  # noqa: E402

from src import config, data, features, risk  # noqa: E402


def main():
    df = features.add_features(data.load_clean_data())
    _, test_idx = data.split_index(df)
    manifest = json.loads((config.MODELS_DIR / "manifest.json").read_text(encoding="utf-8"))
    X = df.loc[test_idx, manifest["feature_sets"][features.PRIMARY_FEATURE_SET]]
    y = data.get_target(df).loc[test_idx].to_numpy()
    lr = joblib.load(config.MODELS_DIR / f"logistic_regression_{features.PRIMARY_FEATURE_SET}.joblib")
    cal = joblib.load(config.MODELS_DIR / f"random_forest_calibrated_{features.PRIMARY_FEATURE_SET}.joblib")
    scores = risk.risk_score(cal, X)
    lr_fn = set(test_idx[(lr.predict(X) == 0) & (y == 1)])
    low_phish = set(test_idx[(scores < risk.LOW_MAX) & (y == 1)])
    bare = data.bare_homepage_mask(df.loc[test_idx, "URL"])
    print(f"LR false negatives: {len(lr_fn)} | Low-band phishing (calibrated RF): {len(low_phish)} | "
          f"in both: {len(lr_fn & low_phish)}")
    print(f"Low-band phishing that are bare https://www homepages: {int(bare.loc[list(low_phish)].sum())}")
    print(f"High-band legitimate: {int(((scores >= risk.HIGH_MIN) & (y == 0)).sum())}")


if __name__ == "__main__":
    main()
