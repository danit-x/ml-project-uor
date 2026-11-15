"""Features part: feature engineering and feature selection.

Owner: Madushan S.C  (proposal sections 4.6, 4.7)

Keep the function names. Add your new features inside add_features() and
list every new column name in ENGINEERED_FEATURES.
"""
import pandas as pd

from . import config

# Names of every column created in add_features().
ENGINEERED_FEATURES = [
    "url_digit_ratio",
    "url_special_char_count",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the engineered features added.

    This runs on the whole dataset before the train/test split, so every
    feature must be row-wise: computed from that row alone. No counts,
    vocabularies, means or other statistics across rows (that would leak
    test data into training).

    The two features below are examples only.
    TODO (owner): replace / extend with meaningful URL features, e.g. counts of
    particular characters, ratios, subdomain depth, combinations of existing
    attributes. Justify each one in the notebook.
    """
    df = df.copy()
    url = df["URL"].astype(str)
    length = url.str.len().clip(lower=1)

    df["url_digit_ratio"] = url.str.count(r"\d") / length
    df["url_special_char_count"] = url.str.count(r"[^A-Za-z0-9]")
    return df


def select_features(df: pd.DataFrame, feature_cols: list) -> list:
    """Return the list of features to keep for modelling.

    main.py passes the training rows only (target column included), so
    correlation / importance analysis here never sees the test set. Keep it
    that way when calling it from a notebook: pass df.loc[X_train.index].

    TODO (owner): remove redundant, irrelevant or highly correlated features
    (correlation analysis, importance scores, etc.) and justify the choice.
    Also decide with the group what to do with features that almost reveal
    the label on their own (check URLSimilarityIndex).
    """
    return list(feature_cols)
