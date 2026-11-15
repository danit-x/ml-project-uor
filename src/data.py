"""Data part: inspection, integration, cleaning, preprocessing, train/test split.

Owner: Madushanka D.M.C  (proposal sections 4.2 - 4.5, 4.8)

The functions below are a working baseline so the other parts can run from
day one. Improve them on your branch; keep the function names and return types.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_raw(path=None) -> pd.DataFrame:
    """Read the raw dataset CSV."""
    path = path or config.RAW_FILE
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the dataset.

    TODO (owner):
    - inspect and report duplicates, missing and inconsistent values
    - if several dataset files exist, check structure and overlap before combining
    - decide how each problem is handled and justify it in the notebook
    """
    df = df.drop_duplicates()
    df = df.dropna(subset=[config.TARGET])
    return df.reset_index(drop=True)


def load_clean_data(path=None) -> pd.DataFrame:
    """Load the raw data and clean it. Everyone calls this to get the dataset."""
    return clean(load_raw(path))


def get_target(df: pd.DataFrame) -> pd.Series:
    """Target as 1 = phishing, 0 = legitimate.

    Phishing is the positive class, so precision, recall and the risk score
    all refer to phishing.
    """
    return (df[config.TARGET] == config.PHISHING_LABEL).astype(int)


def get_original_features(df: pd.DataFrame) -> list:
    """Numeric columns that came with the dataset.

    Excludes the target, the text columns and any engineered features, so it
    is safe to call before or after features.add_features().
    """
    from .features import ENGINEERED_FEATURES

    drop = set(config.TEXT_COLUMNS) | {config.TARGET} | set(ENGINEERED_FEATURES)
    numeric = df.select_dtypes(include="number").columns
    return [c for c in numeric if c not in drop]


def split(df: pd.DataFrame, feature_cols: list):
    """Stratified train/test split, identical for every member.

    Returns X_train, X_test, y_train, y_test.
    Scaling is done inside each model's pipeline, so it is fitted on the
    training data only (no data leakage).
    """
    X = df[feature_cols]
    y = get_target(df)
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
