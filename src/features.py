"""URL feature engineering and training-only selection (proposal 4.6–4.7)."""
import re
from urllib.parse import urlsplit

import pandas as pd

from . import config

ENGINEERED_FEATURES = [
    "url_path_depth",
    "url_longest_path_segment",
    "url_query_field_count",
    "url_encoded_byte_count",
]

# Keep this in sync with the group exclusion in main.py.
EXCLUDED_FEATURES = frozenset({"URLSimilarityIndex"})
CORRELATION_THRESHOLD = 0.98
ENCODED_BYTE = re.compile(r"%[0-9A-Fa-f]{2}")
SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def _url_features(url: str) -> tuple[int, int, int, int]:
    """Measure URL structure using only the supplied string."""
    # A bare domain/path has no scheme delimiter but still has a hostname.
    try:
        parts = urlsplit(url if url.startswith("//") or SCHEME.match(url) else f"//{url}")
    except ValueError:
        return 0, 0, 0, 0

    segments = [segment for segment in parts.path.split("/") if segment]
    fields = sum(bool(field) for field in parts.query.split("&"))
    encoded_bytes = len(ENCODED_BYTE.findall(parts.path + "?" + parts.query))
    return len(segments), max(map(len, segments), default=0), fields, encoded_bytes


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric features without changing input rows, index or raw URLs."""
    url = df["URL"].astype("string").fillna("").str.strip()
    metrics = pd.DataFrame(
        url.map(_url_features).tolist(), index=df.index,
        columns=ENGINEERED_FEATURES, dtype="int64",
    )
    return df.assign(**metrics)


def select_features(df: pd.DataFrame, feature_cols: list) -> list:
    """Drop excluded, constant and near-duplicate numeric columns on train rows.

    Retain the first column in each redundant pair. Callers put original
    columns first so engineered features must contribute distinct information.
    The rule uses no labels and does not inspect the holdout.
    """
    excluded = EXCLUDED_FEATURES | {config.TARGET} | set(config.TEXT_COLUMNS)
    candidates = [col for col in dict.fromkeys(feature_cols) if col not in excluded]
    numeric = df[candidates].select_dtypes(include="number")
    varying = numeric.loc[:, numeric.nunique() > 1]
    correlation = varying.corr().abs()
    selected = []
    for col in varying.columns:
        if not correlation.loc[col, selected].ge(CORRELATION_THRESHOLD).any():
            selected.append(col)
    if not selected:
        raise ValueError("Feature selection found no eligible varying numeric columns.")
    return selected
