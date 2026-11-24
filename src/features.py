"""URL feature engineering, shared feature groups and training-only selection
(proposal 4.6–4.7).

Feature sets agreed by the group on 2026-11-24 (see build_feature_sets()):
- url_base:       the dataset's URL-string features, minus EXCLUDED_FEATURES
                  and COLLECTION_BIAS_FEATURES (no page-content features).
- url_engineered: url_base plus the non-redundant ENGINEERED_FEATURES.
                  PRIMARY set for the main results.
- extended:       url_engineered plus IsHTTPS and the page-content features.
                  Secondary comparison only.
"""
import re
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config

ENGINEERED_FEATURES = [
    "url_path_depth",
    "url_longest_path_segment",
    "url_query_field_count",
    "url_encoded_byte_count",
]

# Numeric columns the PhiUSIIL description computes from the URL string itself.
# Moved here from notebook 04 (phase 2 cell) so every module uses one list.
# URLSimilarityIndex and IsHTTPS are listed because they are URL-derived; the
# feature-set builder removes them (see below).
URL_FEATURES = [
    "URLLength", "DomainLength", "IsDomainIP", "URLSimilarityIndex",
    "CharContinuationRate", "TLDLegitimateProb", "URLCharProb", "TLDLength",
    "NoOfSubDomain", "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
    "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS",
]

# Numeric columns computed from the fetched web page (HTML content, page
# behaviour). Together with URL_FEATURES these are exactly the 50 numeric
# columns of data.get_original_features() (checked by check_feature_groups()).
PAGE_FEATURES = [
    "LineOfCode", "LargestLineLength", "HasTitle", "DomainTitleMatchScore",
    "URLTitleMatchScore", "HasFavicon", "Robots", "IsResponsive",
    "NoOfURLRedirect", "NoOfSelfRedirect", "HasDescription", "NoOfPopup",
    "NoOfiFrame", "HasExternalFormSubmit", "HasSocialNet", "HasSubmitButton",
    "HasHiddenFields", "HasPasswordField", "Bank", "Pay", "Crypto",
    "HasCopyrightInfo", "NoOfImage", "NoOfCSS", "NoOfJS", "NoOfSelfRef",
    "NoOfEmptyRef", "NoOfExternalRef",
]

# Excluded from EVERY feature set (group decision). URLSimilarityIndex is 100
# for every legitimate URL because it measures similarity to the list the
# legitimate URLs were drawn from, so it is a proxy for the label (notebook 04
# phase 2: the one-line rule "USI < 100 means phishing" gives training F1 0.996).
EXCLUDED_FEATURES = frozenset({"URLSimilarityIndex"})

# Kept out of the primary URL-only sets (group decision): IsHTTPS reflects how
# the two classes were collected rather than the URL text, like the page counts.
# It enters only the secondary "extended" set.
COLLECTION_BIAS_FEATURES = ("IsHTTPS",)

FEATURE_SETS = ("url_base", "url_engineered", "extended")
PRIMARY_FEATURE_SET = "url_engineered"

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

    This is a redundancy filter only: |Pearson r| >= CORRELATION_THRESHOLD
    (0.98) means a column is almost a linear copy of one already kept. Passing
    the filter does NOT show that a feature is useful; usefulness is judged by
    grouped training CV (url_base vs url_engineered).
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


def check_feature_groups(original_cols) -> None:
    """Raise ValueError unless URL_FEATURES + PAGE_FEATURES match original_cols.

    Input: data.get_original_features(df). Why: every original column must be
    classed as URL or page content, so that no page feature slips into the
    URL-only sets and no column is silently forgotten.
    """
    original = set(original_cols)
    listed = set(URL_FEATURES) | set(PAGE_FEATURES)
    overlap = set(URL_FEATURES) & set(PAGE_FEATURES)
    missing, unknown = listed - original, original - listed
    if overlap or missing or unknown:
        raise ValueError(
            f"Feature groups do not match the data: in both lists {sorted(overlap)}, "
            f"listed but absent {sorted(missing)}, unclassified {sorted(unknown)}"
        )


def url_base_candidates() -> list:
    """URL_FEATURES minus EXCLUDED_FEATURES and COLLECTION_BIAS_FEATURES (order kept)."""
    drop = EXCLUDED_FEATURES | set(COLLECTION_BIAS_FEATURES)
    return [col for col in URL_FEATURES if col not in drop]


def build_feature_sets(train_df: pd.DataFrame) -> dict:
    """The three agreed feature sets, chosen on TRAINING rows only.

    Input: the training rows (df.loc[X_train.index]) after add_features().
    Never pass the full dataset or test rows: the redundancy filter looks at
    the data. Output: {"url_base": [...], "url_engineered": [...],
    "extended": [...]}, each a list of column names.

    Each set is select_features() over the previous set plus its additions,
    with the previous set first. The greedy keep-first rule then keeps every
    column of the previous set, so the sets are nested and a comparison
    between neighbours changes only the added columns:
    url_base ⊆ url_engineered ⊆ extended. URLSimilarityIndex is in none.
    """
    base = select_features(train_df, url_base_candidates())
    engineered = select_features(train_df, base + ENGINEERED_FEATURES)
    extended = select_features(
        train_df, engineered + list(COLLECTION_BIAS_FEATURES) + PAGE_FEATURES
    )
    sets = {"url_base": base, "url_engineered": engineered, "extended": extended}
    for name, cols in sets.items():
        if EXCLUDED_FEATURES & set(cols):
            raise AssertionError(f"{name} contains an excluded feature")
    return sets


def candidate_columns() -> list:
    """Every column build_feature_sets() may choose from, in its order.

    Pass df[candidate_columns()] to FeatureSetSelector so that it can rebuild
    any of the three sets inside a CV fold.
    """
    return list(dict.fromkeys(
        url_base_candidates() + ENGINEERED_FEATURES
        + list(COLLECTION_BIAS_FEATURES) + PAGE_FEATURES
    ))


class FeatureSetSelector(TransformerMixin, BaseEstimator):
    """build_feature_sets() as a pipeline step, so it is refitted per CV fold.

    fit(X) runs the redundancy filter on the rows it is given (in CV: the
    fold's fitting rows only) and stores the chosen columns in columns_;
    transform(X) returns X[columns_]. Put it first in a Pipeline, e.g.
    make_pipeline(FeatureSetSelector("url_engineered"), logreg.build()),
    and pass X = train_df[candidate_columns()]. Then the validation rows of a
    fold never influence which columns are kept for that fold (the earlier
    protocol chose the columns once on all training rows before CV).
    """

    def __init__(self, feature_set: str = PRIMARY_FEATURE_SET):
        self.feature_set = feature_set

    def fit(self, X, y=None):
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
        self.columns_ = build_feature_sets(X)[self.feature_set]
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        return X[self.columns_]

    def get_feature_names_out(self, input_features=None):
        return list(self.columns_)


def fold_selection_report(train_df: pd.DataFrame, folds, outer_sets: dict = None) -> pd.DataFrame:
    """Does the per-fold redundancy filter pick the same columns as the outer one?

    Inputs: the training rows (after add_features()), the CV folds as
    (fit_positions, validation_positions) from evaluate.cv_folds(), and the
    sets chosen on all training rows (computed if omitted).
    Output: one row per fold and feature set with n_selected, same_as_outer,
    added and removed columns. If every row has same_as_outer True, CV with
    the columns fixed in advance gives exactly the same numbers as CV with
    the filter refitted inside each fold, because the filter's output is the
    only thing it passes on.
    """
    outer_sets = outer_sets or build_feature_sets(train_df)
    rows = []
    for fold, (fit_pos, _) in enumerate(folds):
        fold_sets = build_feature_sets(train_df.iloc[fit_pos])
        for name in FEATURE_SETS:
            fold_cols, outer_cols = fold_sets[name], outer_sets[name]
            rows.append({
                "fold": fold, "feature_set": name, "n_selected": len(fold_cols),
                "same_as_outer": fold_cols == outer_cols,
                "added": sorted(set(fold_cols) - set(outer_cols)),
                "removed": sorted(set(outer_cols) - set(fold_cols)),
            })
    return pd.DataFrame(rows)


def correlation_margin(train_df: pd.DataFrame, feature_sets: dict) -> pd.DataFrame:
    """How close the redundancy filter's decisions are to CORRELATION_THRESHOLD.

    For each set: the highest |r| between two KEPT columns (must be below the
    threshold) and, for each candidate it REMOVED, its highest |r| with a kept
    column (at or above the threshold). Small gaps mean the choice could flip
    on other rows; large gaps mean it is stable. Training rows only.
    """
    rows = []
    previous = []
    additions = {
        "url_base": url_base_candidates(),
        "url_engineered": ENGINEERED_FEATURES,
        "extended": list(COLLECTION_BIAS_FEATURES) + PAGE_FEATURES,
    }
    for name in FEATURE_SETS:
        kept = feature_sets[name]
        candidates = list(dict.fromkeys(previous + additions[name]))
        numeric = train_df[candidates].select_dtypes(include="number")
        varying = [c for c in numeric.columns if numeric[c].nunique() > 1]
        corr = train_df[varying].corr().abs()
        kept_corr = corr.loc[kept, kept].to_numpy(copy=True)
        np.fill_diagonal(kept_corr, np.nan)
        rows.append({"feature_set": name, "kind": "max |r| between kept columns",
                     "column": "", "abs_r": float(np.nanmax(kept_corr))})
        for col in varying:
            if col not in kept:
                partner = corr.loc[col, kept].idxmax()
                rows.append({"feature_set": name, "kind": f"removed (closest kept: {partner})",
                             "column": col, "abs_r": float(corr.loc[col, partner])})
        previous = kept
    return pd.DataFrame(rows)
