"""Data part: inspection, integration, cleaning, preprocessing, train/test split.

Owner: Madushanka D.M.C  (proposal sections 4.2 - 4.5, 4.8)

Keep the function names and return types: main.py and the other members'
notebooks call load_raw(), clean(), load_clean_data(), get_target(),
get_original_features() and split().
"""
from urllib.parse import urlsplit

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

# Raw label values the dataset is allowed to contain (UCI: 1 = legitimate, 0 = phishing).
VALID_LABELS = (0, 1)


def load_raw(path=None) -> pd.DataFrame:
    """Read the raw dataset CSV."""
    path = path or config.RAW_FILE
    return pd.read_csv(path)


def check_target(df: pd.DataFrame) -> None:
    """Raise ValueError unless the target column holds only the labels 0 and 1.

    Input: a DataFrame whose missing-target rows have already been removed.
    Why: get_target() maps "== PHISHING_LABEL" to 1 and everything else to 0,
    so an unexpected value (for example 2, "1" or -1) would silently become
    "legitimate". Failing loudly here is safer than training on a wrong label.
    """
    target = df[config.TARGET]
    if target.isna().any():
        raise ValueError(
            f"'{config.TARGET}' still has {int(target.isna().sum())} missing values"
        )
    # A string such as "1" is not equal to 1, so it is caught as invalid too.
    invalid = target[~target.isin(VALID_LABELS)]
    if len(invalid):
        found = sorted(map(repr, invalid.unique()))[:10]
        raise ValueError(
            f"'{config.TARGET}' must contain only {VALID_LABELS}; found "
            f"{len(invalid)} rows with other values, e.g. {', '.join(found)}"
        )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the dataset. The input DataFrame is not modified.

    Steps (the notebook 01 reports the evidence behind each one):
    1. Drop fully identical rows.
    2. Drop repeated URLs, keeping the first occurrence in file order. This is
       deterministic, and it stops the same URL from landing in both train and
       test. Note: in PhiUSIIL each of the 425 repeated URLs appears twice with
       the same label but with different feature values (for example
       URLLength), so "keep first" is a convention, not a merge of identical rows.
    3. Drop rows with a missing target.
    4. Validate that the remaining target values are only 0 and 1 (check_target).

    Near-duplicate URLs (same URL apart from letter case or a trailing slash)
    are deliberately kept. They are reported in notebook 01 for a group
    decision, because changing this function changes every member's data.

    Do not impute with statistics of the whole dataset here (that leaks test
    data); put imputers in the model pipelines instead.
    """
    out = df.drop_duplicates()
    # The same URL in both train and test inflates the scores, so keep one copy.
    out = out.drop_duplicates(subset=["URL"], keep="first")
    out = out.dropna(subset=[config.TARGET])
    check_target(out)
    return out.reset_index(drop=True)


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


# ---------------------------------------------------------------------------
# Inspection helpers (read-only; used by notebook 01, not by clean()).
# ---------------------------------------------------------------------------

def duplicate_url_report(df: pd.DataFrame) -> pd.DataFrame:
    """Describe every URL that appears more than once, one row per URL.

    Input: the raw dataset (before clean()). Output columns:
    URL, n_rows, n_labels (distinct labels), n_features_differing (how many
    non-text columns have more than one value within the group) and
    features_differing (their names). Use it to show what clean()'s
    keep-first rule throws away. The input is not modified.
    """
    repeated = df[df["URL"].duplicated(keep=False)]
    if repeated.empty:
        return pd.DataFrame(
            columns=["URL", "n_rows", "n_labels", "n_features_differing", "features_differing"]
        )
    value_cols = [c for c in df.columns if c not in config.TEXT_COLUMNS and c != config.TARGET]
    groups = repeated.groupby("URL", sort=True)
    differs = groups[value_cols].nunique(dropna=False) > 1
    report = pd.DataFrame({
        "n_rows": groups.size(),
        "n_labels": groups[config.TARGET].nunique(dropna=False),
        "n_features_differing": differs.sum(axis=1),
        "features_differing": differs.apply(lambda r: list(r.index[r]), axis=1),
    })
    return report.reset_index()


def _near_duplicate_key(url: pd.Series) -> pd.Series:
    """Lower-case the URL and strip trailing slashes (used only for reporting)."""
    return url.astype(str).str.lower().str.rstrip("/")


def _only_root_path(url: str) -> bool:
    """True if the URL has no path beyond "/" and no query or fragment."""
    parts = urlsplit(url)
    return parts.path in ("", "/") and not parts.query and not parts.fragment


def near_duplicate_url_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """List pairs of different URLs that match after lower-casing and removing
    trailing slashes.

    Input: a dataset with unique URLs (normally the output of clean()).
    Output: one row per pair with url_a, url_b, label_a, label_b,
    labels_conflict, differs_by_case, differs_by_trailing_slash and
    root_path_only (both URLs have no path beyond "/", the one case where
    RFC 3986 treats "x" and "x/" as equivalent). The input is not modified.

    This only reports; clean() does not drop these rows (group decision pending).
    """
    work = pd.DataFrame({
        "row": df.index,
        "URL": df["URL"].astype(str).to_numpy(),
        "label": df[config.TARGET].to_numpy(),
    })
    work["key"] = _near_duplicate_key(work["URL"]).to_numpy()
    work = work[work["key"].duplicated(keep=False)]

    pairs = work.merge(work, on="key", suffixes=("_a", "_b"))
    pairs = pairs[pairs["row_a"] < pairs["row_b"]]
    pairs = pairs[pairs["URL_a"] != pairs["URL_b"]]

    a, b = pairs["URL_a"], pairs["URL_b"]
    # Number of "/" characters at the end of each URL ("x//" and "x/" differ too).
    slashes_a = a.str.len() - a.str.rstrip("/").str.len()
    slashes_b = b.str.len() - b.str.rstrip("/").str.len()
    out = pd.DataFrame({
        "row_a": pairs["row_a"],
        "row_b": pairs["row_b"],
        "url_a": a,
        "url_b": b,
        "label_a": pairs["label_a"],
        "label_b": pairs["label_b"],
        "labels_conflict": pairs["label_a"] != pairs["label_b"],
        "differs_by_case": a.str.rstrip("/") != b.str.rstrip("/"),
        "differs_by_trailing_slash": slashes_a != slashes_b,
        "root_path_only": a.map(_only_root_path) & b.map(_only_root_path),
    })
    return out.sort_values(["row_a", "row_b"]).reset_index(drop=True)
