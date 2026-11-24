"""Data part: inspection, integration, cleaning, preprocessing, train/test split.

Owner: Madushanka D.M.C  (proposal sections 4.2 - 4.5, 4.8)

Keep the function names and return types: main.py and the other members'
notebooks call load_raw(), clean(), load_clean_data(), get_target(),
get_original_features() and split().
"""
import hashlib
import re
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config

# Raw label values the dataset is allowed to contain (UCI: 1 = legitimate, 0 = phishing).
VALID_LABELS = (0, 1)

# Group decision 2026-11-24: near-duplicate URLs share one group, and a group is
# never split across train/test or across CV folds. Only these parts are
# normalized; everything else (path case, query string, fragment) is kept,
# because on many servers "/Login" and "/login" are different resources.
_SCHEME = re.compile(r"^https?://", re.IGNORECASE)
_WWW = "www."

# The held-out test set is one fold of a StratifiedGroupKFold, so the number of
# folds follows from config.TEST_SIZE (0.2 -> 5 folds -> about 20 % test rows).
SPLIT_FOLDS = int(round(1 / config.TEST_SIZE))


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

    Near-duplicate URLs (same url_group_key(): scheme, "www.", hostname case,
    trailing slash) are deliberately kept, including the groups whose rows
    disagree on the label (conflicting_label_groups()). Group decision
    2026-11-24: instead of dropping them, split() and the training CV keep
    every group on one side, so they cannot leak between train and test.

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


def url_group_key(url) -> str:
    """Normalized key that puts near-duplicate URLs in the same group.

    Input: one raw URL string (the URL column itself is never changed; the
    engineered features are computed from the original string).
    Output: the URL with
      - a leading "http://" or "https://" removed (any letter case),
      - the hostname lower-cased (hostnames are case-insensitive, RFC 3986),
      - one leading "www." removed from the hostname,
      - trailing "/" characters removed from the path.
    Kept exactly as written: user info, port, path case, the query string
    (case and content) and the fragment. Example:
    "HTTPS://WWW.Example.com/Login/?a=B" -> "example.com/Login?a=B".

    Why only these steps: they are the variations that point to the same
    resource, while path and query case can address different pages. Lower-
    casing the whole URL or dropping the query merges unrelated URLs (groups of
    up to 45 rows were measured when the query was dropped).
    """
    if url is None or (not isinstance(url, str) and pd.isna(url)):
        raise ValueError("url_group_key() needs a URL string, got a missing value")
    rest = _SCHEME.sub("", str(url).strip(), count=1)
    # The authority (user info, host, port) ends at the first "/", "?" or "#".
    end = min((i for i in (rest.find(c) for c in "/?#") if i != -1), default=len(rest))
    authority, remainder = rest[:end], rest[end:]
    userinfo, at, hostport = authority.rpartition("@")
    hostport = hostport.lower()
    if hostport.startswith(_WWW):
        hostport = hostport[len(_WWW):]
    # The path is the remainder up to the query or fragment.
    cut = min((i for i in (remainder.find(c) for c in "?#") if i != -1), default=len(remainder))
    path, tail = remainder[:cut], remainder[cut:]
    return f"{userinfo}{at}{hostport}{path.rstrip('/')}{tail}"


def url_group_keys(df: pd.DataFrame) -> pd.Series:
    """url_group_key() for every row, aligned to df.index (name "url_group_key")."""
    if df["URL"].isna().any():
        raise ValueError("URL has missing values; group keys need every URL")
    return df["URL"].map(url_group_key).rename("url_group_key")


def group_ids(df: pd.DataFrame) -> pd.Series:
    """Integer URL-group id per row, aligned to df.index (name "url_group").

    Rows whose url_group_key() matches share an id. Ids come from the sorted
    keys, so the same set of URLs always gets the same ids. Pass these ids to
    every grouped split or CV call (data.split, evaluate.cv_evaluate, tune()).
    """
    codes, _ = pd.factorize(url_group_keys(df), sort=True)
    return pd.Series(codes, index=df.index, name="url_group")


def conflicting_label_groups(df: pd.DataFrame) -> pd.DataFrame:
    """URL groups whose rows carry more than one label (dataset label noise).

    Output: one row per member URL of each conflicting group, with columns
    url_group_key, URL and the raw label, sorted by key and URL. These rows
    are kept unchanged: no label is edited and no row is dropped. Because a
    group never crosses the train/test boundary, the conflict cannot leak.
    """
    keys = url_group_keys(df)
    n_labels = df[config.TARGET].groupby(keys).nunique()
    conflicted = keys.isin(n_labels.index[n_labels > 1])
    out = pd.DataFrame({
        "url_group_key": keys[conflicted],
        "URL": df.loc[conflicted, "URL"],
        config.TARGET: df.loc[conflicted, config.TARGET],
    })
    return out.sort_values(["url_group_key", "URL"]).reset_index(drop=True)


def fingerprint(*parts) -> str:
    """Content hash of pandas objects / arrays (index and values, in order).

    Used to memoize the slow grouped splits: equal inputs give equal keys.
    """
    digest = hashlib.sha1()
    for part in parts:
        obj = part if isinstance(part, (pd.Series, pd.DataFrame, pd.Index)) else pd.Series(np.asarray(part))
        digest.update(pd.util.hash_pandas_object(obj, index=not isinstance(obj, pd.Index)).to_numpy().tobytes())
        digest.update(b"|")
    return digest.hexdigest()


# StratifiedGroupKFold loops over every group in Python (about 70 s for the
# 231,779 groups of the full data), so each distinct split is computed once.
_SPLIT_CACHE = {}


def split_index(df: pd.DataFrame):
    """Index labels of the training and test rows (group-aware, stratified).

    Uses StratifiedGroupKFold(n_splits=SPLIT_FOLDS, shuffle=True,
    random_state=config.RANDOM_STATE) and takes its first fold as the test
    set (about config.TEST_SIZE of the rows). Every URL group lies wholly in
    train or wholly in test, and the phishing share stays close in both.
    The rows are returned in a seeded random order (like train_test_split),
    so taking "the first n test rows" still gives a random subset.
    Depends only on the URLs and labels, not on which feature columns are
    used; the result is memoized on exactly those inputs.
    """
    y = get_target(df)
    key = fingerprint(df["URL"], y)
    if key not in _SPLIT_CACHE:
        groups = group_ids(df).to_numpy()
        splitter = StratifiedGroupKFold(
            n_splits=SPLIT_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
        )
        train_pos, test_pos = next(splitter.split(np.zeros(len(df)), y.to_numpy(), groups))
        rng = np.random.RandomState(config.RANDOM_STATE)
        _SPLIT_CACHE.clear()  # keep only the latest dataset in memory
        _SPLIT_CACHE[key] = (df.index[rng.permutation(train_pos)], df.index[rng.permutation(test_pos)])
    return _SPLIT_CACHE[key]


def group_subset(groups: pd.Series, n_rows: int, random_state=config.RANDOM_STATE) -> pd.Index:
    """Index labels of a reproducible subset that keeps whole URL groups.

    Input: URL group ids of the candidate rows (e.g. the training rows,
    data.group_ids(df).loc[train_idx]) and the target subset size.
    Output: the labels of every row of randomly ordered groups, taken until
    at least n_rows rows are collected (so the size can exceed n_rows by
    less than one group, at most 4 rows on PhiUSIIL), in the input order.
    Why: tuning an expensive model on a subset must not split a group
    between the subset's CV folds differently from the full data, and a
    seeded group order makes the subset reproducible.
    """
    if n_rows >= len(groups):
        return groups.index
    sizes = groups.value_counts(sort=False)
    order = np.random.RandomState(random_state).permutation(sizes.index.to_numpy())
    cum = sizes.reindex(order).cumsum().to_numpy()
    chosen = order[: int(np.searchsorted(cum, n_rows)) + 1]
    return groups.index[groups.isin(chosen).to_numpy()]


def split(df: pd.DataFrame, feature_cols: list):
    """Group-aware stratified train/test split, identical for every member.

    Returns X_train, X_test, y_train, y_test (same interface as before).
    Rows are assigned by split_index(), so near-duplicate URLs (same
    url_group_key) never appear in both train and test, and the rows are the
    same for any feature_cols. Get the matching training groups with
    group_ids(df).loc[X_train.index] for grouped cross-validation.
    Scaling is done inside each model's pipeline, so it is fitted on the
    training data only (no data leakage).
    """
    train_idx, test_idx = split_index(df)
    X = df[feature_cols]
    y = get_target(df)
    return X.loc[train_idx], X.loc[test_idx], y.loc[train_idx], y.loc[test_idx]


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


def bare_homepage_mask(urls: pd.Series) -> pd.Series:
    """True where a URL is a bare "https://www.<host>" homepage (no path, query or fragment).

    Input: the URL column (any rows). Output: boolean Series on the same index.
    Row-wise and fitted on nothing. Why it exists: in PhiUSIIL every legitimate
    URL has this exact shape (dataset collection format: the legitimate URLs
    are homepages of listed domains), but only about 1 % of phishing URLs do.
    The rule "not a bare https://www homepage -> phishing" is therefore a
    reference baseline: a URL-only model is only learning more than the
    collection format to the extent that it beats this rule.
    A trailing "/" alone still counts as a bare homepage.
    """
    text = urls.astype("string").fillna("").str.strip()
    https_www = text.str.lower().str.startswith("https://www.")
    rest = text.str.slice(len("https://www."))
    bare = ~rest.str.rstrip("/").str.contains(r"[/?#]", regex=True)
    return (https_www & bare).astype(bool).rename("bare_https_www_homepage")


def _near_duplicate_key(url: pd.Series) -> pd.Series:
    """Lower-case the URL and strip trailing slashes (used only for reporting).

    This is notebook 01's earlier, broader report key. The split and CV use
    url_group_key() instead, which keeps path and query case.
    """
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

    This only reports; clean() does not drop these rows. Decided 2026-11-24:
    they are kept and grouped with url_group_key() (a narrower key that keeps
    path and query case), so split() and the training CV never separate them.
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
