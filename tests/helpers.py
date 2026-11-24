"""Small synthetic PhiUSIIL-like data for the unit tests (no dataset needed)."""
import numpy as np
import pandas as pd

from src import config, features


def synthetic_df(n_unique=1500, seed=0) -> pd.DataFrame:
    """Rows with URL, raw label (0 = phishing) and every original numeric column.

    Includes near-duplicate groups (scheme / www / host case / trailing slash
    variants) and one label-conflict group, like the real data.
    """
    rng = np.random.RandomState(seed)
    urls, labels = [], []
    for i in range(n_unique):
        label = int(rng.rand() < 0.57)  # about 43 % phishing, as in PhiUSIIL
        path = "" if i % 3 == 0 else f"/Path{i}/Item"
        query = f"?id={i}&Ref=AbC" if i % 7 == 0 else ""
        urls.append(f"https://www.site{i}.com{path}{query}")
        labels.append(label)
        if i % 10 == 0:  # near-duplicate of the row above, same label
            urls.append(f"http://SITE{i}.com{path}/{query}")
            labels.append(label)
    # One group whose two rows disagree on the label (dataset label noise).
    urls += ["http://www.conflict.com", "https://www.conflict.com"]
    labels += [0, 1]
    df = pd.DataFrame({"URL": urls, config.TARGET: labels})
    for col in features.URL_FEATURES + features.PAGE_FEATURES:
        df[col] = rng.normal(size=len(df))
    for col in ("FILENAME", "Domain", "TLD", "Title"):
        df[col] = "x"
    return df
