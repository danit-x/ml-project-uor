"""Engineered URL features, feature groups, exclusions and training-only selection."""
import unittest

import pandas as pd

from src import config, data, features
from tests.helpers import synthetic_df


def _feats(url):
    out = features.add_features(pd.DataFrame({"URL": [url]}))
    return out.loc[0, features.ENGINEERED_FEATURES].tolist()


class EngineeredFeatureTest(unittest.TestCase):
    def test_known_urls(self):
        # depth, longest segment, query fields, percent-encoded bytes
        self.assertEqual(_feats("https://a.com"), [0, 0, 0, 0])
        self.assertEqual(_feats("https://a.com/"), [0, 0, 0, 0])
        self.assertEqual(_feats("https://a.com/abc/de//f/"), [3, 3, 0, 0])
        self.assertEqual(_feats("https://a.com/p?x=1&y=2&&z"), [1, 1, 3, 0])
        self.assertEqual(_feats("https://a.com/a%20b?q=%2F%zz"), [1, 5, 1, 2])

    def test_url_without_scheme_parsed_as_host(self):
        self.assertEqual(_feats("www.a.com/x/y?k=v"), [2, 1, 1, 0])

    def test_fragment_ignored(self):
        self.assertEqual(_feats("https://a.com/p#x/y%20"), [1, 1, 0, 0])

    def test_missing_and_invalid_urls_give_zeros(self):
        out = features.add_features(pd.DataFrame({"URL": [None, "", "http://[::1"]}))
        self.assertEqual(out[features.ENGINEERED_FEATURES].to_numpy().sum(), 0)

    def test_numeric_rowwise_and_input_unchanged(self):
        df = pd.DataFrame({"URL": ["https://a.com/x", "https://b.com/y/z"]}, index=[10, 20])
        out = features.add_features(df)
        self.assertTrue(out.index.equals(df.index))
        self.assertTrue(out["URL"].equals(df["URL"]))
        self.assertNotIn("url_path_depth", df.columns)
        for col in features.ENGINEERED_FEATURES:
            self.assertEqual(str(out[col].dtype), "int64")
        # Row-wise: a row's value does not depend on the other rows.
        alone = features.add_features(df.iloc[[1]])
        self.assertEqual(alone.loc[20, "url_path_depth"], out.loc[20, "url_path_depth"])


class FeatureSetTest(unittest.TestCase):
    def setUp(self):
        self.df = features.add_features(synthetic_df())
        self.train_idx, self.test_idx = data.split_index(self.df)
        self.sets = features.build_feature_sets(self.df.loc[self.train_idx])

    def test_groups_cover_original_columns(self):
        features.check_feature_groups(data.get_original_features(self.df))
        with self.assertRaises(ValueError):
            features.check_feature_groups(data.get_original_features(self.df) + ["NewColumn"])

    def test_url_similarity_index_excluded_everywhere(self):
        self.assertEqual(set(self.sets), set(features.FEATURE_SETS))
        for cols in self.sets.values():
            self.assertNotIn("URLSimilarityIndex", cols)
        self.assertNotIn("URLSimilarityIndex", features.select_features(
            self.df, ["URLSimilarityIndex", "URLLength"]))

    def test_url_sets_have_no_https_or_page_features(self):
        page = set(features.PAGE_FEATURES) | {"IsHTTPS"}
        for name in ("url_base", "url_engineered"):
            self.assertFalse(page & set(self.sets[name]), name)
        self.assertIn("IsHTTPS", self.sets["extended"])
        self.assertTrue(set(features.PAGE_FEATURES) <= set(self.sets["extended"]))

    def test_sets_nested_and_primary(self):
        base, eng, ext = (self.sets[k] for k in features.FEATURE_SETS)
        self.assertEqual(eng[:len(base)], base)
        self.assertEqual(ext[:len(eng)], eng)
        self.assertEqual(features.PRIMARY_FEATURE_SET, "url_engineered")
        self.assertTrue(set(eng) - set(base) <= set(features.ENGINEERED_FEATURES))

    def test_selection_uses_training_rows_only(self):
        # Make an engineered feature an exact copy of URLLength on the TEST rows
        # only. Selection on the training rows must not notice.
        changed = self.df.copy()
        changed["url_path_depth"] = changed["url_path_depth"].astype(float)
        changed.loc[self.test_idx, "url_path_depth"] = changed.loc[self.test_idx, "URLLength"] * 1e6
        again = features.build_feature_sets(changed.loc[self.train_idx])
        self.assertEqual(again, self.sets)
        # Sanity check: the same change on training rows does change the result.
        changed.loc[self.train_idx, "url_path_depth"] = changed.loc[self.train_idx, "URLLength"]
        self.assertNotIn("url_path_depth",
                         features.build_feature_sets(changed.loc[self.train_idx])["url_engineered"])


@unittest.skipUnless(config.RAW_FILE.exists(), "PhiUSIIL CSV not in data/raw/")
class RealDataColumnsTest(unittest.TestCase):
    def test_feature_groups_match_dataset_header(self):
        header = pd.read_csv(config.RAW_FILE, nrows=200)
        features.check_feature_groups(data.get_original_features(header))


if __name__ == "__main__":
    unittest.main()
