"""URL normalization, URL groups, label-conflict groups and the grouped split."""
import unittest
import warnings

import numpy as np
import pandas as pd

from src import config, data, evaluate
from tests.helpers import synthetic_df


class UrlGroupKeyTest(unittest.TestCase):
    def test_scheme_removed_any_case(self):
        for url in ("http://example.com/a", "https://example.com/a", "HTTPS://example.com/a"):
            self.assertEqual(data.url_group_key(url), "example.com/a")

    def test_only_http_and_https_schemes_removed(self):
        self.assertEqual(data.url_group_key("ftp://example.com/a"), "ftp://example.com/a")

    def test_leading_www_removed_once_from_hostname_only(self):
        self.assertEqual(data.url_group_key("https://www.example.com"), "example.com")
        self.assertEqual(data.url_group_key("https://WWW.example.com"), "example.com")
        self.assertEqual(data.url_group_key("https://www.www.example.com"), "www.example.com")
        self.assertEqual(data.url_group_key("https://wwwexample.com"), "wwwexample.com")
        self.assertEqual(data.url_group_key("https://mail.www.example.com"), "mail.www.example.com")
        self.assertEqual(data.url_group_key("https://example.com/www.page"), "example.com/www.page")

    def test_hostname_lowercased(self):
        self.assertEqual(data.url_group_key("https://ExAmple.COM"), "example.com")

    def test_path_and_query_case_preserved(self):
        self.assertEqual(
            data.url_group_key("https://Example.com/Login/Page?User=Bob&Token=XyZ"),
            "example.com/Login/Page?User=Bob&Token=XyZ",
        )
        self.assertNotEqual(data.url_group_key("https://a.com/Login"),
                            data.url_group_key("https://a.com/login"))

    def test_query_not_removed(self):
        self.assertNotEqual(data.url_group_key("https://a.com/p?x=1"),
                            data.url_group_key("https://a.com/p?x=2"))
        self.assertNotEqual(data.url_group_key("https://a.com/p?x=1"),
                            data.url_group_key("https://a.com/p"))

    def test_trailing_slash_removed_from_path(self):
        self.assertEqual(data.url_group_key("https://a.com/"), "a.com")
        self.assertEqual(data.url_group_key("https://a.com/p/"), "a.com/p")
        self.assertEqual(data.url_group_key("https://a.com/p//"), "a.com/p")
        self.assertEqual(data.url_group_key("https://a.com/p/?q=1"), "a.com/p?q=1")
        # A slash inside the query or fragment is not part of the path.
        self.assertEqual(data.url_group_key("https://a.com/p?next=/x/"), "a.com/p?next=/x/")
        self.assertEqual(data.url_group_key("https://a.com/p#Frag/"), "a.com/p#Frag/")

    def test_userinfo_and_port_kept(self):
        self.assertEqual(data.url_group_key("http://User:PW@WWW.Ex.com:8080/P/"),
                         "User:PW@ex.com:8080/P")

    def test_missing_url_rejected(self):
        with self.assertRaises(ValueError):
            data.url_group_key(None)
        with self.assertRaises(ValueError):
            data.url_group_key(np.nan)

    def test_combined_variants_share_one_key(self):
        variants = ["http://www.Example.com/Shop/", "https://example.com/Shop",
                    "HTTPS://WWW.EXAMPLE.COM/Shop//"]
        self.assertEqual(len({data.url_group_key(u) for u in variants}), 1)


class GroupIdsTest(unittest.TestCase):
    def test_ids_shared_by_near_duplicates_and_stable(self):
        df = pd.DataFrame({"URL": ["https://www.a.com/X/", "http://a.com/X", "https://a.com/x"]})
        ids = data.group_ids(df)
        self.assertEqual(ids.iloc[0], ids.iloc[1])
        self.assertNotEqual(ids.iloc[0], ids.iloc[2])
        shuffled = df.iloc[[2, 0, 1]]
        self.assertTrue(data.group_ids(shuffled).sort_index().equals(ids))

    def test_original_urls_unchanged(self):
        df = synthetic_df(200)
        before = df["URL"].copy()
        data.group_ids(df)
        data.conflicting_label_groups(df)
        self.assertTrue(df["URL"].equals(before))


class ConflictGroupTest(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_df()

    def test_conflict_group_reported_not_resolved(self):
        report = data.conflicting_label_groups(self.df)
        self.assertEqual(report["url_group_key"].nunique(), 1)
        self.assertEqual(sorted(report[config.TARGET]), [0, 1])
        # Rows and labels are untouched.
        self.assertEqual(len(data.clean(self.df)), len(self.df))
        rows = self.df[self.df["URL"].str.contains("conflict.com")]
        self.assertEqual(sorted(rows[config.TARGET]), [0, 1])

    def test_conflict_group_stays_on_one_side(self):
        train_idx, test_idx = data.split_index(self.df)
        rows = self.df.index[self.df["URL"].str.contains("conflict.com")]
        self.assertTrue(set(rows) <= set(train_idx) or set(rows) <= set(test_idx))


class GroupedSplitTest(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_df()
        self.cols = ["URLLength", "NoOfImage"]

    def test_no_group_overlap(self):
        X_train, X_test, _, _ = data.split(self.df, self.cols)
        groups = data.group_ids(self.df)
        self.assertFalse(set(groups.loc[X_train.index]) & set(groups.loc[X_test.index]))
        self.assertEqual(len(X_train) + len(X_test), len(self.df))
        self.assertFalse(set(X_train.index) & set(X_test.index))

    def test_test_size_and_class_balance(self):
        _, X_test, y_train, y_test = data.split(self.df, self.cols)
        self.assertAlmostEqual(len(X_test) / len(self.df), config.TEST_SIZE, delta=0.02)
        self.assertAlmostEqual(y_train.mean(), y_test.mean(), delta=0.02)

    def test_reproducible_and_independent_of_columns(self):
        a = data.split(self.df, self.cols)
        data._SPLIT_CACHE.clear()  # recompute from scratch
        b = data.split(self.df, ["DomainLength"])
        self.assertTrue(a[0].index.equals(b[0].index))
        self.assertTrue(a[1].index.equals(b[1].index))
        self.assertTrue(a[3].equals(b[3]))

    def test_split_interface_unchanged(self):
        X_train, X_test, y_train, y_test = data.split(self.df, self.cols)
        self.assertEqual(list(X_train.columns), self.cols)
        self.assertTrue(X_train.index.equals(y_train.index))
        self.assertTrue(X_test.index.equals(y_test.index))
        self.assertEqual(set(y_test.unique()), {0, 1})


class GroupedCVTest(unittest.TestCase):
    def setUp(self):
        df = synthetic_df()
        self.X, _, self.y, _ = data.split(df, ["URLLength"])
        self.groups = data.group_ids(df).loc[self.X.index]

    def test_folds_keep_groups_together(self):
        folds = evaluate.cv_folds(self.X, self.y, self.groups)
        self.assertEqual(len(folds), config.CV_FOLDS)
        g = self.groups.to_numpy()
        for train, valid in folds:
            self.assertFalse(set(g[train]) & set(g[valid]))

    def test_folds_identical_on_repeat(self):
        a = evaluate.cv_folds(self.X, self.y, self.groups)
        evaluate._FOLD_CACHE.clear()
        b = evaluate.cv_folds(self.X, self.y, self.groups)
        for (ta, va), (tb, vb) in zip(a, b):
            np.testing.assert_array_equal(ta, tb)
            np.testing.assert_array_equal(va, vb)

    def test_ungrouped_cv_warns(self):
        with self.assertWarns(evaluate.UngroupedCVWarning):
            evaluate.cv_folds(self.X, self.y)

    def test_group_series_aligned_by_index(self):
        shuffled = self.groups.sample(frac=1, random_state=1)
        a = evaluate.cv_folds(self.X, self.y, self.groups)
        b = evaluate.cv_folds(self.X, self.y, shuffled)
        for (ta, _), (tb, _) in zip(a, b):
            np.testing.assert_array_equal(ta, tb)
        with self.assertRaises(ValueError):
            evaluate.cv_folds(self.X, self.y, self.groups.iloc[:-1].to_numpy())

    def test_cv_evaluate_with_groups(self):
        from sklearn.tree import DecisionTreeClassifier
        with warnings.catch_warnings():
            warnings.simplefilter("error", evaluate.UngroupedCVWarning)
            out = evaluate.cv_evaluate(DecisionTreeClassifier(max_depth=2), self.X, self.y,
                                       groups=self.groups)
        self.assertIn("f1_mean", out)


if __name__ == "__main__":
    unittest.main()
