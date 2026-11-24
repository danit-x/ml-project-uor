"""Decision Tree tuning with grouped CV and phishing as the positive class."""
import unittest
import warnings

import numpy as np

from src import data, evaluate, features
from src.models import tree
from tests.helpers import synthetic_df


class TreeTuningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = features.add_features(synthetic_df())
        # A learnable signal so F1 is not degenerate.
        y = data.get_target(df)
        df["URLLength"] = df["URLLength"] + 2.0 * y
        cls.X, _, cls.y, _ = data.split(df, ["URLLength", "url_path_depth", "NoOfSubDomain"])
        cls.groups = data.group_ids(df).loc[cls.X.index]
        with warnings.catch_warnings():
            warnings.simplefilter("error", evaluate.UngroupedCVWarning)
            cls.search = tree.tune(cls.X, cls.y, groups=cls.groups, n_jobs=1)

    def test_uses_shared_grouped_folds(self):
        expected = evaluate.cv_folds(self.X, self.y, self.groups)
        self.assertEqual(len(self.search.cv), len(expected))
        g = self.groups.to_numpy()
        for (t1, v1), (t2, v2) in zip(self.search.cv, expected):
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(v1, v2)
            self.assertFalse(set(g[t1]) & set(g[v1]))

    def test_refit_on_phishing_f1(self):
        self.assertEqual(self.search.refit, "f1")
        for name in ("precision", "recall", "f1"):
            self.assertEqual(tree.SCORING[name]._kwargs["pos_label"], 1)
        self.assertIn("mean_test_roc_auc", self.search.cv_results_)
        for key, value in self.search.best_params_.items():
            self.assertIn(value, tree.PARAM_GRID[key])

    def test_ungrouped_tuning_warns(self):
        with self.assertWarns(evaluate.UngroupedCVWarning):
            tree.tune(self.X.iloc[:300], self.y.iloc[:300], n_jobs=1)

    def test_build_matches_frozen_constants(self):
        params = tree.build().get_params()
        self.assertEqual(params["max_depth"], tree.BEST_MAX_DEPTH)
        self.assertEqual(params["min_samples_leaf"], tree.BEST_MIN_SAMPLES_LEAF)
        self.assertEqual(params["class_weight"], tree.BEST_CLASS_WEIGHT)
        self.assertTrue(tree.is_frozen_config({"max_depth": tree.BEST_MAX_DEPTH}))
        self.assertTrue(hasattr(tree.train(self.X, self.y), "predict_proba"))


if __name__ == "__main__":
    unittest.main()
