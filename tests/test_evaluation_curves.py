"""Evaluation curves: grouped folds only, no train/validation group overlap, training rows only,
and AUC agreement with cv_evaluate() (the source of reports/cv_results.csv)."""
import unittest

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

import evaluation_curves
from src import config, data, evaluate, features
from tests.helpers import synthetic_df


class _Setup:
    @classmethod
    def setUpClass(cls):
        df = features.add_features(synthetic_df())
        y = data.get_target(df)
        df["URLLength"] = df["URLLength"] + 1.5 * y        # learnable, not perfect
        cls.df = df
        cls.groups_all = data.group_ids(df)
        cls.train_idx, cls.test_idx = data.split_index(df)
        cls.X, cls.y, cls.g = evaluation_curves.training_inputs(df, cls.groups_all, cls.train_idx)
        cls.pipe = make_pipeline(features.FeatureSetSelector("url_engineered"),
                                 DecisionTreeClassifier(max_depth=4, random_state=config.RANDOM_STATE))


class LearningCurveProtocolTest(_Setup, unittest.TestCase):
    def test_curve_inputs_are_training_rows_only(self):
        self.assertTrue(set(self.X.index) <= set(self.train_idx))
        self.assertFalse(set(self.X.index) & set(self.test_idx))
        self.assertFalse(set(self.g) & set(self.groups_all.loc[self.test_idx]))

    def test_ungrouped_curve_is_refused(self):
        with self.assertRaises(ValueError):
            evaluate.grouped_learning_curve(self.pipe, self.X, self.y, None)

    def test_subsets_use_grouped_folds_and_never_share_groups(self):
        folds = evaluate.cv_folds(self.X, self.y, self.g)
        g = self.g.to_numpy()
        seen = set()
        for k, size, fit_pos, valid_pos in evaluate.learning_curve_subsets(folds, g):
            train_pos, fold_valid = folds[k]
            np.testing.assert_array_equal(valid_pos, fold_valid)          # the cv_results folds
            self.assertTrue(set(fit_pos) <= set(train_pos))              # never validation rows
            self.assertFalse(set(g[fit_pos]) & set(g[valid_pos]))         # no URL group on both sides
            fold_g = pd.Series(g[train_pos])
            kept = set(g[fit_pos])
            self.assertEqual(int(fold_g.isin(kept).sum()), len(fit_pos))  # whole groups only
            self.assertGreaterEqual(len(fit_pos), round(size * len(train_pos)))
            if size == 1.0:
                self.assertEqual(set(fit_pos), set(train_pos))
            seen.add((k, size))
        self.assertEqual(len(seen), config.CV_FOLDS * len(evaluate.LEARNING_CURVE_SIZES))

    def test_overlapping_fold_is_rejected(self):
        g = self.g.to_numpy()
        bad = [(np.arange(0, 20), np.arange(10, 30))]
        with self.assertRaises(AssertionError):
            list(evaluate.learning_curve_subsets(bad, g))


class AucConsistencyTest(_Setup, unittest.TestCase):
    def test_full_size_fits_reproduce_cv_evaluate(self):
        table, roc_folds = evaluate.grouped_learning_curve(self.pipe, self.X, self.y, self.g)
        ref = evaluate.cv_evaluate(self.pipe, self.X, self.y, groups=self.g)
        full = table.loc[table["train_size"] == 1.0]
        self.assertAlmostEqual(full["valid_roc_auc"].mean(), ref["roc_auc_mean"], places=10)
        self.assertAlmostEqual(full["valid_f1"].mean(), ref["f1_mean"], places=10)
        _, mean, std, aucs = evaluate.mean_roc_curve(roc_folds)
        self.assertEqual(len(aucs), config.CV_FOLDS)
        self.assertAlmostEqual(aucs.mean(), ref["roc_auc_mean"], places=10)
        self.assertEqual((mean[0], mean[-1]), (0.0, 1.0))
        self.assertTrue(np.all(np.diff(mean) >= -1e-12))                  # a ROC curve never falls

    def test_summary_mean_and_std(self):
        table, _ = evaluate.grouped_learning_curve(self.pipe, self.X, self.y, self.g)
        summary = evaluate.summarize_learning_curve(table.assign(model="Decision Tree"))
        self.assertEqual(sorted(summary["train_size"]), sorted(evaluate.LEARNING_CURVE_SIZES))
        half = table.loc[table["train_size"] == 0.5, "valid_f1"]
        row = summary.loc[summary["train_size"] == 0.5].iloc[0]
        self.assertAlmostEqual(row["valid_f1_mean"], half.mean())
        self.assertAlmostEqual(row["valid_f1_std"], half.std(ddof=0))


if __name__ == "__main__":
    unittest.main()
