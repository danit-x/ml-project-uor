"""Per-fold feature selection, the format reference rule and the official comparison."""
import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

import main
from src import config, data, evaluate, features
from src.models import forest, logreg
from tests.helpers import synthetic_df


class _Setup:
    @classmethod
    def setUpClass(cls):
        df = features.add_features(synthetic_df())
        y = data.get_target(df)
        df["URLLength"] = df["URLLength"] + 2.0 * y        # learnable signal
        df["NoOfLettersInURL"] = df["URLLength"] * 0.999   # redundant copy, |r| > 0.98
        cls.df = df
        cls.train_idx, cls.test_idx = data.split_index(df)
        cls.train_df = df.loc[cls.train_idx]
        cls.y_train = y.loc[cls.train_idx]
        cls.groups = data.group_ids(df).loc[cls.train_idx]
        cls.X = cls.train_df[features.candidate_columns()]
        cls.folds = evaluate.cv_folds(cls.X, cls.y_train, cls.groups)


class FeatureSetSelectorTest(_Setup, unittest.TestCase):
    def test_candidates_cover_every_set(self):
        sets = features.build_feature_sets(self.train_df)
        for cols in sets.values():
            self.assertTrue(set(cols) <= set(features.candidate_columns()))
        self.assertNotIn("URLSimilarityIndex", features.candidate_columns())

    def test_selector_equals_build_feature_sets_on_same_rows(self):
        for name in features.FEATURE_SETS:
            sel = features.FeatureSetSelector(name).fit(self.X)
            self.assertEqual(sel.columns_, features.build_feature_sets(self.X)[name])
            self.assertEqual(list(sel.transform(self.X).columns), sel.columns_)
        self.assertNotIn("NoOfLettersInURL", features.FeatureSetSelector().fit(self.X).columns_)

    def test_unknown_set_rejected(self):
        with self.assertRaises(ValueError):
            features.FeatureSetSelector("bogus").fit(self.X)

    def test_selector_is_refitted_on_fold_rows_only(self):
        pipe = make_pipeline(features.FeatureSetSelector("url_engineered"),
                             DecisionTreeClassifier(random_state=config.RANDOM_STATE))
        cv = cross_validate(pipe, self.X, self.y_train, cv=self.folds,
                            scoring="f1", return_estimator=True)
        for est, (fit_pos, _) in zip(cv["estimator"], self.folds):
            expected = features.build_feature_sets(self.X.iloc[fit_pos])["url_engineered"]
            self.assertEqual(est[0].columns_, expected)

    def test_fold_report_and_equivalence(self):
        report = features.fold_selection_report(self.train_df, self.folds)
        self.assertEqual(len(report), config.CV_FOLDS * len(features.FEATURE_SETS))
        self.assertTrue(report["same_as_outer"].all())
        # When every fold picks the outer columns, per-fold and fixed selection give equal CV.
        cols = features.build_feature_sets(self.train_df)["url_engineered"]
        model = DecisionTreeClassifier(random_state=config.RANDOM_STATE)
        fixed = evaluate.cv_evaluate(model, self.train_df[cols], self.y_train, groups=self.groups)
        per_fold = evaluate.cv_evaluate(
            make_pipeline(features.FeatureSetSelector("url_engineered"), model),
            self.X, self.y_train, groups=self.groups)
        self.assertEqual(fixed, per_fold)

    def test_correlation_margin_reports_removed_column(self):
        sets = features.build_feature_sets(self.train_df)
        margin = features.correlation_margin(self.train_df, sets)
        removed = margin.loc[margin["column"] == "NoOfLettersInURL", "abs_r"]
        self.assertTrue((removed >= features.CORRELATION_THRESHOLD).all() and len(removed))
        kept = margin.loc[margin["kind"].str.startswith("max"), "abs_r"]
        self.assertTrue((kept < features.CORRELATION_THRESHOLD).all())


class FormatBaselineTest(_Setup, unittest.TestCase):
    def test_bare_homepage_mask(self):
        urls = pd.Series([
            "https://www.example.com", "HTTPS://WWW.Example.com/", "https://www.a.co.uk",
            "http://www.example.com", "https://example.com", "https://www.example.com/login",
            "https://www.example.com?a=1", "https://www.example.com/#x", None,
        ])
        self.assertEqual(data.bare_homepage_mask(urls).tolist(),
                         [True, True, True, False, False, False, False, False, False])

    def test_prediction_metrics_matches_evaluate(self):
        cols = ["URLLength", "url_path_depth"]
        model = DecisionTreeClassifier(random_state=0).fit(self.train_df[cols], self.y_train)
        X_te, y_te = self.df.loc[self.test_idx, cols], data.get_target(self.df).loc[self.test_idx]
        self.assertEqual(
            evaluate.evaluate(model, X_te, y_te),
            evaluate.prediction_metrics(y_te, model.predict(X_te), model.predict_proba(X_te)[:, 1]))

    def test_rule_cv_metrics_per_fold(self):
        rule = self.y_train.to_numpy()  # a perfect rule
        out = evaluate.rule_cv_metrics(self.y_train, rule, self.folds)
        self.assertEqual(set(out), {f"{m}_{s}" for m in evaluate.CV_METRICS for s in ("mean", "std")})
        self.assertEqual(out["f1_mean"], 1.0)
        self.assertEqual(out["f1_std"], 0.0)

    def test_format_baseline_row(self):
        out = main.format_baseline(self.df, self.train_idx, self.test_idx, self.groups)
        self.assertEqual(len(out), 1)
        for col in ("cv_f1_mean", "test_f1", "test_tp", "share_phishing_bare_train"):
            self.assertIn(col, out.columns)


class OfficialComparisonTest(unittest.TestCase):
    def _table(self):
        rows = {}
        for s in features.FEATURE_SETS:
            for m in ("Logistic Regression", "Decision Tree", "Random Forest", "SVM"):
                rows[(s, m)] = {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9,
                                "roc_auc": 0.9, "tn": 1, "fp": 1, "fn": 1, "tp": 1, "brier": 0.1,
                                "n_features": 3, "protocol": main._protocol(s, m)}
        return evaluate.compare(rows)

    def test_protocol_labels(self):
        self.assertEqual(main._protocol("url_engineered", "Decision Tree"), "official")
        self.assertEqual(main._protocol("url_base", "Random Forest"), "secondary")
        self.assertEqual(main._protocol("url_engineered", "SVM"), "provisional")

    def test_official_has_three_main_models_on_primary_only(self):
        table = self._table()
        cv = pd.DataFrame([
            {"feature_set": s, "model": m,
             **{f"{k}_{t}": 0.5 for k in evaluate.CV_METRICS for t in ("mean", "std")}}
            for s in features.FEATURE_SETS for m in main.MAIN_MODELS])
        out = main.official_comparison(table, cv)
        self.assertEqual(out["model"].tolist(), list(main.MAIN_MODELS))
        self.assertEqual(set(out["feature_set"]), {features.PRIMARY_FEATURE_SET})
        for col in ("cv_f1_mean", "cv_f1_std", "test_f1", "test_roc_auc", "test_tn", "test_tp"):
            self.assertIn(col, out.columns)
        self.assertNotIn("SVM", out["model"].tolist())
        self.assertEqual(len(main.official_comparison(table)), 3)   # without CV


class FrozenConfigTest(unittest.TestCase):
    def test_logreg_build_uses_constants(self):
        lr = logreg.build()[-1]
        self.assertEqual((lr.C, lr.l1_ratio), (logreg.BEST_C, logreg.BEST_L1_RATIO))

    def test_forest_build_uses_constants(self):
        params = forest.build().get_params()
        for key, value in forest.BEST_PARAMS.items():
            self.assertEqual(params[key], value)
        self.assertEqual(params["random_state"], config.RANDOM_STATE)


if __name__ == "__main__":
    unittest.main()
