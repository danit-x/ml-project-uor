"""Progress-evaluation integration checks: frozen model configurations, bounded
tuning helpers, group-preserving subsets, the format-controlled breakdown,
notebook validity and compilation of every Python file."""
import py_compile
import unittest

import numpy as np
import pandas as pd

import main
from src import config, data, evaluate, features
from src.models import forest, logreg, tree
from tests.helpers import synthetic_df


class _Data:
    @classmethod
    def setUpClass(cls):
        df = features.add_features(synthetic_df())
        y = data.get_target(df)
        df["URLLength"] = df["URLLength"] + 2.0 * y        # learnable signal
        cls.df = df
        cls.train_idx, cls.test_idx = data.split_index(df)
        cls.groups = data.group_ids(df)
        cls.cols = ["URLLength", "url_path_depth", "NoOfSubDomain"]


class LogRegConfigTest(_Data, unittest.TestCase):
    def test_build_uses_all_best_constants(self):
        lr = logreg.build()[-1]
        self.assertEqual((lr.C, lr.l1_ratio, lr.solver),
                         (logreg.BEST_C, logreg.BEST_L1_RATIO, logreg.BEST_SOLVER))
        self.assertEqual(lr.max_iter, logreg.MAX_ITER)
        self.assertEqual(lr.random_state, config.RANDOM_STATE)

    def test_best_solver_supports_best_penalty(self):
        # lbfgs cannot fit an L1 penalty; liblinear cannot fit an elastic net.
        if logreg.BEST_L1_RATIO > 0:
            self.assertEqual(logreg.BEST_SOLVER, "liblinear")
        self.assertIn(logreg.BEST_SOLVER, {"lbfgs", "liblinear"})

    def test_best_config_is_a_searched_candidate(self):
        searched = logreg.search_candidates(logreg.FOLLOWUP_SPACE) + logreg.search_candidates()
        self.assertIn((logreg.BEST_SOLVER, logreg.BEST_L1_RATIO, logreg.BEST_C),
                      {(c["solver"], c["l1_ratio"], c["C"]) for c in searched})

    def test_followup_space_is_l2_lbfgs_only(self):
        cands = logreg.search_candidates(logreg.FOLLOWUP_SPACE)
        self.assertTrue(cands)
        self.assertEqual({(c["penalty"], c["solver"]) for c in cands}, {("l2", "lbfgs")})

    def test_grouped_search_space_callback_and_budget(self):
        X = self.df.loc[self.train_idx, self.cols]
        y = data.get_target(self.df).loc[self.train_idx]
        folds = evaluate.cv_folds(X, y, self.groups.loc[self.train_idx])
        space = [{"solver": "lbfgs", "l1_ratio": 0.0, "C": [0.1, 1.0]}]
        seen = []
        out = logreg.grouped_search(X, y, folds, n_jobs=1, log=lambda s: None,
                                    space=space, on_row=seen.append)
        self.assertEqual(len(out), 2 * config.CV_FOLDS)
        self.assertEqual(len(seen), len(out))
        self.assertTrue(out["converged"].all())
        table, chosen, best = logreg.select_one_se(out)
        self.assertTrue(table["valid"].all())
        self.assertLessEqual(chosen["C"], best["C"])
        # A zero budget stops after the first collected fit and keeps it.
        short = logreg.grouped_search(X, y, folds, n_jobs=1, log=lambda s: None,
                                      space=space, time_budget=0)
        self.assertGreaterEqual(len(short), 1)
        self.assertLess(len(short), 2 * config.CV_FOLDS)
        table, _, _ = logreg.select_one_se(short)
        self.assertFalse(table["valid"].all())   # incomplete candidates are never valid

    def test_non_converged_candidate_cannot_win(self):
        rows = []
        for C, conv, f1 in ((1.0, True, 0.90), (10.0, False, 0.99)):
            for k in range(config.CV_FOLDS):
                rows.append({"penalty": "l2", "solver": "lbfgs", "C": C, "fold": k, "fit_ok": True,
                             "converged": conv, "n_iter": 5, "convergence_warnings": int(not conv),
                             "n_nonzero_coef": 3, "fit_seconds": 0.1, "accuracy": f1,
                             "precision": f1, "recall": f1, "f1": f1 + 0.001 * k, "roc_auc": f1})
        table, chosen, best = logreg.select_one_se(pd.DataFrame(rows))
        self.assertEqual((chosen["C"], best["C"]), (1.0, 1.0))


class TreeAndForestConfigTest(unittest.TestCase):
    def test_tree_frozen_to_complexity_review_choice(self):
        self.assertEqual((tree.BEST_MAX_DEPTH, tree.BEST_MIN_SAMPLES_LEAF, tree.BEST_CLASS_WEIGHT),
                         (20, 1, None))
        self.assertIn(tree.BEST_MAX_DEPTH, tree.PARAM_GRID["max_depth"])
        self.assertEqual(tree.build().random_state, config.RANDOM_STATE)

    def test_forest_best_params_from_bounded_grid(self):
        for key, values in forest.SUBSET_PARAM_GRID.items():
            self.assertIn(forest.BEST_PARAMS[key], values)
        self.assertEqual(forest.BEST_PARAMS["n_estimators"], forest.SUBSET_N_ESTIMATORS)
        self.assertNotEqual(forest.TUNING_NOTE, "PENDING grouped retune")


class GroupSubsetTest(_Data, unittest.TestCase):
    def test_whole_groups_reproducible_and_sized(self):
        g = self.groups.loc[self.train_idx]
        sub = data.group_subset(g, 500)
        self.assertGreaterEqual(len(sub), 500)
        self.assertLess(len(sub), 500 + g.value_counts().max())
        chosen = set(g.loc[sub])
        self.assertEqual(int(g.isin(chosen).sum()), len(sub))      # no group cut
        self.assertTrue(sub.equals(data.group_subset(g, 500)))     # reproducible
        self.assertTrue(set(sub) <= set(self.train_idx))
        self.assertTrue(data.group_subset(g, 10 ** 9).equals(g.index))


class FormatControlledTest(_Data, unittest.TestCase):
    def test_breakdown_counts(self):
        primary = features.PRIMARY_FEATURE_SET
        sets = {primary: self.cols}
        X_tr = self.df.loc[self.train_idx, self.cols]
        y_tr = data.get_target(self.df).loc[self.train_idx]
        fitted = {(primary, name): tree.build().set_params(max_depth=3).fit(X_tr, y_tr)
                  for name in main.MAIN_MODELS}
        out = main.format_controlled(self.df, self.test_idx, sets, fitted)
        self.assertEqual(len(out), 2 * (len(main.MAIN_MODELS) + 1))
        self.assertEqual(int(out.groupby("model")["n_rows"].sum().iloc[0]), len(self.test_idx))
        for r in out.itertuples():
            self.assertEqual(r.tn + r.fp + r.fn + r.tp, r.n_rows)
        rule = out[(out["model"] == "format rule (reference)") & (out["subset"] != "other URLs")]
        self.assertEqual(int(rule["tp"].iloc[0] + rule["fp"].iloc[0]), 0)   # rule never flags bare homepages


class NotebookAndSourceTest(unittest.TestCase):
    def test_notebooks_valid_and_ids_unique(self):
        import nbformat

        for path in sorted((config.ROOT / "notebooks").glob("*.ipynb")):
            nb = nbformat.read(path, as_version=4)
            nbformat.validate(nb)
            ids = [c.get("id") for c in nb.cells]
            self.assertEqual(len(ids), len(set(ids)), path.name)
            for cell in nb.cells:
                if cell.cell_type == "code" and not cell.source.lstrip().startswith(("%", "!")):
                    compile(cell.source, f"{path.name}:{cell.get('id')}", "exec")

    def test_python_files_compile(self):
        files = [p for p in config.ROOT.glob("*.py")] + list((config.ROOT / "src").rglob("*.py")) \
            + list((config.ROOT / "tests").glob("*.py"))
        self.assertTrue(files)
        for path in files:
            py_compile.compile(str(path), doraise=True)


if __name__ == "__main__":
    unittest.main()
