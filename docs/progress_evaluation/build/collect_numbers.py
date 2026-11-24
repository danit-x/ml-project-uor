"""Collect every number used in the progress document and slides into numbers.json.

    python docs/progress_evaluation/build/collect_numbers.py

Reads only files written by committed code (`python main.py`, `python tune_models.py`) and the
raw dataset; it trains nothing and changes nothing in reports/. The document and slide
generators (build_document.js, build_presentation.js) read numbers.json, so no metric is typed
by hand. Not part of the ML pipeline.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src import config, data  # noqa: E402

R = config.RESULTS_DIR
T = R / "tuning"
OUT = Path(__file__).with_name("numbers.json")
LOG = R / "main_full_run_2026-11-24.log"


def f4(v):
    return f"{v:.4f}"


def f5(v):
    return f"{v:.5f}"


def n(v):
    return f"{int(v):,}"


def pct(v, d=1):
    return f"{100 * v:.{d}f} %"


def log_numbers():
    text = LOG.read_text(encoding="utf-8")
    def grab(pattern):
        m = re.search(pattern, text)
        if not m:
            raise ValueError(f"pattern not in {LOG.name}: {pattern}")
        return m.groups()
    rows_clean, = grab(r"Rows after cleaning: ([\d,]+)")
    groups, multi, conflict, conflict_rows = grab(
        r"URL groups: ([\d,]+) \(([\d,]+) with several rows\); label-conflict groups kept as label noise: (\d+) \(([\d,]+) rows\)")
    train, test, share_tr, share_te, shared = grab(
        r"Train rows ([\d,]+) \| test rows ([\d,]+) .*phishing share train ([\d.]+), test ([\d.]+) \| shared groups: (\d+)")
    brier, logloss, auc = grab(r"Calibrated forest, test set: \{'brier': ([\d.]+), 'log_loss': ([\d.]+), 'roc_auc': ([\d.]+)\}")
    return {"rows_clean": rows_clean, "groups": groups, "multi_groups": multi, "conflict_groups": conflict,
            "conflict_rows": conflict_rows, "train_rows": train, "test_rows": test,
            "phish_share_train": share_tr, "phish_share_test": share_te, "shared_groups": shared,
            "cal_test_brier": f"{float(brier):.4f}", "cal_test_logloss": f"{float(logloss):.4f}",
            "cal_test_auc": f"{float(auc):.4f}"}


def dataset_numbers():
    raw = data.load_raw()
    clean = data.clean(raw)
    y = data.get_target(clean)
    return {"raw_rows": n(len(raw)), "raw_cols": str(raw.shape[1]),
            "missing_values": n(int(raw.isna().sum().sum())),
            "repeated_urls_removed": n(len(raw) - len(clean)),
            "clean_rows_check": n(len(clean)),
            "clean_phishing": n(int(y.sum())), "clean_legit": n(int((1 - y).sum()))}


def model_numbers():
    o = pd.read_csv(R / "official_comparison.csv").set_index("model")
    keys = {"Logistic Regression": "lr", "Decision Tree": "dt", "Random Forest": "rf"}
    out = {}
    for name, k in keys.items():
        r = o.loc[name]
        out[k] = {
            "cv_f1": f4(r.cv_f1_mean), "cv_f1_std": f4(r.cv_f1_std), "cv_auc": f4(r.cv_roc_auc_mean),
            "cv_precision": f4(r.cv_precision_mean), "cv_recall": f4(r.cv_recall_mean),
            "test_acc": f4(r.test_accuracy), "test_precision": f4(r.test_precision),
            "test_recall": f4(r.test_recall), "test_f1": f4(r.test_f1), "test_auc": f4(r.test_roc_auc),
            "tn": n(r.test_tn), "fp": n(r.test_fp), "fn": n(r.test_fn), "tp": n(r.test_tp),
            "test_f1_raw": float(r.test_f1), "cv_f1_raw": float(r.cv_f1_mean),
        }
    b = pd.read_csv(R / "format_baseline.csv").iloc[0]
    out["rule"] = {"cv_f1": f4(b.cv_f1_mean), "cv_f1_std": f4(b.cv_f1_std), "test_acc": f4(b.test_accuracy),
                   "test_precision": f4(b.test_precision), "test_recall": f4(b.test_recall),
                   "test_f1": f4(b.test_f1), "tn": n(b.test_tn), "fp": n(b.test_fp), "fn": n(b.test_fn),
                   "tp": n(b.test_tp), "test_f1_raw": float(b.test_f1),
                   "legit_bare_share_train": pct(b.share_legitimate_bare_train, 0),
                   "phish_bare_share_train": pct(b.share_phishing_bare_train, 1)}
    out["lr_minus_rule_test_f1"] = f4(out["lr"]["test_f1_raw"] - out["rule"]["test_f1_raw"])
    out["rf_minus_rule_test_f1"] = f4(out["rf"]["test_f1_raw"] - out["rule"]["test_f1_raw"])
    out["dt_minus_rule_test_f1"] = f4(out["dt"]["test_f1_raw"] - out["rule"]["test_f1_raw"])
    return out


def format_controlled():
    fc = pd.read_csv(R / "format_controlled.csv")
    bare = fc[fc.subset == "bare https://www homepage"].set_index("model")
    other = fc[fc.subset == "other URLs"].set_index("model")
    keys = {"format rule (reference)": "rule", "Logistic Regression": "lr", "Decision Tree": "dt",
            "Random Forest": "rf"}
    out = {"bare_rows": n(bare.iloc[0].n_rows), "bare_phishing": n(bare.iloc[0].n_phishing),
           "bare_legit": n(bare.iloc[0].n_legitimate), "other_rows": n(other.iloc[0].n_rows)}
    for name, k in keys.items():
        out[k] = {"bare_tp": n(bare.loc[name].tp), "bare_fp": n(bare.loc[name].fp), "bare_fn": n(bare.loc[name].fn),
                  "bare_recall": f"{bare.loc[name].phishing_recall:.2f}",
                  "other_fn": n(other.loc[name].fn)}
    return out


def feature_sets():
    cv = pd.read_csv(R / "cv_results.csv")
    piv = cv.pivot(index="model", columns="feature_set", values="f1_mean")
    std = cv.pivot(index="model", columns="feature_set", values="f1_std")
    res = pd.read_csv(R / "results.csv")
    out = {}
    for name, k in {"Logistic Regression": "lr", "Decision Tree": "dt", "Random Forest": "rf"}.items():
        out[k] = {s: f4(piv.loc[name, s]) for s in ("url_base", "url_engineered", "extended")}
        out[k]["eng_minus_base"] = f4(piv.loc[name, "url_engineered"] - piv.loc[name, "url_base"])
        out[k]["eng_std"] = f4(std.loc[name, "url_engineered"])
    n_feat = res.drop_duplicates("feature_set").set_index("feature_set")["n_features"]
    out["n_features"] = {s: str(int(v)) for s, v in n_feat.items()}
    svm = res[(res.model == "SVM") & (res.feature_set == "url_engineered")].iloc[0]
    out["svm_test_f1"] = f4(svm.f1)
    out["svm_protocol"] = svm.protocol
    m = pd.read_csv(T / "feature_selection_margin.csv")
    removed = m[m.kind.str.startswith("removed")].iloc[0]
    out["removed_feature"] = removed.column
    out["removed_r"] = f"{removed.abs_r:.3f}"
    lr_diff = piv.loc["Logistic Regression", "url_base"] - piv.loc["Logistic Regression", "url_engineered"]
    out["lr_base_gain_in_fold_std"] = f"{lr_diff / std.loc['Logistic Regression', 'url_base']:.1f}"
    out["n_page_features"] = str(int(n_feat["extended"]) - int(n_feat["url_engineered"]) - 1)  # minus IsHTTPS
    return out


def demo_example():
    """First example of the saved full-data demo output (score, band, labels, top SHAP), without its URL."""
    text = (R / "demo_full_2026-11-24.txt").read_text(encoding="utf-8")
    block = text.split("URL: ")[1]
    m = re.search(r"risk score ([\d.]+)\s+->\s+band (\w+)\s+\|\s+predicted (\w+)\s+\|\s+true label (\w+)", block)
    bare = re.search(r"legitimate-URL format\): (\w+)", block).group(1)
    contrib = re.findall(r"^\s*(\w+)\s+([\d.]+)\s+(-?[\d.]+)\s+(phishing|legitimate)\s*$", block, re.M)[:3]
    return {"score": m.group(1), "band": m.group(2), "predicted": m.group(3), "truth": m.group(4),
            "bare_homepage": bare, "top": [{"feature": f, "value": v, "contribution": c} for f, v, c, _ in contrib],
            "source": "reports/demo_full_2026-11-24.txt"}


def constants():
    """Protocol constants straight from the code, so the documents cannot drift from it."""
    from src import features, risk
    from src.models import forest
    return {"seed": str(config.RANDOM_STATE), "cv_folds": str(config.CV_FOLDS),
            "test_size_pct": f"{100 * config.TEST_SIZE:.0f} %",
            "corr_threshold": f"{features.CORRELATION_THRESHOLD:.2f}",
            "low_max": f"{risk.LOW_MAX:.2f}", "high_min": f"{risk.HIGH_MIN:.2f}",
            "calibration": risk.CALIBRATION, "rf_subset_rows": n(forest.SUBSET_ROWS)}


def tuning():
    lr = pd.read_csv(T / "lr_followup_search.csv")
    best = lr[lr.chosen].iloc[0]
    c300 = lr[lr.C == 300].iloc[0]
    dt = pd.read_csv(T / "dt_complexity.csv")
    d20 = dt[(dt.max_depth == 20) & (dt.min_samples_leaf == 1) & dt.class_weight.isna()].iloc[0]
    unl = dt[dt.max_depth.isna() & (dt.min_samples_leaf == 1) & dt.class_weight.isna()].iloc[0]
    rf = pd.read_csv(T / "rf_subset_search.csv")
    rfc = rf[rf.chosen].iloc[0]
    summ = json.loads((T / "tuning_summary.json").read_text(encoding="utf-8"))
    return {
        "lr_C": f"{best.C:g}", "lr_n_candidates": str(len(lr)), "lr_n_fits": str(int(lr.n_folds.sum())),
        "lr_all_converged": bool(lr.valid.all()),
        "lr_f1": f5(best.f1_mean), "lr_f1_std": f5(best.f1_std), "lr_se": f5(best.f1_se),
        "lr_threshold": f5(best.f1_mean - best.f1_se), "lr_c300_f1": f5(c300.f1_mean),
        "lr_cmin": f"{lr.C.min():g}", "lr_cmax": f"{lr.C.max():g}",
        "dt_depth": str(int(d20.max_depth)), "dt_f1": f5(d20.f1_mean), "dt_f1_std": f5(d20.f1_std),
        "dt_leaves": n(round(d20.leaves_mean)), "dt_unl_f1": f5(unl.f1_mean), "dt_unl_leaves": n(round(unl.leaves_mean)),
        "dt_unl_depth": str(int(round(unl.depth_mean))), "dt_grid_candidates": str(summ["dt"]["n_candidates"]),
        "rf_trees": str(int(rfc.n_estimators)), "rf_depth": str(int(rfc.max_depth)),
        "rf_leaf": str(int(rfc.min_samples_leaf)), "rf_max_features": str(rfc.max_features),
        "rf_n_candidates": str(len(rf)), "rf_subset_f1": f5(rfc.f1_mean), "rf_subset_f1_std": f5(rfc.f1_std),
        "fs_folds_checked": str(summ["fs"]["folds_x_sets_checked"]),
        "fs_all_same": bool(summ["fs"]["all_same_as_outer"]),
    }


def risk():
    b = pd.read_csv(R / "risk_bands.csv").set_index("band")
    total_phish = int(b.n_phishing.sum())
    out = {"total_phishing_test": n(total_phish)}
    for band in ("Low", "Medium", "High"):
        r = b.loc[band]
        out[band.lower()] = {"n_urls": n(r.n_urls), "share": pct(r.share_of_urls, 1 if r.share_of_urls < 0.01 else 0),
                             "phishing_rate": pct(r.phishing_rate, 1), "n_phishing": n(r.n_phishing),
                             "n_legit": n(r.n_legit)}
    out["low_phishing_share_of_phishing"] = pct(b.loc["Low", "n_phishing"] / total_phish, 1)
    shap = pd.read_csv(R / "shap_importance.csv")
    out["shap_top"] = shap.feature.head(6).tolist()
    return out


def main():
    numbers = {"source_log": LOG.name, **log_numbers(), **dataset_numbers()}
    if numbers["clean_rows_check"] != numbers["rows_clean"]:
        raise SystemExit(f"cleaned rows differ: dataset {numbers['clean_rows_check']} vs log {numbers['rows_clean']}")
    numbers["models"] = model_numbers()
    numbers["format_controlled"] = format_controlled()
    numbers["feature_sets"] = feature_sets()
    numbers["tuning"] = tuning()
    numbers["risk"] = risk()
    numbers["constants"] = constants()
    numbers["demo"] = demo_example()
    OUT.write_text(json.dumps(numbers, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
