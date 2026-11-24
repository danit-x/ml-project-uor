# URLSentra: Explainable Phishing URL Risk Scoring — Progress Report (draft)

EC5203 Machine Learning mini project · Group 42 · University of Ruhuna · draft of 2026-11-24

> Draft for the two-page progress document. Every number below is copied from a file in `reports/`
> (named in brackets) that the committed code regenerates. Check the wording against the approved
> proposal before submission; section numbers (4.x) are those cited in `src/`.

## 1. Problem and motivation

Phishing sites imitate trusted services to steal credentials and payment data. A user or a mail/browser filter
often has to decide whether a link is safe *before* opening it, and a bare "phishing / legitimate" label is not
enough: people need to know **how risky** a URL is and **why**. URLSentra trains classical ML models on URL
features, turns the phishing probability into a calibrated **risk score** with Low / Medium / High bands, and
explains each score with SHAP. The main scientific goal of this stage is a **defensible evaluation**, not the
highest possible accuracy.

## 2. Dataset and preprocessing (4.2–4.5)

PhiUSIIL Phishing URL Dataset (UCI #967): 235,795 rows, 56 columns, no missing values. Raw label 0 = phishing is
flipped so that **1 = phishing** (positive class for precision, recall, F1 and the risk score).
`data.clean()` removes 425 repeated URLs (keep first; the copies had the same label but different feature
values) → **235,370 rows** (100,520 phishing, 134,850 legitimate). No imputation with dataset-wide statistics;
scaling is fitted inside each model pipeline.

## 3. Near-duplicate URLs and leakage prevention (4.8)

Near-duplicate URLs (e.g. `http://a.com/x/` vs `https://www.A.com/x`) would let a model memorise a URL in training
and "recognise" it in the test set. `data.url_group_key()` removes the scheme and one leading `www.`, lower-cases
the hostname and strips trailing `/` from the path, but **keeps path and query case** (they can address different
pages); the original URL is kept for the features. Rows with the same key form a **URL group**
(231,779 groups; 3,287 contain more than one row; 43 groups (86 rows) carry conflicting labels). The train/test split is one fold of `StratifiedGroupKFold(5, shuffle, seed 42)`: 188,297 training
/ 47,073 test rows, **0 groups shared**, phishing share 0.427 in both. All training CV uses the same five grouped,
stratified, seeded folds (`evaluate.cv_folds`). Label-conflict groups are kept unchanged as label noise; they
cannot leak because a group never crosses the split. The test rows are used once, for the final report.

## 4. Feature engineering and feature sets (4.6–4.7)

`URLSimilarityIndex` is excluded from every set: it is 100 for every legitimate URL (similarity to the list the
legitimate URLs came from), a label proxy (one-line rule F1 ≈ 0.996). Feature sets, chosen on training rows only:

| Set | Content | Role |
|---|---|---|
| `url_base` (20) | URL-string features, without `URLSimilarityIndex` and `IsHTTPS` | control |
| `url_engineered` (23) | `url_base` + `url_path_depth`, `url_longest_path_segment`, `url_query_field_count` | **primary** |
| `extended` (52) | `url_engineered` + `IsHTTPS` + 28 page-content features | secondary only |

Engineered features are row-wise functions of the URL string (no cross-row statistics, so computing them before the
split leaks nothing). A redundancy filter (|Pearson r| ≥ 0.98, keep the first) removed `url_encoded_byte_count`
(r = 0.994 with `NoOfAmpersandInURL` on training rows). The filter is refitted **inside every CV fold**
(`features.FeatureSetSelector`); on the full training split all 15 (fold, set) selections equal the selection on
all training rows (`reports/tuning/feature_selection_folds.csv`).

## 5. Models and tuning (4.9)

All tuning uses training rows only, `url_engineered`, the shared grouped folds and phishing F1, with a
**one-standard-error rule fixed before each run** (simplest candidate within one SE of the best).

| Model | Search | Frozen configuration | Grouped CV evidence |
|---|---|---|---|
| Logistic Regression (scaled) | L2/lbfgs C 0.001–1000, full training split, 50 fits, all converged. A wider grid with L1/liblinear was stopped after 36/45 fits (L1 C = 0.1 did not converge) | **C = 1000**, L2, lbfgs | F1 0.99674 ± 0.00026 (SE 0.00013, one-SE threshold 0.99661; only C = 1000 meets it; C = 300, at 0.99658, falls just below) — C is at the **upper edge**; gains from C = 30 on are ≈ 1 SE per step |
| Decision Tree | 24-point grid + depth curve, full training split, plus a complexity review | **max_depth 20**, min leaf 1 | F1 0.99053 ± 0.00042, 1,299 leaves (unlimited tree: 0.99008, 1,564 leaves) |
| Random Forest | 8-candidate bounded grid on a **50,000-row group-preserving training subset** | **300 trees, max_depth 20, min leaf 1, max_features 0.5** | subset CV F1 0.99119 ± 0.00181 (subset, not full data) |
| SVM (supplementary) | earlier ungrouped 2,000-row run | C = 0.1 | **provisional**, reported separately |

## 6. Evaluation protocol and results (4.10, 4.12)

Grouped 5-fold CV on the 188,297 training rows (identical folds for all models), then one held-out evaluation of the
frozen models on the same 47,073 test rows. Metrics: accuracy, precision, recall, F1, ROC-AUC, TN/FP/FN/TP.

| Model (`url_engineered`) | Grouped CV F1 (mean ± std) | CV ROC-AUC | Test accuracy | Test precision | Test recall | Test F1 | Test ROC-AUC | TN / FP / FN / TP |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.9967 ± 0.0003 | 0.9986 | 0.9977 | 0.9999 | 0.9946 | **0.9973** | 0.9985 | 26,967 / 2 / 108 / 19,996 |
| Decision Tree | 0.9905 ± 0.0004 | 0.9908 | 0.9929 | 0.9933 | 0.9902 | **0.9917** | 0.9920 | 26,834 / 135 / 197 / 19,907 |
| Random Forest | 0.9950 ± 0.0002 | 0.9989 | 0.9958 | 0.9963 | 0.9938 | **0.9950** | 0.9988 | 26,894 / 75 / 125 / 19,979 |
| *Format rule (no learning)* | *0.9948 ± 0.0004* | — | *0.9956* | *1.0000* | *0.9896* | ***0.9948*** | — | *26,969 / 0 / 209 / 19,895* |

Secondary (grouped CV F1, `url_base` / `url_engineered` / `extended`): LR 0.9972 / 0.9967 / 0.9991, DT 0.9905 / 0.9905 /
0.9988, RF 0.9950 / 0.9950 / 0.9997. **The engineered features did not improve CV**: no difference for DT and RF, and for LR
`url_base` is 0.0005 higher (about 1.5 fold std). **The engineered features produced no measurable improvement.** We keep
`url_engineered` as the primary set because it was fixed as primary before these results were seen; switching to the
set that happened to score higher would be selection on results, and the difference is within fold-to-fold noise. `extended` reaches 0.999+ only
through the collection-biased `IsHTTPS` and page features. SVM (provisional): test F1 0.9838 on `url_engineered`.
Sources: `reports/official_comparison.csv`, `reports/cv_results.csv`, `reports/results.csv`, `reports/format_baseline.csv`
(tables regenerated by `docs/progress_evaluation/results_tables.py`).

## 7. Main limitation: dataset collection bias (4.14)

**Every legitimate PhiUSIIL URL is a bare `https://www.<domain>` homepage** (no path, query or fragment); only about
1 % of phishing URLs are. The fixed rule "not a bare https://www homepage → phishing", which learns nothing, already
reaches grouped-CV F1 0.9948 ± 0.0004 and held-out F1 0.9948 (precision 1.000, 209 missed phishing URLs) (`reports/format_baseline.csv`). Logistic Regression beats it by only 0.0025 F1 on the test rows, the Random Forest merely matches it and the Decision Tree is below it. The URL-only models therefore mostly learn the **collection format**, not general
phishing cues. On the 27,178 held-out bare-homepage URLs, where the format cue is absent (26,969 legitimate, 209 phishing), the rule finds no phishing, while LR finds 102 (2 false alarms), DT 106 (135) and RF 104 (75) (`reports/format_controlled.csv`): the models learned some signal beyond the format, but on these hard cases they catch only about half of the phishing URLs, and 107 of LR's 108 missed phishing URLs are of this kind.
Consequences: (1) the near-perfect scores must not be read as real-world performance; (2) the models are not shown to
generalise to URLs collected differently (e.g. legitimate deep links); (3) the `extended` set (IsHTTPS + page
content) scores even higher for the same reason (`IsHTTPS` = 1 for every legitimate URL; nearly empty phishing pages).
This is the **main limitation** of the project and cannot be fixed inside PhiUSIIL, because the dataset contains no
legitimate deep links. **Future work:** evaluate the frozen models on an independently collected dataset (legitimate
URLs with paths and queries, phishing from a different source and period) before any real-world claim.
Other limitations: LR's C is at the edge of the searched range; RF was tuned on a training subset; SVM is
provisional.

## 8. Risk scoring and explainability (4.11, 4.13)

The served score is the Random Forest probability, sigmoid-calibrated with grouped folds on training rows only.
Sigmoid was re-checked on a 20,000-row development sample (lowest out-of-fold Brier score and log loss); in that check
the internal calibration CV folds were ungrouped, and it was not repeated on the full training split.
Bands: Low < 0.30 ≤ Medium < 0.70 ≤ High. On the held-out rows the calibrated score has Brier 0.0038
(`reports/main_full_run_2026-11-24.log`); Low holds 57 % of URLs (0.4 % phishing), Medium 0.2 % (40 % phishing), High 43 % (99.7 % phishing) (`reports/risk_bands.csv`).
**Disclosure: 108 of the 20,104 held-out phishing URLs are placed in the Low band** (and 61 legitimate URLs in High), so
"Low" means low estimated risk, not "safe". (This 108 is a calibrated-forest band count, not LR's 108 false negatives in §6: the counts are equal by
coincidence and the two sets only partly overlap.)
Calibration and bands are a project-owner decision for this evaluation; group confirmation is pending. SHAP (TreeExplainer) gives the top features of each score;
`python demo.py` shows score, band, prediction, true label and the top contributions for held-out test URLs only
(it refuses arbitrary URLs: several inputs come from the dataset authors' feature extractor).

## 9. Individual contributions

Original work by each member (their own commits, preserved as written). The 2026-11-24 integration and grouped-CV
retuning of all modules (URL grouping and group-aware split, feature-set freeze, LR/DT/RF retune, format baseline,
calibration rerun, SHAP, `demo.py`, tests) was done by Madushanka D.M.C and is not yet committed. Members have not yet
confirmed this table.

| Member | Model | Pipeline part | Evidence (commits) |
|---|---|---|---|
| Madushanka D.M.C | Logistic Regression | Data inspection, cleaning, first LR tuning, workflow documentation, SVM status note, branch integration | `e86f144`, `78ae724`, `6ceada1`, `3eefb14`, `0888f07`, `88804ba` (+ uncommitted 2026-11-24 integration, above) |
| Thanthrige D.P | Random Forest | Project scaffold and rules, leakage investigation (`URLSimilarityIndex`), RF tuning, risk scoring and calibration, SHAP explanations, `main.py` integration | `25e26f5`, `a74ec83`, `dc5e34a`, `87e91ca`, `752c23d`, `bd3e196`, `2569c27`, `b90a58c`, `736ee33`, `2593aec` |
| Madushan S.C | Decision Tree | URL feature engineering, feature selection, Decision Tree tuning | `7973c24` |
| Dharmasiri S.O.R | SVM | Common evaluation utilities (ROC, confusion matrices, feature-set comparison), calibrated linear SVM | `e6cbe11`, `4392061`, `643c94e` |

*Remaining work:* retune the SVM with grouped CV; final analysis and report. *Future work (beyond this project's
scope):* evaluation on an independently collected dataset to test generalisation beyond the PhiUSIIL format.
