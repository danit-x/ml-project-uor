# URLSentra: Explainable Machine Learning for Phishing URL Risk Scoring

EC5203 Machine Learning mini project, Group 42, University of Ruhuna.

We classify URLs as phishing or legitimate with four classical models, compare
them before and after feature engineering, turn the phishing probability into a
risk score, and explain the predictions.

> **Status for the progress evaluation (2026-11-24).** Older dated notes further down are kept as history;
> where they differ, this block and section 15 are current.
> - **Protocol:** near-duplicate URLs share a URL group (`data.url_group_key`); the train/test split is group-aware and
>   stratified, and every model is tuned and cross-validated on the **same 5 grouped folds** of the training rows only.
>   `URLSimilarityIndex` is excluded from every feature set. Primary feature set: `url_engineered` (23 URL-only
>   features); `url_base` and `extended` (+ `IsHTTPS` + page content) are secondary comparisons.
> - **Official models:** Logistic Regression (C = 1000, L2/lbfgs, tuned on the full training split; C is at the upper
>   edge of the searched range), Decision Tree (max_depth 20, complexity review), Random Forest (300 trees, depth 20,
>   max_features 0.5; bounded search on a 50,000-row group-preserving training subset). **SVM is provisional** (C from
>   an earlier ungrouped 2,000-row tuning) and reported separately.
> - **Results:** `python main.py` (full data) writes `reports/official_comparison.csv` (grouped CV + held-out test for
>   the three official models), `reports/results.csv` (all models and feature sets, with a `protocol` column),
>   `reports/format_baseline.csv` and `reports/format_controlled.csv` (collection-bias reference), and
>   `reports/risk_bands.csv`. Progress-evaluation drafts: `docs/progress_evaluation/`.
> - **Main limitation:** every legitimate PhiUSIIL URL is a bare `https://www.<domain>` homepage (about 1 % of phishing
>   URLs are), so URL-only scores near 0.99 largely measure the dataset's collection format. They are always shown next
>   to the fixed rule "not a bare https://www homepage → phishing", and we do **not** claim real-world generalisation.

**How to read status labels in this file**

| Label | Meaning |
|---|---|
| **Verified** | Checked against committed code or a re-run; the numbers can be regenerated. |
| **Provisional** | Produced from committed work, but it may change after pending group decisions (feature list, leakage, evaluation protocol), or it was run on a sample only. Do not present it as a final result. |
| **In progress** | Planned or started, but the implementation and its evaluation are not verified yet. |
| **Assigned** | The task has an owner. **Awaiting confirmation** means there are no commits or code changes for it in the repository yet. |
| **Pending** | Not started, or waiting for other work. |
| **Draft** | Wording that the group still has to check against the approved proposal. |

---

## Contents

1. [Project overview and problem statement](#1-project-overview-and-problem-statement)
2. [Project objectives](#2-project-objectives)
3. [Dataset and target label](#3-dataset-and-target-label)
4. [Technology stack](#4-technology-stack)
5. [Project folder structure](#5-project-folder-structure)
6. [Team members and responsibilities](#6-team-members-and-responsibilities)
7. [File ownership](#7-file-ownership)
8. [Python environment setup](#8-python-environment-setup)
9. [Dataset placement](#9-dataset-placement)
10. [Running the pipeline](#10-running-the-pipeline)
11. [Jupyter notebooks](#11-jupyter-notebooks)
12. [Model development and evaluation rules](#12-model-development-and-evaluation-rules)
13. [Git workflow](#13-git-workflow)
14. [Data leakage and known limitations](#14-data-leakage-and-known-limitations)
15. [Current progress](#15-current-progress)
16. [Progress evaluation preparation](#16-progress-evaluation-preparation)

---

## 1. Project overview and problem statement

> **Draft.** Sections 1 and 2 are draft wording based on the current project scope
> (the code, the notebooks and `CLAUDE.md`). They have **not** been checked against
> the approved group proposal, because the proposal is not in this repository.
> Check them against the proposal before you use them in the report or slides.

Phishing websites copy trusted sites to steal passwords, card details and other
personal data. A user or a security tool often has to decide whether a link is
safe **before** opening it. A simple "phishing / not phishing" answer is not very
helpful on its own: people also want to know **how risky** a URL is and **why**
the system thinks so.

**Problem:** given the features of a URL (and of the web page it points to), predict
whether the URL is phishing, give a risk score between 0 and 1, and explain which
features drove the decision.

**Approach:**

1. Clean and prepare the PhiUSIIL phishing URL dataset.
2. Train four classical models: Logistic Regression, Decision Tree, Random Forest
   and SVM.
3. Compare every model on the **original** features and on **engineered** features.
4. Use the predicted phishing probability as a risk score with Low / Medium / High bands.
5. Explain predictions with SHAP (tree models) and model coefficients (linear models).

---

## 2. Project objectives

**Draft** (see the note in section 1). The section numbers in brackets are the
ones cited in the `src/` docstrings. They have not been checked against the
proposal document.

1. Inspect the dataset and document its structure and class balance (4.2, 4.3).
2. Clean the data: duplicates, missing values, inconsistent labels (4.4).
3. Preprocess the data and create one shared, stratified, URL-group-aware train/test split (4.5, 4.8).
4. Engineer new URL-based features and select a useful feature subset (4.6, 4.7).
5. Train and tune Logistic Regression, Decision Tree, Random Forest and SVM (4.9).
6. Evaluate all models with the same metrics: accuracy, precision, recall, F1,
   ROC-AUC and confusion matrix (4.10).
7. Convert the phishing probability into a calibrated risk score with risk levels (4.11).
8. Compare results before and after feature engineering (4.12).
9. Explain model predictions globally and for single URLs (4.13).
10. Analyse the results, including limitations such as data leakage (4.14).

---

## 3. Dataset and target label

| Item | Value |
|---|---|
| Name | PhiUSIIL Phishing URL Dataset |
| Source | UCI Machine Learning Repository, dataset 967: https://archive.ics.uci.edu/dataset/967 |
| File | `PhiUSIIL_Phishing_URL_Dataset.csv` (about 57 MB) |
| Size (raw) | 235,795 rows, 56 columns, 0 missing values |
| Size (after `clean()`) | 235,370 rows: 100,520 phishing, 134,850 legitimate |
| Model input | 50 numeric "original" features |
| Not used as model input | `FILENAME`, `URL`, `Domain`, `TLD`, `Title` (`config.TEXT_COLUMNS`) and `label` |
| Shared split | 80 % train (188,297 rows) / 20 % test (47,073 rows): one fold of `StratifiedGroupKFold(5, shuffle=True, RANDOM_STATE = 42)` over normalized URL groups (changed 2026-11-24; previously 188,296 / 47,074 with a random stratified split) |

The row counts above are **Verified** (notebook 01, sections 2 and 3, and an
independent re-check on 2026-11-17).

The features fall into two groups:

- **URL features**, computed from the URL string, for example `URLLength`,
  `NoOfSubDomain`, `IsHTTPS`, `URLSimilarityIndex`.
- **Page features**, computed from the downloaded web page, for example
  `LineOfCode`, `NoOfImage`, `NoOfJS`, `HasPasswordField`.

### Target label definition

| | Raw dataset `label` | Our target (`data.get_target()`) |
|---|---|---|
| Phishing | 0 | **1** |
| Legitimate | 1 | **0** |

The raw label is flipped so that **1 = phishing**. Phishing is the positive class,
so precision, recall, F1, ROC-AUC and the risk score all refer to phishing.
Always get the target with `data.get_target(df)`; never use `df["label"]` directly.

---

## 4. Technology stack

| Purpose | Tool | Version (from `requirements.txt`) |
|---|---|---|
| Language | Python | 3.12 or newer (3.13.7 tested) |
| Data handling | pandas, NumPy | ~=3.1.1, ~=2.5.4 |
| Machine learning | scikit-learn | ~=1.9.1 |
| Explainability | SHAP | ~=0.54.1 |
| Plots | matplotlib (seaborn is installed but not used yet) | ~=3.11.2, ~=0.13.2 |
| Saving models | joblib | ~=1.6.0 |
| Notebooks | Jupyter | ~=1.1.1 |
| Clean notebook commits | nbstripout | ~=0.9.1 |
| Version control | Git and GitHub (`danit-x/ml-project-uor`) | |

Do not add a new library without group agreement. If one is needed, add it to
`requirements.txt` in a separate commit.

---

## 5. Project folder structure

```
ml-project-uor/
├── data/
│   ├── raw/                  dataset CSV goes here (not in git)
│   └── processed/            cleaned data, if saved (not in git)
├── notebooks/
│   ├── 01_data_logreg.ipynb           data cleaning + Logistic Regression
│   ├── 02_features_tree.ipynb         feature engineering + Decision Tree
│   ├── 03_evaluation_svm.ipynb        evaluation and comparison + SVM
│   └── 04_risk_explain_forest.ipynb   risk score, SHAP + Random Forest
├── src/
│   ├── config.py             shared paths and settings (seed, test size, CV folds)
│   ├── data.py               loading, cleaning, target, train/test split
│   ├── features.py           feature engineering and feature selection
│   ├── evaluate.py           shared metrics, CV, comparison table, plots
│   ├── risk.py               risk score, risk levels, calibration
│   ├── explain.py            SHAP explanations
│   └── models/
│       ├── __init__.py       MODELS dictionary used by main.py
│       ├── logreg.py         Logistic Regression
│       ├── tree.py           Decision Tree
│       ├── forest.py         Random Forest
│       └── svm.py            SVM
├── reports/
│   ├── figures/              saved plots (PNG)
│   ├── logreg_final_test_results.md   PROVISIONAL LR test report (see section 15)
│   └── *.csv                 result tables
├── saved_models/             fitted models (.joblib, not in git)
├── main.py                   runs the whole pipeline end to end
├── tune_models.py            grouped training-only tuning of LR, DT, RF (writes reports/tuning/)
├── requirements.txt          pinned Python packages
├── CLAUDE.md                 project rules for the Claude Code assistant
├── .gitignore                keeps data, models and sample results out of git
└── .gitattributes            strips notebook outputs on commit (nbstripout)
```

**What is (and is not) in git**

| Path | In git? | Why |
|---|---|---|
| `data/raw/*`, `data/processed/*` | No | The dataset is large and has its own licence; everyone downloads it. |
| `saved_models/*`, `*.joblib` | No | Large binary files; regenerate them with `main.py`. |
| `reports/results_sample_*.csv` | No | Quick sample runs are for checking only, not for reporting. |
| `reports/results.csv` | Yes, once it exists | The full-dataset results we report from. **Does not exist yet.** |
| `reports/figures/*.png` | Yes | Figures are generated by code and used in the report. |
| Notebook outputs | No | nbstripout removes them on commit. |

---

## 6. Team members and responsibilities

Everyone owns **one model** and **one pipeline part**. For your own model you
train it, tune its hyperparameters, evaluate it with all the shared metrics, and
run it on both the original and the engineered features.

| # | Member | Responsibility | Model | Pipeline part | Proposal sections (as cited in `src/`) |
|---|---|---|---|---|---|
| 1 | Madushanka D.M.C | Data Preprocessing + Logistic Regression | Logistic Regression | Dataset inspection, file integration, cleaning, preprocessing, train/test split | 4.2 to 4.5, 4.8 |
| 2 | Thanthrige D.P | Random Forest + Risk Scoring + Explainability | Random Forest | Risk scoring, calibration, explainability (SHAP), integration (`main.py`), final analysis | 4.11, 4.13, 4.14 |
| 3 | Madushan S.C | Feature Engineering + Decision Tree | Decision Tree | Feature engineering, feature selection | 4.6, 4.7 |
| 4 | Dharmasiri S.O.R | Evaluation + SVM | SVM | Common evaluation, model comparison, before/after feature engineering comparison | 4.10, 4.12 |

Members 3 and 4 committed their work on 2026-11-21 (Decision Tree: `7973c24`, Madushan S.C;
SVM and shared evaluation utilities: `e6cbe11`, `4392061`, `643c94e`, Dharmasiri S.O.R; merged as PR #7
`f022c97` and the integration merge `88804ba`). Since 2026-11-24, Madushanka D.M.C has been finishing the
remaining integration, retuning and evaluation work for all four parts on `integration/decision-tree-corrections`
(uncommitted working tree at the time of writing); the original commits keep their authors.

### Models for the progress evaluation

The **three official models** are Logistic Regression, Decision Tree and Random Forest. The SVM stays in the
project as a supplementary, **provisional** model.

| Model | Owner | Official for progress evaluation | Status | Frozen configuration and evidence |
|---|---|---|---|---|
| Logistic Regression | Member 1 | Yes | Tuned (grouped CV, full training split) | L2/lbfgs, C = 1000 (one-SE rule; **upper edge** of the searched range). `src/models/logreg.py`, `reports/tuning/lr_followup_search.csv`, nb01 §6 |
| Decision Tree | Member 3 | Yes | Tuned (grouped CV, full training split) + complexity review | max_depth 20, min_samples_leaf 1. `src/models/tree.py`, `reports/tuning/dt_complexity.csv`, nb02 |
| Random Forest | Member 2 | Yes | Tuned with a **bounded** grouped search on a 50,000-row group-preserving training subset | 300 trees, max_depth 20, min_samples_leaf 1, max_features 0.5. `src/models/forest.py`, `reports/tuning/rf_subset_search.csv`, nb04 Part A |
| SVM | Member 4 | No (supplementary) | **Provisional**: C = 0.1 from the earlier ungrouped 2,000-row tuning | `src/models/svm.py`, nb03; reported with `protocol = provisional` in `reports/results.csv` |

---

## 7. File ownership

Edit **only the files you own**. If you need a change in someone else's file, ask
the owner and tell them the file, the function and the exact change.

| Member | Files you own |
|---|---|
| Madushanka D.M.C | `src/data.py`, `src/models/logreg.py`, `notebooks/01_data_logreg.ipynb`, `reports/logreg_*` |
| Thanthrige D.P | `src/risk.py`, `src/explain.py`, `src/models/forest.py`, `main.py`, `notebooks/04_risk_explain_forest.ipynb` |
| Madushan S.C | `src/features.py`, `src/models/tree.py`, `notebooks/02_features_tree.ipynb` |
| Dharmasiri S.O.R | `src/evaluate.py`, `src/models/svm.py`, `notebooks/03_evaluation_svm.ipynb` |

**Shared files (no single owner):** `src/config.py`, `src/__init__.py`,
`src/models/__init__.py`, `requirements.txt`, `.gitignore`, `.gitattributes`,
`README.md`, `CLAUDE.md`. Change them only after the group agrees, and make the
change in its own commit. **`README.md` is shared:** every change to it goes
through a pull request that the group reviews.

**Interface contract.** `main.py` and other members' code call these names, so do
not rename them or change their arguments or return types:

- `src/data.py`: `load_raw()`, `clean()`, `load_clean_data()`, `get_target()`,
  `get_original_features()`, `split()`
- `src/features.py`: `ENGINEERED_FEATURES`, `add_features()`, `select_features()`
- `src/evaluate.py`: `CV_METRICS`, `cv_splitter()`, `cv_evaluate()`, `evaluate()`,
  `compare()`, `plot_confusion_matrix()`. You may **add** new functions.
- `src/models/*.py`: every model file has `build()` (unfitted model) and
  `train(X, y)` (fitted model). Fitted models must support `predict()` and
  `predict_proba()`.
- `src/risk.py`: `risk_score()`, `risk_level()`, `calibrate()`

---

## 8. Python environment setup

You need Python 3.12 or newer and Git.

```bash
# 1. Clone (HTTPS URL; use the SSH URL instead if you have set up an SSH key)
git clone https://github.com/danit-x/ml-project-uor.git
cd ml-project-uor

# 2. Create a virtual environment (once)
python -m venv .venv

# 3. Activate it (every new terminal)
.venv\Scripts\activate             # Windows Command Prompt
.venv\Scripts\Activate.ps1         # Windows PowerShell
source .venv/Scripts/activate      # Windows Git Bash
source .venv/bin/activate          # macOS / Linux

# 4. Install the packages
pip install -r requirements.txt

# 5. Turn on notebook output stripping (once per clone, required)
nbstripout --install
```

If PowerShell refuses to run `Activate.ps1`, run this once, then try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**Check your setup:**

```bash
python --version                                   # 3.12 or newer
python -c "import sklearn, pandas, shap; print(sklearn.__version__, pandas.__version__, shap.__version__)"
git config --get filter.nbstripout.clean           # must print a path; if empty, run nbstripout --install
```

In VS Code, select the `.venv` interpreter (Ctrl+Shift+P, "Python: Select
Interpreter") and pick the same environment as the notebook kernel.

---

## 9. Dataset placement

The dataset is **not** stored in the repository.

1. Download it from https://archive.ics.uci.edu/dataset/967.
2. Unzip it.
3. Put the CSV file here, with exactly this name:

   ```
   data/raw/PhiUSIIL_Phishing_URL_Dataset.csv
   ```

4. Check that git ignores it. `git status` must **not** list the CSV.

The path comes from `config.RAW_FILE`. Do not hard-code paths in your code. To use
a file in another location, pass it to the pipeline with `--data`:

```bash
python main.py --data path/to/PhiUSIIL_Phishing_URL_Dataset.csv
```

Never edit the raw CSV by hand. All cleaning happens in `data.clean()`, so every
member gets exactly the same data.

---

## 10. Running the pipeline

Run these from the repository root with the virtual environment activated.

| Command | What it does | Writes | Time |
|---|---|---|---|
| `python main.py --sample 20000` | Quick end-to-end check on a random 20,000-row sample | `reports/results_sample_20000*.csv` (not committed), `reports/figures/shap_summary_sample_20000.png` (do not commit), `saved_models/*.joblib` | about 80 seconds (about 10 s with `--explain-rows 0`), measured 2026-11-19 |
| `python main.py` | Full dataset, all four models, the three feature sets (official comparison = LR/DT/RF on `url_engineered`) | `reports/results.csv`, `reports/cv_results.csv`, `reports/official_comparison.csv`, `reports/format_baseline.csv`, `reports/format_controlled.csv`, `reports/risk_bands.csv`, `reports/risk_scores.csv`, `reports/shap_importance.csv`, `reports/figures/shap_summary.png`, `saved_models/*.joblib` (13 files) + `manifest.json` | about 53 min on an 8-core, 8 GB laptop (2026-11-24; grouped CV of the forests dominates; `--skip-cv` skips it) |
| `python main.py --data <file>` | Same as above with a different CSV path | as above | |
| `python tune_models.py` | Grouped 5-fold tuning of LR, DT (+ complexity review) and RF on the full training split, `url_engineered`; per-fold feature-selection check | `reports/tuning/*.csv`, `reports/tuning/tuning_summary.json` | see `seconds` in the summary JSON (RF search dominates) |
| `python tune_models.py --parts lr_followup rf_subset --time-budget 1500` | The bounded searches used for the progress evaluation: LR L2/lbfgs C 0.001–1000 on the full training split; RF 8 candidates on a 50,000-row group-preserving training subset. Each stops collecting results after the time budget and keeps partial results | `reports/tuning/lr_followup_*.csv`, `reports/tuning/rf_subset_search*.csv` | about 5 min each (measured 2026-11-24) |
| `python demo.py` | Risk score, band, prediction and top SHAP contributions for three held-out test URLs (`--url "<test URL>"` for one URL; arbitrary URLs are refused) | nothing (reads `saved_models/`) | about 1–2 min (data loading) |
| `python main.py --explain-rows N` | Explain the first N test URLs with SHAP (default 500; 0 skips risk scoring and SHAP) | as above | SHAP takes about 30 s per 1,000 URLs |

**What `main.py` does** (details in its module docstring):

1. Loads and cleans the data; builds URL groups and the group-aware split (0 groups shared by train and test).
2. Builds `url_base`, `url_engineered` (primary) and `extended` on **training rows only**
   (`URLSimilarityIndex` is in none).
3. Grouped 5-fold CV of LR, DT and RF on every set (identical folds; feature selection refitted per fold)
   → `reports/cv_results.csv`.
4. Fits every model's frozen `build()` on the training rows and evaluates it once on the held-out rows
   → `reports/results.csv` (with a `protocol` column) and `reports/official_comparison.csv`.
5. Scores the fixed format rule on the same folds and test rows (`reports/format_baseline.csv`) and splits the
   held-out results by URL format (`reports/format_controlled.csv`).
6. Calibrates the Random Forest on training rows (grouped folds, sigmoid) → `reports/risk_bands.csv`, SHAP for the
   first `--explain-rows` test URLs → `reports/shap_importance.csv`, `reports/risk_scores.csv`; saves the models and
   `saved_models/manifest.json` for `demo.py`.

**Always run the sample first.** If `--sample 20000` fails, fix it before trying
the full run.

> **Sample results are not official results.** A `--sample` run splits the
> 20,000-row sample into its own 16,000 / 4,000 train/test sets. That is **not**
> the official 47,073-row test set. Use sample runs only to check that the code
> works. Never put sample numbers in the report or slides as model performance.

> **The full run touches the test set.** `python main.py` evaluates on the
> official test split. Run it for final reporting only, after the feature list and
> hyperparameters are frozen. Do not change models or features because of what a
> full run shows (see [section 12](#12-model-development-and-evaluation-rules)).

---

## 11. Jupyter notebooks

Each member has one notebook. Use it for exploration, evidence and explanation.
Reusable code belongs in your `src/` file, and the notebook imports it.

**Start Jupyter** (from the repository root, with the venv active):

```bash
jupyter notebook notebooks/
```

Or open the notebook in VS Code and choose the `.venv` kernel.

**Important:**

- The notebooks start with `sys.path.append("..")`, so they must run with
  `notebooks/` as the working directory. Jupyter and VS Code do this by default
  when you open the notebook file.
- `SAMPLE = 20000` at the top: work on a sample while developing. Set
  `SAMPLE = None` only for final full-data runs. (Notebook 01 uses the full data
  for inspection and tuning regardless of `SAMPLE`.)
- `RUN_FINAL_TEST = False`: keeps the test set untouched. Set it to `True` only
  once, for the final report, after the group has frozen the features and
  hyperparameters. Notebooks 01 and 04 already have this guard. Notebooks 02 and
  03 do not have it yet, so **do not use "Run All" (or the `nbconvert` command
  below) on them** until their owners add it.
- In notebooks 02, 03 and 04 the train/test split is made from the `SAMPLE`
  rows. A final test-set run there needs `SAMPLE = None` as well; otherwise it
  scores a 4,000-row sample test set, not the official 47,074-row test set.
  (Notebook 01 already splits the full data for tuning and the final test.)
- Fill in the **Experiment log** table at the end of your notebook for every
  experiment: date, what changed, why, CV or test, key metrics, commit hash.

**Run a whole notebook from the terminal** (output goes to a temporary folder, so
the committed notebook stays clean):

```powershell
jupyter nbconvert --to notebook --execute notebooks\01_data_logreg.ipynb --output-dir $env:TEMP
```

Run it from the repository root. `nbconvert` runs the notebook with
`notebooks/` as its working directory, so `sys.path.append("..")` still works.
This is not a quick check: notebook 01 runs the full-training-split grid search
(section 6), which takes several minutes.

**Before you commit a notebook:** clear all outputs (Jupyter: Kernel, Restart &
Clear Output; VS Code: Clear All Outputs). nbstripout also does this
automatically if you ran `nbstripout --install`. Save figures you need with
`savefig()` into `reports/figures/` instead of relying on notebook outputs.

---

## 12. Model development and evaluation rules

These rules keep the four models comparable and the results honest.

**Shared settings.** `src/config.py` holds `RANDOM_STATE = 42`, `TEST_SIZE = 0.2`
and `CV_FOLDS = 5`. Always import them from `config`; never type a seed or path
in your code. Do not change these values without group agreement.

**One split for everyone.** Always split with `data.split(df, feature_cols)`. It
returns the same rows for any feature list, so all models use the same test set.

**Tune with cross-validation on the training set only.**

- Use `evaluate.cv_evaluate(module.build(), X_train, y_train)` to compare variants.
- Use `GridSearchCV` or `RandomizedSearchCV` with `cv=evaluate.cv_splitter()`
  on `X_train`, `y_train` only (see `logreg.tune()` and `forest.tune()`).
- Write the chosen hyperparameters as fixed values into `build()`, so `main.py`
  stays fast and reproducible.
- The SVM is slow on the full data: tune it on a training sample first.

**The test set is not for decisions.** Do **not** use the test set to:

- choose hyperparameters,
- choose or drop features (including `select_features()`),
- choose between models,
- set the risk-band thresholds or the calibration method.

Use it once at the end, with `evaluate.evaluate()`, after everything is frozen.
If you look at test results and then change something, the test score is no
longer an honest estimate.

**Fit everything on training data only.** Scalers, imputers, encoders, feature
selection and calibration must learn only from training data. Put them inside the
model's sklearn `Pipeline` in `build()` (see `logreg.py` and `svm.py`).
`add_features()` runs before the split, so each new feature must be computed from
its own row only, with no counts, averages or vocabularies over many rows.

**Evaluate the same way.** Use `evaluate.evaluate()` for every model and report
all its metrics (accuracy, precision, recall, F1, ROC-AUC, confusion matrix), not
accuracy alone. Compare `original` and `engineered` features for every model.

**Near-perfect scores are a warning.** If a metric is about 0.99 or higher, check
for leaking features before you report it (see section 14).

### Evaluation protocol

These steps happen in this order. Status updated 2026-11-24 for the progress evaluation.

| # | Step | Rule | Current state (2026-11-24) |
|---|---|---|---|
| 1 | Near-duplicate URL handling | Decide before any final run, because it changes everyone's data. Handle it in `clean()` or `split()` so all models see the same rows. | **Done 2026-11-24.** Rows are kept; `data.url_group_key()` groups them (scheme, `www.`, hostname case, trailing `/`; path and query kept) and `data.split()` / `evaluate.cv_folds()` keep each group on one side. 231,779 groups, 3,287 with several rows; 43 label-conflict groups (86 rows) kept as label noise (`data.conflicting_label_groups()`). |
| 2 | Domain-grouped vs random split | Decide one split for everyone. `data.split()` stays the only way to split. | **Group decision 2026-11-24:** group-aware split on the normalised URL (`url_group_key`). **Domain-level grouping is deferred for this progress evaluation, not permanently rejected:** it was not evaluated at this stage, and the remaining domain overlap between train and test (about 4 % of test URLs, measured on a 20k sample under the old random split) is a documented limitation to revisit for the final report. |
| 3 | Feature selection and feature freeze | Select features on training rows only (`select_features(df.loc[X_train.index], ...)`). Then freeze the feature list for both feature sets (`original`, `engineered`) before final tuning. | **Done 2026-11-24.** `features.build_feature_sets()` (training rows only) builds `url_base` (20), `url_engineered` (23, primary) and `extended` (52). `URLSimilarityIndex` is excluded from all (`features.EXCLUDED_FEATURES`); `url_encoded_byte_count` is removed as redundant (\|r\| 0.994 with `NoOfAmpersandInURL`). |
| 4 | Tuning and model selection on training data | Tune hyperparameters and compare models with cross-validation on the training split only. Re-tune after the feature freeze. | **Done 2026-11-24.** Grouped CV (`evaluate.cv_folds`) on `url_engineered`: LR C = 1000 (full training split, upper edge), DT max_depth 20 (full training split, complexity review), RF bounded search on a 50,000-row group-preserving subset. SVM provisional. |
| 5 | Final test-set usage | Run `evaluate.evaluate()` on the official 47,073-row test set **once per model**, after steps 1 to 4 are frozen. Nothing is changed after that. | **Done 2026-11-24** by `python main.py` (frozen `build()` of every model; `reports/official_comparison.csv`). The 2026-11-17 LR test report is superseded. |

**Feature and leakage decisions:** `URLSimilarityIndex` is dropped (group decision,
2026-11-19). `IsHTTPS` and the page features are excluded from the primary URL-only
sets and used only in the secondary `extended` set (group decision, 2026-11-24; see
section 14 and the decision table in section 15).

**Reproducibility.** Every number in the report must be regenerated by committed
code. Do not type metrics into slides or reports by hand from memory. Save tables
to `reports/` and figures to `reports/figures/` using `config.RESULTS_DIR` and
`config.FIGURES_DIR`.

**Explain your decisions.** For every ML choice, write in your notebook what you
did, why, what the alternative was and what the trade-off is. Every member must
be able to defend their part in the viva.

---

## 13. Git workflow

**Never push to `main` directly.** All work reaches `main` through a pull request.

### Branches

One branch per member and task, created from the latest `main`:

```bash
git checkout main
git pull origin main
git checkout -b feature/<yourname>-<task>      # e.g. feature/madushan-features-tree
```

Use the `feature/<name>-<task>` pattern for every new branch. (The early `dp`
branch was created before this convention and is already merged. There is no need
to rename it.)

### Daily work

```bash
git status                       # check what changed: only your own files?
git add src/features.py          # add files by name, not "git add ."
git commit -m "Add subdomain depth and path depth features"
git push -u origin feature/<yourname>-<task>
```

Commit rules:

- Small commits, one logical change each.
- Message in the imperative mood that says what and why, for example
  `Tune Decision Tree max_depth with 5-fold CV on train set`.
- Never commit: the dataset, `data/processed/`, `saved_models/`, `.joblib`
  files, large CSVs, notebook outputs or your `.venv/`.
- Check `git status` before every commit for anything unexpected.

### Keep your branch up to date

```bash
git checkout main
git pull origin main
git checkout feature/<yourname>-<task>
git merge main
```

### Pull requests

1. Push your branch, then open a pull request into `main` on GitHub.
2. In the description, write what you changed, why, and how you checked it.
3. Before you open it, confirm:
   - [ ] `python main.py --sample 20000` runs without errors
   - [ ] only your own files changed (plus shared files the group agreed on)
   - [ ] notebook outputs are cleared
   - [ ] no test-set results were used for decisions
   - [ ] the experiment log in your notebook is updated
4. **At least one other member** must review and approve the pull request.
   Do not merge your own pull request without that review.
5. The designated repository maintainer merges approved pull requests.
6. Delete the branch after the merge.

Changes to shared files (`README.md`, `CLAUDE.md`, `src/config.py`, ...) go in
their own pull request, and the group reviews them.

---

## 14. Data leakage and known limitations

**Every model currently reaches about 0.999 to 1.0 on all metrics (sample runs
and cross-validation). That is a warning sign, not a result.** Both investigations so far (notebook 01 section 7
and notebook 04 phase 2) used training rows only. The findings are below.

### Leakage and data quality findings

| Finding | Type | Evidence | Status |
|---|---|---|---|
| `URLSimilarityIndex` is exactly 100 for **every** legitimate URL (0.79 % of phishing URLs). The rule "below 100 means phishing" alone gives F1 about 0.996 on training rows. | Label leakage by construction: the feature measures similarity to the legitimate URL list the dataset was built from. | nb01 §7; nb04 phase 2 | **Dropped** (group decision, 2026-11-19), via `main.DROPPED_FEATURES` (both feature sets). |
| Removing `URLSimilarityIndex` hardly changes the scores (LR CV F1 0.99991 to 0.99965), because page features also separate the classes almost perfectly on their own (`LineOfCode`, `NoOfExternalRef`, `NoOfImage`, ... single-feature ROC-AUC 0.96 to 0.99). | Collection bias: phishing pages in the dataset are mostly bare or already taken down. | nb01 §7; nb04 phase 2 | Provisional (RF numbers from a 20k sample). **Group decision pending** on retaining or excluding affected page features. |
| `IsHTTPS` = 1 for every legitimate URL, and no legitimate URL has `?`, `=`, `&`, an IP address or obfuscation. | Collection bias: the legitimate class is mostly clean HTTPS homepages. | nb04 phase 2 | Report it; group decision pending |
| With **URL features only**, without `URLSimilarityIndex` and `IsHTTPS`, Random Forest CV F1 drops to about 0.96. This is closer to how hard the real task is. | Harder, more honest variant | nb04 ablation (20k sample) | Provisional |
| After the drop, the re-tuned Random Forest's SHAP values put about 74 % of the weight on page features (top: `LineOfCode`, `NoOfSelfRef`, `NoOfExternalRef`, `NoOfImage`, then `IsHTTPS`). LR relies most on `IsHTTPS` and URL character ratios instead. | Confirms the page-feature collection bias in the forest | nb04 phase 5 (20k sample) | Provisional |
| 425 URLs appear twice. The copies always have the same label but **different feature values** (for example `URLLength`). `clean()` keeps the first copy. | Data quality | nb01 §3 | Done (required by the project rules in `CLAUDE.md`) |
| 1,165 near-duplicate URL pairs (differ only by trailing `/`). 392 pairs fall on both sides of the train/test split. One pair has conflicting labels (`https://www.education.gouv.fr/` phishing vs `https://www.education.gouv.fr` legitimate). | Possible leakage and label noise | nb01 §4 | **Group decision 2026-11-24:** kept, grouped, never split across train/test or CV folds. |
| About 4 % of test URLs share a domain with a training URL. | Mild overlap | nb04 (20k sample, old random split) | **Deferred for the progress evaluation** (not permanently rejected): domain-level grouping to be revisited for the final report; documented limitation |

### How we protect against leakage in the code

- `clean()` removes repeated URLs so the same URL cannot be in train and test.
- `clean()` never fills missing values with dataset-wide statistics.
- Scaling (LR and SVM) and the SVM's probability calibration are inside the
  model's pipeline in `build()`, so they are fitted on training folds only.
  `risk.calibrate()` fits on training data with internal CV.
- `select_features()` receives training rows only (`main.py` passes
  `df.loc[X_train.index]`).
- All tuning uses cross-validation on the training split. The test set is kept
  for the final report.

### Known limitations

Current as of 2026-11-24 (progress evaluation). The superseded list of 2026-11-19 is in the git history.

- **URL-format collection bias (main limitation).** Every legitimate URL is a bare `https://www.<domain>` homepage;
  about 1 % of phishing URLs are. The fixed rule "not a bare https://www homepage → phishing" (no learning) scores
  close to the URL-only models (`reports/format_baseline.csv`, same grouped folds and test rows). The models therefore
  mostly learn the collection format; `reports/format_controlled.csv` shows how they do on bare homepages, where that
  cue is absent. No claim of real-world generalisation is made.
- **Page-content and `IsHTTPS` collection bias.** `IsHTTPS` = 1 for every legitimate URL and page counts
  (`LineOfCode`, `NoOfExternalRef`, ...) nearly separate the classes (phishing pages were often bare or taken down).
  They are used only in the secondary `extended` set, whose near-perfect scores are a warning, not a result.
- **Logistic Regression C at the edge.** C = 1000 is the largest value searched; F1 still rose slightly with C
  (about one SE per step from C = 30). L1 models were not evaluated (liblinear did not converge at C = 0.1).
- **Random Forest tuned on a subset.** The bounded search used a 50,000-row group-preserving subset of the training
  rows; only the chosen forest was trained and cross-validated on the full training split.
- **Decision Tree size.** The frozen tree (depth 20, about 1,300 leaves) cannot be read leaf by leaf; it is explained
  by its top levels and importances. A readable depth-4 tree loses about 0.17 F1.
- **SVM is provisional** (C from the earlier ungrouped 2,000-row tuning, not retuned with grouped CV).
- **Calibration check.** The served model, `risk.calibrate()`, uses the grouped folds. The method choice (sigmoid)
  was re-checked only on a 20,000-row development sample, and the nested comparison in `risk.calibration_report()`
  calibrates inside each grouped outer fold with the legacy **ungrouped** inner splitter. On the held-out rows, 108
  phishing URLs fall in the Low band and 61 legitimate URLs in High (`reports/risk_bands.csv`).
- **Domain-level grouping deferred.** Near-duplicate URLs are grouped, but URLs from the same domain can still be on
  both sides of the split (about 4 % of test URLs on an earlier 20k-sample check); deferred, not rejected.
- **Demo scope.** `demo.py` scores only held-out PhiUSIIL test URLs: several inputs (`TLDLegitimateProb`,
  `URLCharProb`, ...) come from the dataset authors' feature extractor, which this project does not reproduce.
- **Dataset scope.** One dataset from one collection period.

---

## 15. Current progress

**Current summary (2026-11-24, progress evaluation).** Done on `integration/decision-tree-corrections`
(uncommitted working tree): URL grouping + group-aware split + grouped CV; the three agreed feature sets with
per-fold feature selection; LR, DT and RF tuned with grouped CV and frozen (see section 6); full-data `main.py`
run with grouped CV and held-out test for all models; format reference rule and format-controlled breakdown;
grouped calibration, risk bands, SHAP and `demo.py`; notebooks 01–04 updated (old results kept as labelled
history); 64 unit tests (`python -m unittest discover -s tests -t .`). Remaining: SVM grouped retune; final
report. Future work: evaluation on an independently collected dataset (the URL-format collection bias is reported
as the main limitation). Progress-evaluation drafts: `docs/progress_evaluation/`.

The tables below are the history up to 2026-11-19 (last updated then); where they differ, the summary above
is current.

### Completed

| Task | Owner | Evidence | Status |
|---|---|---|---|
| Initial project scaffold: config, module stubs, notebooks, `main.py`, README | (initial scaffold) | commit `165eead` | Verified |
| Project rules and file ownership (`CLAUDE.md`) | Thanthrige D.P | commit `25e26f5` | Verified |
| Leakage and reproducibility safeguards in the scaffold (train-only feature selection, shared CV folds, separate sample-results file, nbstripout) | Thanthrige D.P | commit `a74ec83` | Verified |
| Dataset inspection: shape, types, missing values, class balance, duplicates | Madushanka D.M.C | nb01 §2 to §4 | Verified |
| Data cleaning (`clean()`, `check_target()`), target flip, shared split | Madushanka D.M.C | `src/data.py`, commit `e86f144` | Verified |
| Duplicate and near-duplicate URL reports | Madushanka D.M.C | `src/data.py`, nb01 §3 and §4 | Verified |
| `main.py --sample 20000` runs end to end with all four models, risk scoring and SHAP | (code check) | re-run on 2026-11-19 after PR #4 | Verified (as a code check only) |
| Push `feature/madushanka-data-logreg` and open a pull request | Madushanka D.M.C | PR #2 merged (`724370e`); PR #3 merged (`f808569`) | Verified |
| Drop `URLSimilarityIndex` from both feature sets (group decision) | Thanthrige D.P (`main.py`) | `main.DROPPED_FEATURES`, commit `2569c27`, PR #4 | Verified. The `original` set does not pass through `select_features()`, so `main.py` keeps this drop for it |

**Commit attribution for Member 2.** Commits with the git author name `danit-x`
(`25e26f5`, `a74ec83`, `dc5e34a`, `87e91ca`, `752c23d`, `bd3e196`, `2569c27`,
`b90a58c`) are credited to Thanthrige D.P. They
use the same author identity as the PR #1 merge commit `7242dc4`, which is authored
as Danith Thanthrige.

### Completed but provisional

| Task | Owner | Evidence | Why provisional |
|---|---|---|---|
| Logistic Regression tuned (GridSearchCV over C and L1/L2, 5-fold CV, full training split): L2, C = 10 | Madushanka D.M.C | `src/models/logreg.py`, nb01 §6 | Tuned with `URLSimilarityIndex`; re-tune if the feature list changes. Tuning gained almost nothing over the default. |
| `URLSimilarityIndex` evidence for LR (with/without) | Madushanka D.M.C | nb01 §7 | Waiting for the group decision |
| **PROVISIONAL:** Logistic Regression test-set report | Madushanka D.M.C | `reports/logreg_final_test_results.md` | See the note below. |
| Random Forest tuned (RandomizedSearchCV, 20 candidates) | Thanthrige D.P | `src/models/forest.py`, nb04 phase 1; commit `dc5e34a`, merged in PR #1 (`7242dc4`) | Superseded by the phase 3 re-tune below |
| Random Forest leakage check: single-feature AUC, class-constant features, ablation, domain overlap | Thanthrige D.P | nb04 phase 2, `reports/figures/rf_single_feature_auc.png`; commit `87e91ca`, merged in PR #1 (`7242dc4`) | 20k sample; `IsHTTPS`, page features and domain split still undecided |
| Random Forest re-tuned without `URLSimilarityIndex`: 500 trees, depth 20, leaf 1, sqrt (CV F1 0.99956; URL-only 0.99576) | Thanthrige D.P | `src/models/forest.py`, nb04 phase 3; commit `2569c27`, PR #4 (`c372c82`) | 20k sample; re-check on the full training set after the feature freeze |
| Risk score calibration and bands: sigmoid chosen out-of-fold (Brier 0.00131 to 0.00043); 0.30 / 0.70 bands checked with `band_table()` | Thanthrige D.P | `src/risk.py`, nb04 phase 4, `reports/figures/rf_calibration_curve.png`; commits `752c23d`, `b90a58c`, PR #4 | 20k sample; bands and the calibration trade-off need group confirmation |
| Explainability: SHAP global and single-URL explanations for RF, coefficient contributions for LR | Thanthrige D.P | `src/explain.py`, nb04 phase 5, `reports/figures/rf_shap_summary.png`; commits `bd3e196`, `b90a58c`, PR #4 | 20k sample; LR view waits for LR re-tune; DT and SVM not shown yet |
| Risk scoring and SHAP connected to `main.py` (`risk_and_explain()`, Brier column, `--explain-rows`) | Thanthrige D.P | `main.py`, commit `2569c27`, PR #4 | Only checked with `--sample 20000` |

> **`reports/logreg_final_test_results.md` is PROVISIONAL.** Despite the word
> "Final" in its name, it is **not** an official final project result:
>
> - It was made before the feature list and the evaluation protocol (near-duplicate
>   handling, domain-grouped split) were decided.
> - Its metrics were not written by committed code. The final-test cell in nb01 is
>   switched off (`RUN_FINAL_TEST = False`).
>
> It **must be regenerated** from committed code once the feature and
> evaluation-protocol decisions are final. Until then, do not quote its numbers
> in the report or slides.
>
> The report file itself does **not** carry a PROVISIONAL warning yet. Adding one
> is a pending task (below) for a separate, approved change.

### In progress and assigned

| Task | Owner | Status | Progress | Notes |
|---|---|---|---|---|
| Decision Tree: tuning with shared CV folds, then evaluation | Madushan S.C | Initial scaffold exists; Member 3 implementation, tuning and evaluation pending | Awaiting confirmation | Baseline is the untuned scaffold default in `src/models/tree.py` (commit `165eead`). No tuning commits yet. |
| Feature engineering (row-wise URL features) and feature selection on training rows | Madushan S.C | Assigned | Awaiting confirmation | `src/features.py` holds only the two example features from the scaffold. |
| SVM: tuning (C, on a training sample first), then evaluation | Dharmasiri S.O.R | Initial scaffold exists; Member 4 implementation, tuning and evaluation pending | Awaiting confirmation | Baseline is the untuned scaffold default in `src/models/svm.py` (commit `165eead`). No tuning commits yet. |
| Common evaluation: CV comparison table, ROC curves, before/after chart, interpretation | Dharmasiri S.O.R | Assigned | Awaiting confirmation | `src/evaluate.py` has the shared metrics and CV helpers from the scaffold only. |
| Move the `URLSimilarityIndex` drop into `select_features()` | Madushan S.C | Requested | Awaiting confirmation | Covers the `engineered` set only; `main.DROPPED_FEATURES` stays for the `original` set unless the group agrees a shared constant in `config.py`. |
| Re-tune LR without `URLSimilarityIndex` | Madushanka D.M.C | Requested | Awaiting confirmation | `BEST_C` / `BEST_L1_RATIO` in `logreg.py` were chosen with it. |

### Pending

| Task | Owner | Depends on |
|---|---|---|
| Group decisions: `IsHTTPS`, near-duplicate URLs, domain-grouped split, retain or exclude biased page features, risk bands and calibration trade-off | Whole group | |
| Add a clear PROVISIONAL warning at the top of `reports/logreg_final_test_results.md` | Madushanka D.M.C | A separate change, approved by the group (not part of this README revision) |
| Re-run RF tuning, calibration check and SHAP on the full training set (nb04 with `SAMPLE = None`) | Thanthrige D.P | Feature freeze |
| Decision Tree and SVM explanations in nb04 | Thanthrige D.P | Tuned DT and SVM |
| Re-run LR and RF on the agreed feature list (original and engineered) | Madushanka D.M.C, Thanthrige D.P | Member 3's merged `features.py` |
| Regenerate the LR test report from committed code | Madushanka D.M.C | Feature and evaluation-protocol decisions |
| Full-dataset `python main.py` run, `reports/results.csv` committed | Thanthrige D.P (`main.py` owner) | All models tuned, features frozen |
| Final test-set evaluation, once per model | Each model owner | Full run |
| Final analysis and report (4.14) | Whole group | Everything above |

### Pending group decisions

**Group decision** rows were agreed by the group. **Project-owner decision** rows were taken by Madushanka D.M.C on
2026-11-24 for the progress evaluation and still need group confirmation. **Evidence-based finding** rows record a
measured fact, not a vote. **Deferred** rows stay open.

| Decision | Options under discussion | Affects | Status |
|---|---|---|---|
| `URLSimilarityIndex` | keep / drop in `select_features()` / report both | everyone's results | **Group decision 2026-11-19: drop.** Applied in `main.DROPPED_FEATURES` for both feature sets. |
| Risk bands and calibration | keep the Low / Medium / High bands (0.30 / 0.70; not in proposal 4.11) or report the score only; sigmoid-calibrated score (better Brier score, 4 phishing URLs in Low on the 20k sample) or raw score (no phishing in Low) | risk score, demo | **Project-owner decision 2026-11-24, group confirmation pending:** keep the sigmoid-calibrated score and the 0.30 / 0.70 bands. Sigmoid was re-checked on a 20,000-row development sample (outer folds grouped; the internal calibration CV folds of that check were ungrouped). Disclose that 108 held-out phishing URLs fall in Low and 61 legitimate URLs in High (`reports/risk_bands.csv`) |
| Near-duplicate URLs | (a) keep as now; (b) normalise only safe cases and deduplicate; (c) group-aware split; (d) drop only the conflicting pair (details in nb01 §4) | `clean()` or `split()`, everyone's data | **Group decision 2026-11-24: (c)** group-aware split and CV with the agreed normalization; conflicting groups kept |
| Domain-grouped splitting | keep the URL-group split / split grouped by domain | `split()`, everyone's data | **Deferred for this progress evaluation, not permanently rejected.** The current split groups near-duplicate URLs (group decision above); domain-level grouping is to be revisited for the final report |
| `IsHTTPS` | keep / drop / keep and also report a URL-only result | everyone's results | **Group decision 2026-11-24:** out of the primary URL-only sets; only in `extended` |
| Page features with collection bias (`LineOfCode`, `NoOfExternalRef`, `NoOfImage`, ...) | retain / exclude / retain and also report a URL-only result | everyone's results, feature freeze | **Group decision 2026-11-24:** out of the primary URL-only sets; only in `extended` |
| Engineered features | keep `url_engineered` as primary / switch to `url_base` | primary results | **Project-owner decision 2026-11-24, group confirmation pending:** keep `url_engineered` as primary; report that the engineered features gave no measurable improvement |
| URL-format collection bias | report as limitation / format-controlled evaluation / second dataset | interpretation of all URL-only results | **Evidence-based finding (not a group vote), 2026-11-24:** reported as the main limitation, shown with `reports/format_baseline.csv` and `format_controlled.csv`; independent-dataset evaluation is future work |

---

## 16. Progress evaluation preparation

### Status against the evaluation criteria

| Criterion | What we must show | Owner(s) | Status |
|---|---|---|---|
| Problem analysis | Problem statement, objectives, dataset description, class balance | Madushanka D.M.C (data), all (problem) | Done: README §1–3, nb01, `docs/progress_evaluation/` |
| Data preprocessing | Cleaning steps with evidence and reasons, shared split | Madushanka D.M.C | Done: cleaning, URL groups, group-aware split, grouped CV (nb01 §3–4b, `src/data.py`, tests) |
| Feature engineering | New features with justification, selection method, before/after effect | Madushan S.C | Done: 4 engineered URL features (3 kept), redundancy filter refitted per fold, `url_base` vs `url_engineered` comparison (nb02, `reports/cv_results.csv`) |
| Model selection and implementation | Why each model, tuning method and chosen hyperparameters | Each model owner | Done for LR, DT, RF (grouped CV, one-SE rule, see §6); SVM provisional |
| Model evaluation | Same metrics for all models, comparison table and plots, leakage discussion | Dharmasiri S.O.R (comparison), each owner (own model) | Done: `reports/official_comparison.csv`, `reports/results.csv`, format baseline and format-controlled breakdown |
| Demonstration | Live run of the pipeline and notebook sections | Thanthrige D.P (`main.py`), each owner | Ready: `python demo.py` (held-out URLs: score, band, SHAP), nb04 Part A |
| Viva | Every member explains and defends their own part | Everyone | Notes: `docs/progress_evaluation/presentation_outline.md` |
| Individual contribution | Visible commits and pull requests from every member | Everyone | Original commits from all four members are preserved (see the progress document). The 2026-11-24 integration and grouped-CV retuning across all modules was done by Madushanka D.M.C and is not yet committed. Members still have to confirm the contribution table |

### Evaluation logistics

The evaluation date is not specified in this README. Follow the latest official
course announcement for the date, time and format. The
presentation order and who presents which part are **not decided yet**. The group
agrees on them and records them here.

### Checklist before the evaluation

**Everyone**

- [ ] Your work is pushed and merged into `main` through a pull request.
- [ ] `python main.py --sample 20000` runs on a fresh clone of `main`.
- [ ] Your notebook explains every decision (what, why, alternative, trade-off)
      and has an up-to-date experiment log.
- [ ] Your notebook outputs are cleared; the figures you need are saved in
      `reports/figures/`.
- [ ] You can explain your own model and pipeline part without notes, including
      why the scores are near 1.0 and what we did about leakage.
- [ ] Every number you show comes from committed code and is labelled as CV,
      sample or test.

**Group**

- [ ] The pending group decisions in section 15 are made and written down.
- [ ] We confirm the proposed models (LR, RF, DT), and that the Decision Tree is
      verified in time to present.
- [ ] One shared leakage explanation that every member tells the same way.
- [ ] A demo plan agreed by the group: who shows what, in which order, with which
      command or notebook section.
- [ ] Slides use only Verified results, or label results clearly as Provisional.
      No sample results are presented as model performance.
- [ ] The numbers in `reports/logreg_final_test_results.md` are not shown as final
      results, and the file carries a PROVISIONAL warning (separate approved change).

### What we will not claim

- That feature engineering, risk scoring or explainability are finished before
  they are.
- That the Decision Tree or SVM is evaluated before its tuned code is committed
  and checked.
- That the group has decided any of the pending decisions in section 15.
- That the problem statement and objectives match the approved proposal before
  someone has checked them.
- That near-1.0 scores mean the models would work this well on real-world URLs.
- Any metric from a `--sample` run as model performance.

---

## Using Claude Code

If you use Claude Code, `CLAUDE.md` holds the project rules it follows. Tell it
which member you are at the start of every session, so it edits only your files.
