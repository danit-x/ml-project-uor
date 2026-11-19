# URLSentra: Explainable Machine Learning for Phishing URL Risk Scoring

EC5203 Machine Learning mini project, Group 42, University of Ruhuna.

We classify URLs as phishing or legitimate with four classical models, compare
them before and after feature engineering, turn the phishing probability into a
risk score, and explain the predictions.

> **Status on 2026-11-19:** work in progress. Data cleaning is done (group
> decisions on near-duplicate URLs and the split method are pending).
> `URLSimilarityIndex` is now dropped (group decision, label leakage). Logistic
> Regression and Random Forest are tuned, but their results are **provisional**
> (LR was tuned before the drop; RF was re-tuned without it on a 20k sample).
> Risk scoring (calibration, risk bands) and explainability (SHAP, LR coefficients,
> single-URL explanations) are implemented and run from `main.py` and nb04, on a
> 20k sample so far (**provisional**). Decision Tree: **initial scaffold exists;
> Member 3 implementation, tuning and evaluation pending**. SVM: **initial scaffold
> exists; Member 4 implementation, tuning and evaluation pending**. Feature
> engineering is **not finished yet**. Every model currently scores close to
> 1.0, which we treat as a warning sign of data leakage, **not** as a result.
> **There are no official final results yet.** See [Current progress](#15-current-progress)
> and [Data leakage and known limitations](#14-data-leakage-and-known-limitations).

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
3. Preprocess the data and create one shared, stratified train/test split (4.5, 4.8).
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
| Shared split | 80 % train (188,296 rows) / 20 % test (47,074 rows), stratified, `RANDOM_STATE = 42` |

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

Members 3 and 4: the work is **assigned**. Implementation progress is **awaiting
confirmation**, because the repository has no commits or code changes from them yet
(checked on 2026-11-18). Update this when their first commits are pushed.

### Models for the progress evaluation

All four models stay in the project. **Proposed** for the progress evaluation:
Logistic Regression, Random Forest and Decision Tree.

| Model | Owner | Proposed for progress evaluation | Status | Evidence |
|---|---|---|---|---|
| Logistic Regression | Member 1 | Yes | Provisional | Tuned in `src/models/logreg.py`, nb01 §6 (commit `e86f144`) |
| Random Forest | Member 2 | Yes | Provisional | Re-tuned without `URLSimilarityIndex` in `src/models/forest.py` (20k sample), nb04 phase 3 (commit `2569c27`, PR #4) |
| Decision Tree | Member 3 | Yes | Initial scaffold exists; Member 3 implementation, tuning and evaluation pending | `src/models/tree.py` holds the untuned scaffold baseline (commit `165eead`). No tuning commits yet. |
| SVM | Member 4 | No | Initial scaffold exists; Member 4 implementation, tuning and evaluation pending | `src/models/svm.py` holds the untuned scaffold baseline (commit `165eead`). No tuning commits yet. |

Decision Tree and SVM keep this status until their tuned implementation and
evaluation are committed and checked. The overall progress of Members 3 and 4
stays **Awaiting confirmation** until then.

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
| `python main.py` | Full dataset, all four models, original and engineered features | `reports/results.csv`, `reports/risk_scores.csv`, `reports/shap_importance.csv`, `reports/figures/shap_summary.png`, `saved_models/*.joblib` (8 files) | not measured yet; the SVM is the slowest |
| `python main.py --data <file>` | Same as above with a different CSV path | as above | |
| `python main.py --explain-rows N` | Explain the first N test URLs with SHAP (default 500; 0 skips risk scoring and SHAP) | as above | SHAP takes about 30 s per 1,000 URLs |

**What `main.py` does:**

1. Loads and cleans the data (`data.load_clean_data()`).
2. Drops the features in `main.DROPPED_FEATURES` (`URLSimilarityIndex`, group decision)
   and builds two feature sets: `original` (49 features) and `engineered`
   (`features.add_features()` then `features.select_features()` on **training rows only**).
3. Splits with `data.split()`, then trains every model in `src/models/__init__.py`.
4. Evaluates every model on the test split with `evaluate.evaluate()`, plus the
   Brier score of its phishing probability.
5. Saves the comparison table and the fitted models.
6. Risk scoring and explanations (`risk_and_explain()`, Random Forest on original
   features): calibrates the forest on training rows (`risk.calibrate()`, sigmoid),
   prints the test-set Brier score and risk-band table, and saves SHAP importance and
   per-URL risk scores with their top three features for the first `--explain-rows`
   test URLs.

**Always run the sample first.** If `--sample 20000` fails, fix it before trying
the full run.

> **Sample results are not official results.** A `--sample` run splits the
> 20,000-row sample into its own 16,000 / 4,000 train/test sets. That is **not**
> the official 47,074-row test set. Use sample runs only to check that the code
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

These steps happen in this order. The protocol is **not frozen yet**: several
parts wait for group decisions (section 15), and nothing below claims a decision
has been made.

| # | Step | Rule | Current state (checked 2026-11-18) |
|---|---|---|---|
| 1 | Near-duplicate URL handling | Decide before any final run, because it changes everyone's data. Handle it in `clean()` or `split()` so all models see the same rows. | **Unresolved.** 1,165 near-duplicate pairs are kept; 392 pairs cross the train/test split (nb01 §4). |
| 2 | Domain-grouped vs random split | Decide one split for everyone. `data.split()` stays the only way to split. | **Unresolved.** Current code uses a random stratified split; about 4 % of test URLs share a domain with a training URL (nb04, 20k sample). |
| 3 | Feature selection and feature freeze | Select features on training rows only (`select_features(df.loc[X_train.index], ...)`). Then freeze the feature list for both feature sets (`original`, `engineered`) before final tuning. | **Pending.** `URLSimilarityIndex` is dropped (decided 2026-11-19), applied in `main.DROPPED_FEATURES` until `select_features()` does it. `select_features()` still returns every feature. Feature freeze waits for Member 3's features and the decisions on `IsHTTPS` and page features. |
| 4 | Tuning and model selection on training data | Tune hyperparameters and compare models with cross-validation on the training split only (`evaluate.cv_splitter()`). Re-tune after the feature freeze. | LR tuned on the full training split (with `URLSimilarityIndex`, so needs re-tuning); RF re-tuned without it on a 20k training sample; DT and SVM not tuned yet. All need re-tuning after the freeze. |
| 5 | Final test-set usage | Run `evaluate.evaluate()` on the official 47,074-row test set **once per model**, after steps 1 to 4 are frozen. Nothing is changed after that. | **Not done.** `RUN_FINAL_TEST = False` in nb01 and nb04; nb02 and nb03 have no guard yet. The existing LR test report is provisional (section 15). |

**Feature and leakage decisions:** `URLSimilarityIndex` is dropped (group decision,
2026-11-19). `IsHTTPS` and the page features are still unresolved (see section 14
and the pending group decisions in section 15); they stay in the shared feature
list as a temporary default, not as a group choice.

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
| `URLSimilarityIndex` is exactly 100 for **every** legitimate URL (0.79 % of phishing URLs). The rule "below 100 means phishing" alone gives F1 about 0.996 on training rows. | Label leakage by construction: the feature measures similarity to the legitimate URL list the dataset was built from. | nb01 §7; nb04 phase 2 | **Dropped** (group decision, 2026-11-19), via `main.DROPPED_FEATURES` until `select_features()` does it. |
| Removing `URLSimilarityIndex` hardly changes the scores (LR CV F1 0.99991 to 0.99965), because page features also separate the classes almost perfectly on their own (`LineOfCode`, `NoOfExternalRef`, `NoOfImage`, ... single-feature ROC-AUC 0.96 to 0.99). | Collection bias: phishing pages in the dataset are mostly bare or already taken down. | nb01 §7; nb04 phase 2 | Provisional (RF numbers from a 20k sample). **Group decision pending** on retaining or excluding affected page features. |
| `IsHTTPS` = 1 for every legitimate URL, and no legitimate URL has `?`, `=`, `&`, an IP address or obfuscation. | Collection bias: the legitimate class is mostly clean HTTPS homepages. | nb04 phase 2 | Report it; group decision pending |
| With **URL features only**, without `URLSimilarityIndex` and `IsHTTPS`, Random Forest CV F1 drops to about 0.96. This is closer to how hard the real task is. | Harder, more honest variant | nb04 ablation (20k sample) | Provisional |
| After the drop, the re-tuned Random Forest's SHAP values put about 74 % of the weight on page features (top: `LineOfCode`, `NoOfSelfRef`, `NoOfExternalRef`, `NoOfImage`, then `IsHTTPS`). LR relies most on `IsHTTPS` and URL character ratios instead. | Confirms the page-feature collection bias in the forest | nb04 phase 5 (20k sample) | Provisional |
| 425 URLs appear twice. The copies always have the same label but **different feature values** (for example `URLLength`). `clean()` keeps the first copy. | Data quality | nb01 §3 | Done (required by the project rules in `CLAUDE.md`) |
| 1,165 near-duplicate URL pairs (differ only by trailing `/`). 392 pairs fall on both sides of the train/test split. One pair has conflicting labels (`https://www.education.gouv.fr/` phishing vs `https://www.education.gouv.fr` legitimate). | Possible leakage and label noise | nb01 §4 | **Group decision pending.** Kept in the data for now. |
| About 4 % of test URLs share a domain with a training URL. | Mild overlap | nb04 (20k sample) | Optional: split grouped by domain (group decision) |

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

- **Feature engineering is a placeholder.** `add_features()` contains only two
  example features (`url_digit_ratio`, `url_special_char_count`) and
  `select_features()` returns every feature unchanged. The current "engineered"
  results are therefore 50 original + 2 example features, **not** a real
  before/after comparison.
- **Risk scoring is checked on a 20k sample only.** Sigmoid calibration was chosen
  out-of-fold on training rows (Brier score 0.00131 to 0.00043, nb04 phase 4) and the
  0.30 / 0.70 bands match their names there. Trade-off: with calibration, 4 phishing
  URLs fall in the Low band (none without it). The Low / Medium / High bands are not
  in proposal 4.11, so the group must confirm them. Both choices must be re-checked
  on the full training set.
- **Explainability is shown for the Random Forest and LR only.** SHAP for the forest
  and coefficient contributions for LR (nb04 phase 5). The LR view is provisional
  until LR is re-tuned without `URLSimilarityIndex`. Decision Tree and SVM are
  supported by `explain.contributions()` but not shown until they are tuned.
- **Decision Tree and SVM use untuned default settings.** The fully grown tree
  gives probabilities of almost only 0 or 1, so its ROC-AUC and risk score are not
  meaningful yet. The SVM is a linear SVM; a kernel SVM is too slow for the full data.
- **Random Forest was tuned on a 20,000-row sample only** (re-tuned without
  `URLSimilarityIndex`), and its settings are provisional until the feature list
  is frozen and they are checked on the full training set.
- **No full-dataset results yet.** `reports/results.csv` does not exist, and the
  full `main.py` run time has not been measured.
- **Page-feature collection bias.** Page features such as `LineOfCode`,
  `NoOfExternalRef` and `NoOfImage` separate the classes almost perfectly on
  their own (single-feature ROC-AUC 0.96 to 0.99 on training rows, nb01 §7 and
  nb04 phase 2). This most likely reflects how the pages were collected (many
  phishing pages were bare or already taken down), not a property a real-world
  detector could rely on. The group has **not** decided whether to retain or
  exclude these features (section 15).
- **Dataset scope.** One dataset from one collection period. High scores here do
  not prove the models would work on new, real-world URLs.

---

## 15. Current progress

Last updated: 2026-11-19. Update this section in every pull request that changes
the status of a task.

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
| Drop `URLSimilarityIndex` from both feature sets (group decision) | Thanthrige D.P (`main.py`) | `main.DROPPED_FEATURES`, commit `2569c27`, PR #4 | Verified (temporary location until `select_features()` does it) |

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
| Move the `URLSimilarityIndex` drop into `select_features()` | Madushan S.C | Requested | Awaiting confirmation | Then `main.DROPPED_FEATURES` can be removed (Thanthrige D.P). |
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

Only `URLSimilarityIndex` is decided. For the others, the options are for
discussion only, and the current code behaviour (for example, keeping `IsHTTPS`)
is a temporary default, not a group choice.

| Decision | Options under discussion | Affects | Status |
|---|---|---|---|
| `URLSimilarityIndex` | keep / drop in `select_features()` / report both | everyone's results | **Decided 2026-11-19: drop.** Applied in `main.DROPPED_FEATURES` until `select_features()` does it. |
| Risk bands and calibration | keep the Low / Medium / High bands (0.30 / 0.70; not in proposal 4.11) or report the score only; sigmoid-calibrated score (better Brier score, 4 phishing URLs in Low on the 20k sample) or raw score (no phishing in Low) | risk score, demo | Unresolved (current code: bands kept, sigmoid) |
| Near-duplicate URLs | (a) keep as now; (b) normalise only safe cases and deduplicate; (c) group-aware split; (d) drop only the conflicting pair (details in nb01 §4) | `clean()` or `split()`, everyone's data | Unresolved |
| Domain-grouped splitting | keep the random split / split grouped by domain | `split()`, everyone's data | Unresolved |
| `IsHTTPS` | keep / drop / keep and also report a URL-only result | everyone's results | Unresolved |
| Page features with collection bias (`LineOfCode`, `NoOfExternalRef`, `NoOfImage`, ...) | retain / exclude / retain and also report a URL-only result | everyone's results, feature freeze | Unresolved |

---

## 16. Progress evaluation preparation

### Status against the evaluation criteria

| Criterion | What we must show | Owner(s) | Status |
|---|---|---|---|
| Problem analysis | Problem statement, objectives, dataset description, class balance | Madushanka D.M.C (data), all (problem) | Partial: this README and nb01; class-balance figure missing |
| Data preprocessing | Cleaning steps with evidence and reasons, shared split | Madushanka D.M.C | Done (two group decisions pending) |
| Feature engineering | New features with justification, selection method, before/after effect | Madushan S.C | Pending (placeholder only) |
| Model selection and implementation | Why each model, tuning method and chosen hyperparameters | Each model owner | Proposed models: LR, RF, DT. LR and RF: Provisional. DT: initial scaffold exists; Member 3 implementation, tuning and evaluation pending. SVM: initial scaffold exists; Member 4 implementation, tuning and evaluation pending. |
| Model evaluation | Same metrics for all models, comparison table and plots, leakage discussion | Dharmasiri S.O.R (comparison), each owner (own model) | Partial: metrics and CV exist; comparison and plots pending |
| Demonstration | Live run of the pipeline and notebook sections | Thanthrige D.P (`main.py`), each owner | Partial: `main.py --sample 20000` runs end to end with risk scores and SHAP; nb04 phases 4 and 5 show calibration, bands and single-URL explanations (20k sample) |
| Viva | Every member explains and defends their own part | Everyone | Each member prepares their part |
| Individual contribution | Visible commits and pull requests from every member | Everyone | Member 1 (Madushanka D.M.C): PRs #2 and #3 merged (`724370e`, `f808569`). Member 2 (Thanthrige D.P): PR #1 merged as `7242dc4`, PR #4 merged as `c372c82`. Members 3 and 4: assigned, awaiting confirmation. |

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
