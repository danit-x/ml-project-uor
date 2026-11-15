# CLAUDE.md: URLSentra (EC5203 ML mini project, Group 42)

Explainable phishing URL risk scoring on the PhiUSIIL dataset (UCI #967) with
Logistic Regression, Decision Tree, Random Forest and SVM. The approved group
proposal is the scope. Do not add models, datasets or features beyond it without
asking first. Proposal section numbers (4.2 to 4.14) are cited in `src/` docstrings.

## First thing in every session

Four people share this file. If it is not clear which member you are working
with, ask before editing anything. After that, edit **only that member's files**
(table below). If a change is needed in someone else's file, do not make it.
Say what to ask that member for: the file, the function, and the exact change.

## Project overview

Target: `data.get_target()` flips the raw label, so **1 = phishing**, 0 = legitimate
(raw `label` 0 = phishing, `config.PHISHING_LABEL`). Positive class, precision,
recall and risk score all refer to phishing.

Pipeline in `main.py:run()`:
1. `data.load_clean_data()`: `load_raw()` then `clean()` (optional `--sample` uses `RANDOM_STATE`)
2. `data.get_original_features(df)`: numeric columns, minus target, `TEXT_COLUMNS`, engineered columns
3. `features.add_features(df)` adds `ENGINEERED_FEATURES`; `features.select_features(train_rows, cols)`
   returns the kept columns. main.py passes **training rows only** (`df.loc[X_train.index]`)
4. For feature sets `original` and `engineered`: `data.split(df, cols)` then, for each
   module in `src/models/__init__.py:MODELS`, `module.train(X_train, y_train)`
5. `evaluate.evaluate(model, X_test, y_test)` then `evaluate.compare(results)` writes
   `reports/results.csv` (a `--sample N` run writes `reports/results_sample_N.csv`, gitignored)
6. Fitted models go to `saved_models/<model>_<set>.joblib`

`src/risk.py` (`risk_score`, `risk_level`, `calibrate`) and `src/explain.py` (SHAP)
are not yet called by `main.py`. They are used from notebook 04.

Interface contract (main.py depends on it, so keep names and signatures):
- every `src/models/*.py` exposes `build()` (unfitted estimator) and `train(X, y)` (fitted)
- fitted models must support `predict` and `predict_proba`, because `evaluate()` and `risk_score()` use `[:, 1]`
- `config` holds all paths and shared constants; import from it, never hard-code paths

## Commands

Python 3.12+ (`requirements.txt` pins minor versions).
```bash
python -m venv .venv
.venv\Scripts\activate              # Windows; macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
nbstripout --install                # once per clone: strips notebook outputs on commit
```
Dataset: put `PhiUSIIL_Phishing_URL_Dataset.csv` in `data/raw/` (download from https://archive.ics.uci.edu/dataset/967).

```bash
python main.py --sample 20000       # quick end-to-end check (always do this first)
python main.py                      # full dataset (SVM is slow)
python main.py --data path/to/file.csv
jupyter notebook notebooks/         # notebooks use sys.path.append(".."), so run them from notebooks/
```
Notebooks start with `SAMPLE = 20000`; set it to `None` only for final full-data runs.
A full `main.py` run overwrites `reports/results.csv` and `saved_models/`.

## Ownership and git workflow

| Member | Model | Pipeline part | Files you own |
|---|---|---|---|
| Madushanka D.M.C | Logistic Regression | Dataset inspection, file integration, cleaning, preprocessing, train/test split | `src/data.py`, `src/models/logreg.py`, `notebooks/01_data_logreg.ipynb` |
| Madushan S.C | Decision Tree | Feature engineering, feature selection | `src/features.py`, `src/models/tree.py`, `notebooks/02_features_tree.ipynb` |
| Dharmasiri S.O.R | SVM | Common evaluation, model comparison, before/after feature engineering comparison | `src/evaluate.py`, `src/models/svm.py`, `notebooks/03_evaluation_svm.ipynb` |
| Thanthrige D.P | Random Forest | Risk scoring, calibration, explainability (SHAP), integration | `src/risk.py`, `src/explain.py`, `src/models/forest.py`, `main.py`, `notebooks/04_risk_explain_forest.ipynb` |

Shared files with no single owner: `src/config.py`, `src/__init__.py`,
`src/models/__init__.py`, `requirements.txt`, `.gitignore`, `.gitattributes`, `README.md`, `CLAUDE.md`.
Change them only when the member confirms the group agreed. Make that change in its own commit.

- Each member works on their own branch (`git checkout -b <name>`), and opens a pull request into `main`.
- Never push to `main`. Pull `main` regularly (`git pull origin main`).
- Keep commits small and focused, one logical change each. Use an imperative message
  that says what and why (for example, "Tune LR C with 5-fold CV on train set").
- Never commit: the dataset (`data/raw/`, `data/processed/`), `saved_models/`, `.joblib`
  files, large CSVs, or notebook outputs (nbstripout removes them; save figures to `reports/figures/`).
  Before committing, check `git status` for anything unexpected.
- Commit or push only when the member asks.

## ML engineering rules

**No data leakage.** Anything that is fitted (scalers, imputers, encoders, feature
selection, calibration) is fitted on training data only, inside an sklearn
`Pipeline` returned by `build()` (see `logreg.py` and `svm.py`).
- `add_features()` runs on the whole dataset before the split, so it must stay row-wise:
  each value is computed from that row alone, with no counts, vocabularies or statistics across rows.
- `select_features()` receives training rows only. In notebooks, call it with
  `df.loc[X_train.index]`, never the full `df`. Per-model selection can also go in the
  pipeline (`SelectKBest`/`SelectFromModel`).
- `clean()` may drop rows but must not impute with dataset-wide statistics. It drops
  repeated URLs so the same URL cannot be in both train and test. Keep that.
- `risk.calibrate()` fits on training data (internal CV). Never calibrate on the test set.

**Splits and config.** Always split with `data.split()`. Use `config.RANDOM_STATE`
(42) and `config.TEST_SIZE` (0.2) everywhere, never literal seeds. Do not change
`config.py` values without group agreement.

**Tuning.** Use `GridSearchCV`/`RandomizedSearchCV` with `cv=evaluate.cv_splitter()`
(`config.CV_FOLDS` stratified, shuffled, seeded folds) on `X_train` only. Compare
variants with `evaluate.cv_evaluate(module.build(), X_train, y_train)`. Write the chosen
hyperparameters as fixed values into `build()` so `main.py` stays fast and reproducible.
The test set is for final reporting, not model selection. Tune the SVM on a training sample first.

**Evaluation.** Always use `evaluate.evaluate()` so all models are comparable.
Report all of its metrics (accuracy, precision, recall, F1, ROC-AUC, confusion
counts), not accuracy alone. Compare `original` and `engineered` feature sets for every model.

**Near-perfect scores are a warning, not a result.** If any metric is about 0.99 or
higher, check for leaking features before reporting it. Start with `URLSimilarityIndex`
(already flagged in `select_features()`), and also check single-feature ROC-AUC on the
training set. The group decides what to drop.

**Reproducibility.** Fixed seeds. Results go to `reports/` and figures to
`reports/figures/` (`config.RESULTS_DIR`, `config.FIGURES_DIR`). Every number in the
report must be regenerable from committed code, never typed in by hand.

**Experiment log.** For each experiment, add a row to an "Experiment log" table in the
member's notebook: what changed, why, CV or test, metrics, commit hash.

## Code standards

- Put reusable logic in the member's `src/` file, with docstrings that state inputs, outputs and the reason.
- Notebooks are for exploration and narrative. They import from `src`, and final code moves into `src`.
- Keep existing function names, signatures and return types. `main.py` and other members' code call them.
- No new dependency unless it is added to `requirements.txt` (a shared file, so group agreement is needed).
- No hard-coded paths or seeds. Use `config`.

## How to work with the member

- Plan before large changes (more than one function, or anything touching the
  interface contract). State the plan, then wait for agreement.
- Run the code and show the real output before saying it works. Never invent or estimate metrics.
- Run `python main.py --sample 20000` (or a sampled notebook cell) before full-dataset runs.
- Explain the reasoning behind every ML decision in plain language: what, why, the alternative,
  and the trade-off. Every member must defend their part in a viva.
- If a requested change breaks these rules or the proposal scope, say so before doing it.

## Definition of done

- `python main.py --sample 20000` runs end to end, then `python main.py` on the full data.
- Results are reproducible: a rerun gives the same numbers, and figures and CSVs are regenerated by code.
- The member's notebook explains the decision and logs the experiment, with outputs cleared.
- `git status` shows changes only in the member's own files, plus agreed shared-file changes.
