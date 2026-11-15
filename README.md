# URLSentra: Explainable Machine Learning for Phishing URL Risk Scoring

EC5203 Machine Learning mini project, Group 42, University of Ruhuna.

We classify URLs as phishing or legitimate with four classical models, compare
them before and after feature engineering, turn the phishing probability into a
risk score, and explain the predictions.

## Who does what

Everyone owns one model and one pipeline part.

| Member | Model | Pipeline part | Files you own |
|---|---|---|---|
| Madushanka D.M.C | Logistic Regression | Dataset inspection, file integration, cleaning, preprocessing, train/test split | `src/data.py`, `src/models/logreg.py`, `notebooks/01_data_logreg.ipynb` |
| Madushan S.C | Decision Tree | Feature engineering, feature selection | `src/features.py`, `src/models/tree.py`, `notebooks/02_features_tree.ipynb` |
| Dharmasiri S.O.R | SVM | Common evaluation, model comparison, before/after feature engineering comparison | `src/evaluate.py`, `src/models/svm.py`, `notebooks/03_evaluation_svm.ipynb` |
| Thanthrige D.P | Random Forest | Risk scoring, calibration, explainability (SHAP), integration | `src/risk.py`, `src/explain.py`, `src/models/forest.py`, `main.py`, `notebooks/04_risk_explain_forest.ipynb` |

For your own model: train it, tune its hyperparameters, evaluate it with all the
metrics, and run it on both the original and the engineered features.

## Setup

Needs Python 3.12 or newer.

```bash
git clone git@github.com:danit-x/ml-project-uor.git
cd ml-project-uor

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
nbstripout --install            # once per clone: strips notebook outputs on commit
```

### Dataset

The dataset is not stored in the repo.

1. Download it from the UCI repository: https://archive.ics.uci.edu/dataset/967
2. Unzip it and put `PhiUSIIL_Phishing_URL_Dataset.csv` in `data/raw/`.

In the dataset `label = 1` is legitimate and `label = 0` is phishing. In our code
the target is flipped so that **1 = phishing**, which makes precision, recall and
the risk score refer to phishing.

### Check that it works

```bash
python main.py --sample 20000
```

This runs the whole pipeline on a 20,000-row sample, prints a results table and
saves it to `reports/results_sample_20000.csv` (not committed).
`python main.py` runs it on the full dataset (the SVM takes a while) and writes
`reports/results.csv`, the file we report from.

## How we work

1. Create your own branch: `git checkout -b yourname`
2. Edit only the files you own (table above). Your notebook is your workspace;
   reusable code goes in your `src` file.
3. Commit and push to your branch: `git push -u origin yourname`
4. Open a pull request into `main`. Do not push to `main` directly.
5. Pull `main` regularly to get the others' work: `git pull origin main`

You do not have to wait for anyone. Every part already has a working baseline,
so start from the raw dataset and replace the baseline with your real work.

If you use Claude Code, `CLAUDE.md` holds the project rules it follows. Tell it
which member you are at the start of a session.

## Shared rules

- Settings live in `src/config.py` (`RANDOM_STATE = 42`, `TEST_SIZE = 0.2`,
  `CV_FOLDS = 5`). Do not change them on your own; agree in the group first.
- Always split with `data.split()`, so all four models use the same test set.
- Develop and tune with cross-validation on the training set only:
  `evaluate.cv_evaluate()`, or `cv=evaluate.cv_splitter()` in `GridSearchCV`.
  Use `evaluate.evaluate()` on the test set once, for the final numbers.
- Anything fitted (scaling, imputation, selection) goes inside the model's
  pipeline so it only learns from training data.
- Keep the existing function names. `main.py` calls them to combine everything.
- Save figures to `reports/figures/`.
- Shared files with no single owner (`src/config.py`, `src/__init__.py`,
  `src/models/__init__.py`, `requirements.txt`, `.gitignore`, `.gitattributes`,
  `README.md`, `CLAUDE.md`): change them only after the group agrees.

## Project layout

```
data/raw/            dataset CSV (not in git)
data/processed/      cleaned data (not in git)
notebooks/           one notebook per member
src/config.py        shared settings
src/data.py          loading, cleaning, split
src/features.py      feature engineering and selection
src/models/          one file per model
src/evaluate.py      metrics and comparison
src/risk.py          risk score and calibration
src/explain.py       SHAP explainability
main.py              runs the full pipeline
reports/             results.csv and figures
```
