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

```bash
git clone git@github.com:danit-x/ml-project-uor.git
cd ml-project-uor

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
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

This runs the whole pipeline on a 20,000-row sample and prints a results table.
`python main.py` runs it on the full dataset (the SVM takes a while).

## How we work

1. Create your own branch: `git checkout -b yourname`
2. Edit only the files you own (table above). Your notebook is your workspace;
   reusable code goes in your `src` file.
3. Commit and push to your branch: `git push -u origin yourname`
4. Open a pull request into `main`. Do not push to `main` directly.
5. Pull `main` regularly to get the others' work: `git pull origin main`

You do not have to wait for anyone. Every part already has a working baseline,
so start from the raw dataset and replace the baseline with your real work.

## Shared rules

- Settings live in `src/config.py` (`RANDOM_STATE = 42`, `TEST_SIZE = 0.2`).
  Do not change them on your own; agree in the group first.
- Always split with `data.split()` and evaluate with `evaluate.evaluate()`, so
  all four models are compared on the same test set with the same metrics.
- Keep the existing function names. `main.py` calls them to combine everything.
- Save figures to `reports/figures/`.

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
