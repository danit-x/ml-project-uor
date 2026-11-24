# Logistic Regression – Final Test Results

> **Superseded (2026-11-24).** These numbers come from the earlier protocol (random split, 50 original
> features including `URLSimilarityIndex` and page content). The current Logistic Regression results under the
> agreed protocol (group-aware split, grouped CV, `url_engineered`, C = 1000) are in
> `reports/official_comparison.csv`, written by `python main.py`. This file is kept as history only.

> **⚠️ PROVISIONAL RESULTS — NOT OFFICIAL FINAL RESULTS**
>
> These Logistic Regression test results are preliminary and were obtained before the project's feature selection and evaluation protocol were finalized. They are provided for development reference only and must not be presented as the official project results.
>
> The group has not yet finalized decisions regarding URLSimilarityIndex, near-duplicate URLs, and the train/test splitting strategy. After these decisions and the feature set are finalized, the model must be re-evaluated using the agreed evaluation protocol. The final test results will then replace or supersede these provisional results.

**Project:** URLSentra – Explainable Machine Learning for Phishing URL Risk Scoring  
**Model:** Logistic Regression  
**Date:** 2026-11-17  
**Evaluation:** Held-out Test Set

## 1. Performance Metrics

| Metric | Result |
|---|---:|
| Accuracy | 0.99989 |
| Precision | 1.00000 |
| Recall | 0.99975 |
| F1-score | 0.99988 |
| ROC-AUC | 0.99999 |

## 2. Confusion Matrix

| Actual / Predicted | Legitimate | Phishing |
|---|---:|---:|
| Legitimate | 26,970 | 0 |
| Phishing | 5 | 20,099 |

- True Negative (TN): 26,970
- False Positive (FP): 0
- False Negative (FN): 5
- True Positive (TP): 20,099

**Total test records:** 47,074

## 3. Observation

The Logistic Regression model achieved an F1-score of 0.99988 and ROC-AUC of 0.99999 on the held-out test set. It correctly classified 20,099 phishing URLs and 26,970 legitimate URLs. However, 5 phishing URLs were incorrectly classified as legitimate.

## 4. Limitation

The dataset contains features such as `URLSimilarityIndex` that may act as proxies for the target label. Therefore, these results should not yet be interpreted as proof of real-world generalisation. Further investigation of feature dependency and potential data leakage is required.