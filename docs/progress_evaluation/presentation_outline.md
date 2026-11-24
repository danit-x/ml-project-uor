# URLSentra — 15-minute progress presentation (outline and viva notes)

Draft of 2026-11-24. Numbers come from `reports/` (full-data run of `python main.py`); figure files are in
`reports/figures/`. The speaker column is a **suggestion by file ownership, not agreed by the group**; the final deck is
`URLSentra_progress_presentation.pptx` (its speaker notes are the current script).

## Timing (15 min + viva)

| # | Slide | Time | Suggested speaker (not agreed) |
|---|---|---|---|
| 1 | Title, team, problem in one sentence | 0:30 | Madushanka |
| 2 | Problem and motivation: risk score + explanation, not just a label | 1:00 | Madushanka |
| 3 | Dataset and cleaning | 1:00 | Madushanka |
| 4 | Leakage found and fixed: `URLSimilarityIndex`, near-duplicate URL groups, grouped split/CV | 1:30 | Madushanka / Thanthrige |
| 5 | Feature engineering and the three feature sets | 1:30 | Madushan |
| 6 | Models and tuning protocol (one-SE rule, bounded searches) | 1:30 | each owner, 30 s each |
| 7 | Decision Tree complexity review | 1:00 | Madushan |
| 8 | Results: three-model comparison (grouped CV + held-out test) | 1:30 | Dharmasiri |
| 9 | Collection bias: the bare-homepage rule and the format-controlled breakdown | 1:30 | Madushanka |
| 10 | Risk score, calibration, bands | 1:00 | Thanthrige |
| 11 | Live demo: `python demo.py` (score, band, SHAP for held-out URLs) | 1:30 | Thanthrige |
| 12 | Completed vs remaining work, individual contributions | 1:00 | Dharmasiri |
| — | Buffer | 0:30 | |

## Slide content

**1. Title.** URLSentra: explainable phishing URL risk scoring. Group 42, EC5203.

**2. Problem.** Decide *before clicking* how risky a URL is and why. Output = calibrated phishing probability
(0–1), Low/Medium/High band, top contributing features. Goal of this stage: a *defensible* evaluation.

**3. Dataset.** PhiUSIIL (UCI #967): 235,795 URLs, 56 columns, no missing values; 425 repeated URLs removed (keep
first) → 235,370 rows (100,520 phishing, 134,850 legitimate). Target flipped so 1 = phishing.

**4. Leakage we found and fixed.**
- `URLSimilarityIndex` = 100 for every legitimate URL → one-line rule F1 ≈ 0.996 → dropped from every set.
- Near-duplicate URLs (scheme / `www.` / host case / trailing slash variants) → URL groups (231,779 groups; 3,287 contain more than one row; 43 groups (86 rows) carry conflicting labels);
  group-aware stratified split (188,297 / 47,073 rows, 0 shared groups) and the same 5 grouped folds for every model.
- Feature selection refitted inside each CV fold; test rows never used for selection or tuning.
- Visual: the `url_group_key` normalisation table (nb02 `grouping-design`).

**5. Features.** `url_base` (20 URL-string features), `url_engineered` (+3 engineered: path depth, longest path
segment, query field count; `url_encoded_byte_count` removed as redundant, r = 0.994) = **primary**, `extended`
(+ `IsHTTPS` + 28 page features) = secondary. Engineered features are row-wise, so no leakage. Honest result (slide 8):
they gave no measurable improvement; `url_engineered` stays primary because it was fixed as primary before the results.

**6. Models and tuning.** Training rows, `url_engineered`, grouped 5-fold CV, phishing F1, one-SE rule fixed before
each run.
- LR: L2/lbfgs, C = 1000 (F1 0.99674 ± 0.00026; **upper edge** of the range; L1 search stopped: no convergence).
- DT: max_depth 20 (F1 0.99053; 1,299 leaves vs 1,564 for the unlimited tree).
- RF: 300 trees, depth 20, max_features 0.5, bounded 8-candidate search on a 50k-row group-preserving subset
  (subset F1 0.99119 ± 0.00181).
- SVM: provisional (C from an earlier ungrouped 2,000-row run), reported separately.

**7. DT complexity.** Figure `reports/figures/dt_complexity_curve.png` (F1 vs leaves); "as good and simpler";
top-levels figure from nb02 for interpretability (only a 20,000-row sample version,
`dt_top_levels_sample_20000.png`, exists and it is not committed: label it "sample" on the slide or regenerate it on full data).

**8. Results.** Table from `reports/official_comparison.csv`:

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

**9. Main limitation: dataset collection bias (the key scientific point).** Every legitimate URL is a bare
`https://www.<domain>` homepage; about 1 % of phishing URLs are. The fixed rule "not a bare https://www homepage →
phishing" (no learning): grouped-CV F1 0.9948 ± 0.0004 and held-out F1 0.9948 (precision 1.000, 209 missed phishing
URLs) (`reports/format_baseline.csv`). Logistic Regression beats it by only 0.0025 F1 on the test rows, the Random
Forest merely matches it and the Decision Tree is below it.
On the 27,178 held-out bare-homepage URLs, where the format cue is absent (26,969 legitimate, 209 phishing), the rule finds no phishing, while LR finds 102 (2 false alarms), DT 106 (135) and RF 104 (75) (`reports/format_controlled.csv`): the models learned some signal beyond the format, but on these hard cases they catch only about half of the phishing URLs, and 107 of LR's 108 missed phishing URLs are of this kind.
Message: the high scores mostly measure the dataset's collection format; we do **not** claim real-world
generalisation. Future work: evaluate the frozen models on an independently collected dataset.

**10. Risk score.** Sigmoid-calibrated RF probability (grouped folds, training rows only); bands 0.30 / 0.70.
Caveat: sigmoid was re-checked on a 20,000-row development sample only, and the internal calibration CV folds of that
check were ungrouped.
On the held-out rows the calibrated score has Brier 0.0038 (`reports/main_full_run_2026-11-24.log`); Low holds 57 % of URLs (0.4 % phishing), Medium 0.2 % (40 % phishing), High 43 % (99.7 % phishing) (`reports/risk_bands.csv`).
**Say explicitly: 108 of the 20,104 held-out phishing URLs fall in the Low band** (and 61 legitimate URLs in High).
"Low" means low estimated risk, not "safe". (Not the same thing as LR's 108 false negatives: one is a calibrated-forest band count,
the other a Logistic Regression error count. The numbers are equal by coincidence and the two sets only partly
overlap (`docs/progress_evaluation/build/check_low_band.py`). Do not describe the Low-band URLs' format; it is not part of the reported results.)

**11. Demo.** `python demo.py` → three held-out URLs (clear phishing, clear legitimate, most uncertain): score, band,
prediction, true label, bare-homepage flag, top-3 SHAP contributions. `python demo.py --url "<test URL>"` for one
URL. Arbitrary URLs are refused on purpose (inputs such as `TLDLegitimateProb` come from the dataset's extractor).
Optional fourth example for the bias message: a held-out phishing URL in the bare-homepage format, chosen from
`reports/format_controlled.csv` cases; its output is not in the saved fallback, so show it only after running it.
Logistics: each `demo.py` call takes about 90 s to start (loads data and models), so launch it before the slot or
in a second terminal; the expected output is saved in `reports/demo_full_2026-11-24.txt` as a fallback.

**12. Status.** Done: cleaning, grouping, grouped split/CV, feature sets, three tuned models, full-data evaluation,
format baseline, calibrated risk score, SHAP, demo, 64 unit tests. Remaining: SVM grouped retune; final report.
Future work: evaluation on an independently collected dataset. Contribution table as in the progress document.

## Likely viva questions and defensible answers

1. **Why are your scores so high? Is it leakage?** We removed the known leaks (`URLSimilarityIndex`, near-duplicate
   URLs across the split; tuning and feature selection use training rows only). What remains is *collection bias*:
   legitimate URLs are all bare https://www homepages, and a rule with no learning already scores close to our
   models. We report that rule next to every URL-only result and do not claim real-world performance.
2. **Why group near-duplicates instead of deleting them?** Deleting loses correctly labelled rows and still needs a
   rule for which copy to keep; grouping keeps all rows and guarantees that no variant of a URL is in both training
   and test (or in both parts of a CV fold).
3. **Why keep path and query case in the group key?** Hostnames are case-insensitive (RFC 3986) but paths and
   queries are not, so `/Login` and `/login` can be different resources; lower-casing everything or dropping the
   query merged unrelated URLs (groups of up to 45 rows).
4. **What about the label-conflict groups?** Kept unchanged as label noise; because a group never crosses the split,
   the conflict cannot leak and we do not edit labels we cannot verify.
5. **Why F1 for selection?** Phishing is the positive class; F1 balances missed phishing (recall) and false alarms
   (precision). All five metrics and confusion counts are still reported.
6. **What is the one-standard-error rule?** Among candidates whose mean CV score is within one standard error of the
   best, pick the simplest (strongest regularisation / fewest leaves). Differences smaller than the fold-to-fold noise
   are not evidence that a more complex model is better.
7. **Why is LR's C so large (1000), and is that a problem?** The URL features are almost linearly separable because
   of the format bias; a strong penalty stops LR from drawing that boundary sharply. C = 1000 is at the edge of our
   range — we say so; the gains from C = 30 to 1000 are about one SE per step, so the curve has flattened.
   *Why not a smaller C under the one-SE rule?* Best mean F1 0.99674 (C = 1000), SE 0.00013, threshold 0.99661.
   C = 300 has 0.99658, just below the threshold, so it is not eligible; C = 1000 is the only eligible candidate
   (`reports/tuning/lr_followup_search.csv`). The rule was fixed before the run and applied as written.
8. **Why did you stop the L1 search?** L1/liblinear at C = 0.1 ran 33 minutes on one fold without converging; a
   non-converged fit is not a valid candidate. We kept the log and evaluated only converged L2 fits.
9. **Why tune the Random Forest on a subset?** A full-data search costs about 100 forest fits; we used a reproducible
   50,000-row subset that keeps whole URL groups and the class balance, then trained the chosen forest on the full
   training split and reported its full-data CV and test results separately from the subset CV.
10. **Why max_depth 20 for the tree?** It is the simplest tree within one SE of the best (here also the best mean);
    the unlimited tree is deeper and larger with no CV gain. No tree in this range is readable leaf by leaf, so we
    explain it by its top levels and importances.
11. **Is the feature selection leak-free?** The redundancy filter uses no labels and is refitted inside each CV fold
    (`FeatureSetSelector`); on the full training data every fold chose the same columns as the training-rows filter.
12. **How is the risk score calibrated without touching the test set?** The served model is
    `CalibratedClassifierCV` fitted on training rows only, with the grouped training folds. The method (sigmoid:
    lowest out-of-fold Brier score and log loss) was re-checked on a 20,000-row development sample; in that check the
    outer folds were grouped but the internal calibration CV folds were ungrouped, and it was not repeated on the full
    training split. We disclose both.
13. **Why can't the demo score any URL?** Several inputs (e.g. `TLDLegitimateProb`, `URLCharProb`) were computed by
    the dataset authors' extractor and are not reproduced in this project; scoring other URLs would be an
    unsupported claim.
14. **Why is the SVM not in the main comparison?** Its C was tuned with the earlier ungrouped protocol on 2,000 rows;
    until it is retuned with grouped CV its numbers are provisional.
15. **Did your engineered features help?** No measurable gain under grouped CV: identical F1 for DT and RF, and
    LR is 0.0005 better *without* them. One of the four was removed as redundant. We report this null result and
    keep `url_engineered` as primary because it was chosen as primary before the results; picking the set that
    scored higher afterwards would be selection on results. With the format bias, the base URL features already
    separate almost everything, so there is little room for new URL-structure features to add.
16. **What would you do next?** Evaluate the frozen models on an independently collected dataset (legitimate deep
    links, phishing from another source and period): PhiUSIIL itself has no legitimate non-homepage URLs, so the
    bias cannot be tested inside it. Also the SVM retune and the final analysis.
17. **Why use risk bands if 108 phishing URLs land in Low?** We disclose it: 108 of 20,104 held-out phishing URLs
    (0.5 %) are Low, and 61 legitimate URLs are High. These 108 are not the same measurement as LR's 108 false negatives
    (calibrated forest band vs LR errors); the counts match by coincidence and the sets only partly overlap. We make
    no claim about their URL format in the reported results. The bands
    communicate estimated risk; they are not a safety guarantee, and the calibration choice still needs group
    confirmation.
18. **Is the collection bias leakage?** No: leakage is information about the label that would not be available at
    prediction time (like `URLSimilarityIndex`). The URL format is available at prediction time, but in this dataset
    it is correlated with the label only because of how the two classes were collected. It inflates the scores in
    the same way, which is why we report the format rule next to every result.
