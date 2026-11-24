// 15-minute progress-evaluation deck (12 slides, 16:9) for URLSentra.
//
//   python docs/progress_evaluation/build/collect_numbers.py      # numbers.json from reports/
//   node docs/progress_evaluation/build/build_presentation.js     # writes ../URLSentra_progress_presentation.pptx
//
// Needs the npm package "pptxgenjs" (3.12). Every number comes from numbers.json and the two figures
// from reports/figures/ (both written by `python main.py` / `python tune_models.py`). Speaker notes hold
// the script, timing and likely viva questions. Not part of the ML pipeline.
// Optional: APPLY_THEME=<path to apply_theme.js> also writes the palette into the deck's theme.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const N = JSON.parse(fs.readFileSync(path.join(__dirname, "numbers.json"), "utf8"));
const ROOT = path.resolve(__dirname, "..", "..", "..");
const OUT = path.join(__dirname, "..", "URLSentra_progress_presentation.pptx");
const FIG = (f) => path.join(ROOT, "reports", "figures", f);
const M = N.models, FC = N.format_controlled, FS = N.feature_sets, TU = N.tuning, RK = N.risk, K = N.constants, D = N.demo;
const num = (s) => Number(String(s).replace(/,/g, ""));

// ---------------------------------------------------------------- theme
const THEME = {
  name: "URLSentra", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1B262C", lt1: "FFFFFF", dk2: "0B3C49", lt2: "EEF4F5", accent1: "0E7C86", accent2: "D9822B",
    accent3: "5B6B73", accent4: "7FB7BE", accent5: "B5543C", accent6: "3E8E5E", hlink: "0E7C86", folHlink: "5B6B73" },
};
const C = { ink: "1B262C", deep: "0B3C49", teal: "0E7C86", tealSoft: "E3F0F1", amber: "D9822B", amberSoft: "FBEEDF",
  slate: "5B6B73", grid: "D5DEE1", white: "FFFFFF", paper: "F6F9F9", mist: "B9D3D6" };
const HEAD = THEME.headFontFace, BODY = THEME.bodyFontFace;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.333 x 7.5 in
pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };
pres.author = "URLSentra Group 42";
pres.title = "URLSentra progress evaluation";
pres.subject = "EC5203 Machine Learning mini project";
const W = 13.333, MX = 0.6;

pres.defineSlideMaster({
  title: "DARK_TITLE", background: { color: C.deep },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 2.05, w: 12.1, h: 1.6, fontFace: HEAD, fontSize: 40, bold: true, color: C.white, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: MX, y: 3.8, w: 12.1, h: 0.9, fontFace: BODY, fontSize: 20, color: C.mist, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.white },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.35, w: 12.1, h: 0.85, fontFace: HEAD, fontSize: 32, bold: true, color: C.deep, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "URLSentra · Group 42 · Progress evaluation", options: { x: MX, y: 7.0, w: 6, h: 0.3, fontFace: BODY, fontSize: 10, color: C.slate, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 10, color: C.slate, align: "right" },
});

// ---------------------------------------------------------------- helpers
const T = (slide, text, o) => slide.addText(text, { isTextBox: true, fontFace: BODY, color: C.ink, margin: 0, valign: "top", ...o });
const card = (slide, x, y, w, h, fill = C.paper, name) => slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
  x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08, objectName: name });
const content = (title, section) => {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
};
const stat = (slide, x, y, w, big, label, sub, color = C.teal) => {
  T(slide, big, { x, y, w, h: 0.75, fontFace: HEAD, fontSize: 36, bold: true, color });
  T(slide, label, { x, y: y + 0.78, w, h: 0.35, fontSize: 15, bold: true });
  T(slide, sub, { x, y: y + 1.13, w, h: 0.55, fontSize: 12, color: C.slate });
};
const bullets = (items, size = 16, extra = {}) => items.map((t, i) => ({
  text: t, options: { bullet: { indent: 18 }, breakLine: i < items.length - 1, paraSpaceAfter: 6, fontSize: size, ...extra } }));
const runs = (parts) => parts.map(([text, o = {}]) => ({ text, options: o }));
const AXIS = { catAxisLabelColor: C.slate, valAxisLabelColor: C.slate, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 13, valAxisLabelFontSize: 12, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 14, dataLabelColor: C.ink,
  titleFontFace: "+mn-lt", titleFontSize: 14, titleColor: C.ink, valGridLine: { color: C.grid, size: 0.75 }, catGridLine: { style: "none" },
  showLegend: false };

// ---------------------------------------------------------------- 1. title
pres.addSection({ title: "Introduction" });
{
  const s = pres.addSlide({ masterName: "DARK_TITLE", sectionTitle: "Introduction" });
  s.addText("URLSentra: Explainable Phishing URL Risk Scoring", { placeholder: "title" });
  s.addText("Progress evaluation · EC5203 Machine Learning · Group 42 · University of Ruhuna", { placeholder: "body" });
  T(s, "Madushanka D.M.C  ·  Madushan S.C  ·  Dharmasiri S.O.R  ·  Thanthrige D.P          24 November 2026",
    { x: MX, y: 4.75, w: 12.1, h: 0.4, fontSize: 16, color: C.white });
  // motif: the three risk bands as one segmented bar
  const bx = MX, by = 5.75, bw = 12.1, bh = 0.32;
  const lo = Number(K.low_max), hi = Number(K.high_min);
  s.addShape(pres.shapes.RECTANGLE, { x: bx, y: by, w: bw * lo, h: bh, fill: { color: C.teal }, line: { color: C.teal }, objectName: "band-low" });
  s.addShape(pres.shapes.RECTANGLE, { x: bx + bw * lo, y: by, w: bw * (hi - lo), h: bh, fill: { color: C.mist }, line: { color: C.mist }, objectName: "band-medium" });
  s.addShape(pres.shapes.RECTANGLE, { x: bx + bw * hi, y: by, w: bw * (1 - hi), h: bh, fill: { color: C.amber }, line: { color: C.amber }, objectName: "band-high" });
  T(s, `Low  < ${K.low_max}`, { x: bx, y: by + 0.42, w: 3, h: 0.3, fontSize: 12, color: C.mist });
  T(s, `Medium`, { x: bx + bw * lo, y: by + 0.42, w: 3, h: 0.3, fontSize: 12, color: C.mist });
  T(s, `≥ ${K.high_min}  High`, { x: bx + bw - 3, y: by + 0.42, w: 3, h: 0.3, fontSize: 12, color: C.mist, align: "right" });
  s.addNotes(`[0:30] Speaker: to be agreed by the group.
We are Group 42. URLSentra gives a URL a phishing risk score between 0 and 1, puts it in a Low, Medium or High band, and explains which URL features drove the score. Today's main message is about evaluation: our scores are near-perfect, and we will show why that mostly reflects how the dataset was collected.`);
}

// ---------------------------------------------------------------- 2. problem and approach
{
  const s = content("Decide before the click: how risky is this URL, and why?", "Introduction");
  T(s, "A bare phishing / legitimate label is not enough. A user or a mail or browser filter needs a calibrated risk and the reason behind it.",
    { x: MX, y: 1.4, w: 12.1, h: 0.75, fontSize: 18, color: C.ink });
  const steps = [
    ["1", "URL features", `${FS.n_features.url_engineered} features computed from the URL string (primary set)`],
    ["2", "Three classical models", "Logistic Regression, Decision Tree, Random Forest (SVM provisional)"],
    ["3", "Calibrated risk score", `Probability 0–1, then Low < ${K.low_max} ≤ Medium < ${K.high_min} ≤ High`],
    ["4", "Explanation", "SHAP shows the features that pushed each score up or down"],
  ];
  const w = 2.75, gap = 0.37, y = 2.45, h = 2.55;
  steps.forEach(([n, head, body], i) => {
    const x = MX + i * (w + gap);
    card(s, x, y, w, h, i === 2 ? C.tealSoft : C.paper, `step-${n}`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.25, w: 0.55, h: 0.55, fill: { color: C.teal }, line: { color: C.teal } });
    T(s, n, { x: x + 0.25, y: y + 0.25, w: 0.55, h: 0.55, fontSize: 18, bold: true, color: C.white, align: "center", valign: "middle" });
    T(s, head, { x: x + 0.25, y: y + 0.95, w: w - 0.5, h: 0.45, fontSize: 17, bold: true, color: C.deep });
    T(s, body, { x: x + 0.25, y: y + 1.4, w: w - 0.5, h: 1.0, fontSize: 14, color: C.ink });
    if (i < steps.length - 1) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + w + 0.06, y: y + h / 2 - 0.15, w: 0.25, h: 0.3, fill: { color: C.mist }, line: { color: C.mist } });
  });
  card(s, MX, 5.4, 12.1, 1.05, C.amberSoft, "goal");
  T(s, runs([["Goal of this stage: ", { bold: true, color: C.amber }], ["a defensible evaluation, not the highest possible score. Every number shown comes from files the committed code regenerates.", {}]]),
    { x: MX + 0.3, y: 5.55, w: 11.5, h: 0.8, fontSize: 17, valign: "middle" });
  s.addNotes(`[1:00] Problem and approach.
A phishing site copies a trusted service to steal credentials. People and filters often have to judge a link before opening it, so we want a risk score with a reason, not just a label.
Pipeline: URL-string features, three classical models from our proposal, a calibrated probability turned into three bands, and SHAP explanations.
Stress the goal: at this stage we care more about a defensible evaluation than about the highest score.

Likely viva question: Why classical models rather than deep learning? Answer: the approved proposal scopes LR, DT, RF and SVM; they are fast, interpretable and enough to expose the data issues we found.`);
}

// ---------------------------------------------------------------- 3. data and leakage
pres.addSection({ title: "Method" });
{
  const s = content("Data, preprocessing and leakage control", "Method");
  const cols = [
    [N.rows_clean, "URLs after cleaning", `PhiUSIIL (UCI #967): ${N.raw_rows} rows, ${N.raw_cols} columns, ${N.missing_values} missing; ${N.repeated_urls_removed} repeated URLs removed`],
    [N.groups, "URL groups", `${N.multi_groups} with several rows; ${N.conflict_groups} label-conflict groups (${N.conflict_rows} rows) kept as label noise`],
    [N.test_rows, "Held-out test rows", `${N.train_rows} training rows; group-aware stratified split (seed ${K.seed}); phishing share ${N.phish_share_train} in both`],
    [N.shared_groups, "Groups shared by train and test", "No near-duplicate URL can be memorised in training and recognised in test"],
  ];
  cols.forEach(([big, label, sub], i) => stat(s, MX + i * 3.08, 1.45, 2.85, big, label, sub, i === 3 ? C.amber : C.teal));
  const ctl = [
    ["Label proxy removed", "URLSimilarityIndex equals 100 for every legitimate URL, so it is excluded from every feature set."],
    ["One shared protocol", `All tuning and CV use the same ${K.cv_folds} grouped folds of the training rows. The test rows were used once.`],
    ["Fitted on training folds only", "Scaling, feature selection and calibration are refitted inside each training fold."],
  ];
  ctl.forEach(([head, body], i) => {
    const x = MX + i * 4.1, y = 3.85;
    card(s, x, y, 3.85, 2.05, C.paper, `control-${i}`);
    T(s, head, { x: x + 0.25, y: y + 0.22, w: 3.4, h: 0.4, fontSize: 17, bold: true, color: C.deep });
    T(s, body, { x: x + 0.25, y: y + 0.7, w: 3.4, h: 1.25, fontSize: 14 });
  });
  T(s, "Near-duplicates = same URL after removing the scheme and a leading “www.”, lower-casing the hostname and stripping a trailing “/”. Domain-level grouping is deferred for this evaluation, not rejected.",
    { x: MX, y: 6.15, w: 12.1, h: 0.6, fontSize: 12, color: C.slate });
  s.addNotes(`[1:30] Data and leakage control.
${N.raw_rows} URLs, ${N.raw_cols} columns, no missing values; label flipped so 1 = phishing. ${N.repeated_urls_removed} repeated URLs removed, first copy kept.
Near-duplicate URLs, for example the same address with http or https or a trailing slash, are put into one URL group, and a group never crosses the train/test split or a CV fold. Zero shared groups.
URLSimilarityIndex was dropped: it is 100 for every legitimate URL, so it encodes the label.
Domain-level grouping is deferred for this evaluation, not rejected.

Likely viva questions:
- Why group near-duplicates instead of deleting them? Deleting loses correctly labelled rows and needs a rule for which copy to keep; grouping keeps every row and still prevents leakage.
- Why keep path and query case in the group key? Hostnames are case-insensitive, paths and queries are not; dropping the query merged unrelated URLs into groups of up to 45 rows.
- What about the ${N.conflict_groups} label-conflict groups? Kept as label noise; a group never crosses the split, so the conflict cannot leak.`);
}

// ---------------------------------------------------------------- 4. feature sets
{
  const s = content("Three feature sets, fixed before the results", "Method");
  const sets = [
    ["url_base", FS.n_features.url_base, "Control", "URL-string features only. No URLSimilarityIndex, no IsHTTPS, no page content.", C.paper, C.deep],
    ["url_engineered", FS.n_features.url_engineered, "Primary", `url_base plus path depth, longest path segment and query-field count. ${FS.removed_feature} removed as redundant (r = ${FS.removed_r}).`, C.tealSoft, C.teal],
    ["extended", FS.n_features.extended, "Secondary only", `Adds IsHTTPS and ${FS.n_page_features} page-content features, which carry the same collection bias.`, C.paper, C.deep],
  ];
  sets.forEach(([name, n, role, body, fill, col], i) => {
    const x = MX + i * 4.1, y = 1.4;
    card(s, x, y, 3.85, 2.55, fill, `set-${name}`);
    T(s, name, { x: x + 0.25, y: y + 0.2, w: 2.6, h: 0.45, fontSize: 19, bold: true, color: col });
    T(s, n, { x: x + 2.85, y: y + 0.12, w: 0.8, h: 0.6, fontFace: HEAD, fontSize: 28, bold: true, color: col, align: "right" });
    T(s, role, { x: x + 0.25, y: y + 0.68, w: 3.4, h: 0.35, fontSize: 14, bold: true, color: i === 1 ? C.amber : C.slate });
    T(s, body, { x: x + 0.25, y: y + 1.08, w: 3.4, h: 1.4, fontSize: 14 });
  });
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.deep }, align: "center" } });
  const rows = [[hdr("Grouped CV F1"), hdr("url_base"), hdr("url_engineered"), hdr("extended")],
    ...[["Logistic Regression", FS.lr], ["Decision Tree", FS.dt], ["Random Forest", FS.rf]].map(([m, v]) => [
      { text: m, options: { align: "left" } }, v.url_base, { text: v.url_engineered, options: { bold: true } }, v.extended])];
  s.addTable(rows, { x: MX, y: 4.3, w: 7.0, colW: [2.5, 1.5, 1.5, 1.5], fontFace: BODY, fontSize: 14, color: C.ink, align: "center",
    valign: "middle", rowH: 0.42, border: { type: "solid", pt: 0.75, color: C.grid }, objectName: "cv-by-set" });
  card(s, 7.95, 4.3, 4.75, 2.1, C.amberSoft, "null-result");
  T(s, "Null result, reported as such", { x: 8.2, y: 4.45, w: 4.3, h: 0.4, fontSize: 16, bold: true, color: C.amber });
  T(s, `The engineered features gave no measurable gain: DT and RF are unchanged, and LR is ${FS.lr.url_base} without them vs ${FS.lr.url_engineered} with them. url_engineered stays primary because it was fixed before the results (group confirmation pending).`,
    { x: 8.2, y: 4.9, w: 4.3, h: 1.45, fontSize: 14 });
  s.addNotes(`[1:15] Feature sets.
Three sets, all chosen on training rows only. url_base is the control, url_engineered the primary set, and extended adds IsHTTPS and page-content features only as a secondary comparison.
The engineered features are row-wise functions of the URL string, so computing them before the split leaks nothing. A label-free redundancy filter (|r| >= ${K.corr_threshold}) removed ${FS.removed_feature}; it is refitted inside every CV fold, and all ${TU.fs_folds_checked} fold selections matched.
Honest result: no measurable gain. Switching to url_base after seeing it score higher would be selecting on results. Keeping url_engineered as primary is a project-owner decision; the group still has to confirm it.
Extended reaches ${FS.dt.extended} to ${FS.rf.extended}, but only through collection-biased features.

Likely viva questions:
- Did your engineered features help? No. We report the null result. With the format bias, the base URL features already separate almost everything.
- Is feature selection leak-free? Yes: it uses no labels and is refitted inside each CV fold.`);
}

// ---------------------------------------------------------------- 5. models and tuning
{
  const s = content("Three official models, one tuning protocol", "Method");
  card(s, MX, 1.35, 12.1, 0.6, C.tealSoft, "protocol");
  T(s, `Training rows only · url_engineered · ${K.cv_folds} grouped folds · phishing F1 · one-standard-error rule fixed before each run`,
    { x: MX + 0.25, y: 1.35, w: 11.6, h: 0.6, fontSize: 15, bold: true, color: C.deep, valign: "middle" });
  const models = [
    ["Logistic Regression", `C = ${TU.lr_C}, L2, lbfgs`, `CV F1 ${TU.lr_f1} ± ${TU.lr_f1_std}. C sits at the upper edge of the range searched; an L1 search was stopped (no convergence).`],
    ["Decision Tree", `max_depth ${TU.dt_depth}`, `CV F1 ${TU.dt_f1}, ${TU.dt_leaves} leaves. The unlimited tree scores ${TU.dt_unl_f1} with ${TU.dt_unl_leaves} leaves.`],
    ["Random Forest", `${TU.rf_trees} trees, depth ${TU.rf_depth}, max_features ${TU.rf_max_features}`, `${TU.rf_n_candidates} candidates on a ${K.rf_subset_rows}-row grouped training subset (subset F1 ${TU.rf_subset_f1} ± ${TU.rf_subset_f1_std}).`],
  ];
  models.forEach(([name, cfg, body], i) => {
    const y = 2.15 + i * 1.42;
    card(s, MX, y, 6.6, 1.27, C.paper, `model-${i}`);
    T(s, name, { x: MX + 0.25, y: y + 0.14, w: 2.6, h: 0.4, fontSize: 17, bold: true, color: C.deep });
    T(s, cfg, { x: MX + 2.85, y: y + 0.14, w: 3.55, h: 0.4, fontSize: 15, bold: true, color: C.teal, align: "right" });
    T(s, body, { x: MX + 0.25, y: y + 0.56, w: 6.15, h: 0.68, fontSize: 13 });
  });
  T(s, `SVM is provisional: its C comes from an earlier ungrouped 2,000-row tuning (test F1 ${FS.svm_test_f1}), so it is reported separately.`,
    { x: MX, y: 6.4, w: 6.6, h: 0.5, fontSize: 12, color: C.slate });
  const imgW = 5.3, imgH = imgW * 611 / 947;
  s.addImage({ path: FIG("dt_complexity_curve.png"), x: 7.4, y: 2.25, w: imgW, h: imgH, altText: "Decision Tree grouped CV F1 against mean number of leaves; depth 20 is the chosen point within one SE of the best" });
  T(s, "Decision Tree complexity review: depth 20 is as good as the unlimited tree and smaller (reports/figures/dt_complexity_curve.png).",
    { x: 7.4, y: 2.25 + imgH + 0.12, w: imgW, h: 0.6, fontSize: 12, color: C.slate });
  s.addNotes(`[1:30] Models and tuning.
One protocol for all three official models: training rows, url_engineered, the same grouped folds, phishing F1, and the one-SE rule written down before each run.
LR: C = ${TU.lr_C}. Best mean F1 ${TU.lr_f1}, SE ${TU.lr_se}, threshold ${TU.lr_threshold}; C = 300 scored ${TU.lr_c300_f1}, just below, so ${TU.lr_C} was the only eligible value. It is at the edge of the range, and we say so.
DT: depth ${TU.dt_depth}, chosen by a complexity review (figure).
RF: bounded search on a ${K.rf_subset_rows}-row subset that keeps whole URL groups; the chosen forest was then trained and cross-validated on the full training split.
SVM is provisional and not in the official comparison.

Likely viva questions:
- What is the one-SE rule? Among candidates within one standard error of the best mean CV score, take the simplest. Smaller differences are not evidence.
- Why is C so large? The URL features are almost linearly separable because of the format bias; weak regularisation draws that boundary sharply.
- Why stop the L1 search? L1 at C = 0.1 ran 33 minutes on one fold without converging; a non-converged fit is not a valid candidate.
- Why tune RF on a subset? A full-data search costs about 100 forest fits; the subset keeps whole groups and the class balance.`);
}

// ---------------------------------------------------------------- 6. official comparison
pres.addSection({ title: "Results" });
{
  const s = content("All three models score about 0.99 on held-out URLs", "Results");
  T(s, `Grouped ${K.cv_folds}-fold CV on ${N.train_rows} training rows, then one evaluation of the frozen models on ${N.test_rows} test rows (url_engineered).`,
    { x: MX, y: 1.35, w: 12.1, h: 0.45, fontSize: 16, color: C.slate });
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.deep }, align: "center" } });
  const row = (name, m, fill) => [name, `${m.cv_f1} ± ${m.cv_f1_std}`, m.test_acc, m.test_precision, m.test_recall, { text: m.test_f1, options: { bold: true } }, m.test_auc || "—", m.fp, m.fn]
    .map((v, i) => typeof v === "object" ? { text: v.text, options: { ...v.options, fill: fill ? { color: fill } : undefined } } : { text: String(v), options: { align: i === 0 ? "left" : "center", fill: fill ? { color: fill } : undefined } });
  const rows = [
    ["Model", "CV F1 (mean ± std)", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "FP", "FN"].map(hdr),
    row("Logistic Regression", M.lr), row("Decision Tree", M.dt), row("Random Forest", M.rf),
    row("Format rule (learns nothing)", { ...M.rule, test_auc: "—" }, C.amberSoft),
  ];
  s.addTable(rows, { x: MX, y: 2.0, w: 12.1, colW: [3.1, 1.9, 1.05, 1.05, 1.05, 1.0, 1.05, 0.9, 1.0], fontFace: BODY, fontSize: 15,
    color: C.ink, valign: "middle", align: "center", rowH: 0.55, border: { type: "solid", pt: 0.75, color: C.grid }, objectName: "official-comparison" });
  T(s, "Test columns are on the held-out rows. Sources: reports/official_comparison.csv, reports/format_baseline.csv.",
    { x: MX, y: 4.95, w: 12.1, h: 0.35, fontSize: 12, color: C.slate });
  card(s, MX, 5.45, 12.1, 1.2, C.amberSoft, "warning");
  T(s, runs([["Near-perfect scores are a warning, not a result. ", { bold: true, color: C.amber }],
    ["The last row is a fixed rule with no learning, and it scores as high as the models. The next two slides explain why.", {}]]),
    { x: MX + 0.3, y: 5.55, w: 11.5, h: 1.0, fontSize: 17, valign: "middle" });
  s.addNotes(`[1:15] Official comparison.
Same grouped folds for every model, and the frozen models were evaluated once on the ${N.test_rows} test rows. We report all metrics, not just accuracy.
Logistic Regression has the best test F1 (${M.lr.test_f1}) with only ${M.lr.fp} false positives and ${M.lr.fn} false negatives. Random Forest ${M.rf.test_f1}, Decision Tree ${M.dt.test_f1}.
Then point at the shaded row: a rule that learns nothing reaches ${M.rule.test_f1}. That is the key to reading this table.

Likely viva questions:
- Why F1 for selection? Phishing is the positive class; F1 balances missed phishing and false alarms. All five metrics and the confusion counts are still reported.
- Why is the SVM not here? Its C came from the earlier ungrouped tuning; until it is retuned with grouped CV its numbers are provisional (test F1 ${FS.svm_test_f1}).`);
}

// ---------------------------------------------------------------- 7. central finding I
{
  const s = content("A rule that learns nothing already reaches F1 " + M.rule.test_f1, "Results");
  card(s, MX, 1.4, 5.2, 4.95, C.paper, "rule-card");
  T(s, "The fixed rule", { x: MX + 0.3, y: 1.6, w: 4.6, h: 0.4, fontSize: 16, bold: true, color: C.slate });
  T(s, "“not a bare https://www homepage → phishing”", { x: MX + 0.3, y: 2.0, w: 4.6, h: 0.8, fontSize: 18, bold: true, color: C.deep });
  T(s, M.rule.test_f1, { x: MX + 0.3, y: 2.85, w: 4.6, h: 0.9, fontFace: HEAD, fontSize: 54, bold: true, color: C.amber });
  T(s, `test F1 · precision ${M.rule.test_precision} · CV F1 ${M.rule.cv_f1} ± ${M.rule.cv_f1_std}`, { x: MX + 0.3, y: 3.8, w: 4.6, h: 0.4, fontSize: 14, color: C.slate });
  T(s, runs([[`${M.rule.legit_bare_share_train}`, { bold: true, color: C.deep }], [" of legitimate training URLs are bare https://www.<domain> homepages, but only ", {}],
    [`${M.rule.phish_bare_share_train}`, { bold: true, color: C.deep }], [" of phishing URLs are.", {}]]),
    { x: MX + 0.3, y: 4.4, w: 4.6, h: 1.0, fontSize: 15 });
  T(s, "Evidence-based finding about how the dataset was collected, not a group vote.", { x: MX + 0.3, y: 5.5, w: 4.6, h: 0.7, fontSize: 12, color: C.slate });
  const diffs = [["Decision Tree", M.dt_minus_rule_test_f1], ["Random Forest", M.rf_minus_rule_test_f1], ["Logistic Regression", M.lr_minus_rule_test_f1]];
  s.addChart(pres.charts.BAR, [{ name: "Test F1 minus format rule", labels: diffs.map((d) => d[0]), values: diffs.map((d) => Number(d[1])) }], {
    x: 6.1, y: 1.4, w: 6.6, h: 4.3, barDir: "bar", ...AXIS, showTitle: true, title: `Held-out F1 minus the rule's ${M.rule.test_f1}`,
    chartColors: [C.teal], invertedColors: [C.amber], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.0000;-0.0000;0.0000",
    valAxisMinVal: -0.005, valAxisMaxVal: 0.004, valAxisMajorUnit: 0.001, valAxisLabelFormatCode: "+0.000;-0.000;0", catAxisLabelPos: "low",
    barGapWidthPct: 60, objectName: "f1-minus-rule" });
  card(s, 6.1, 5.85, 6.6, 0.85, C.amberSoft, "message");
  T(s, "Most of the 0.99 measures the dataset's collection format, not phishing cues.", { x: 6.35, y: 5.85, w: 6.2, h: 0.85, fontSize: 16, bold: true, color: C.ink, valign: "middle" });
  s.addNotes(`[2:00] Central finding, part 1. Take time here; this is the scientific core of the talk.
In PhiUSIIL every legitimate URL is a bare https://www.<domain> homepage, with no path or query: a property of how the legitimate class was collected. Only ${M.rule.phish_bare_share_train} of phishing URLs look like that.
So a one-line rule that learns nothing gets test F1 ${M.rule.test_f1} with precision ${M.rule.test_precision}. Logistic Regression beats it by only ${M.lr_minus_rule_test_f1}, the Random Forest by ${M.rf_minus_rule_test_f1}, and the Decision Tree is ${M.dt_minus_rule_test_f1.replace("-", "")} below it.
This is an evidence-based finding from the data, not something the group voted on.

Likely viva questions:
- Why are your scores so high; is it leakage? We removed the real leaks (URLSimilarityIndex, near-duplicates across the split). What remains is collection bias, and we show the rule next to every URL-only result.
- Is collection bias the same as leakage? No. Leakage is label information unavailable at prediction time. The URL format is available at prediction time, but in this dataset it correlates with the label only because of how the two classes were collected. It inflates scores the same way.`);
}

// ---------------------------------------------------------------- 8. central finding II
{
  const s = content("Without the format cue, models catch about half", "Results");
  T(s, `${FC.bare_rows} held-out URLs are bare https://www homepages: ${FC.bare_legit} legitimate and ${FC.bare_phishing} phishing. Here the format says “legitimate” for all of them.`,
    { x: MX, y: 1.35, w: 12.1, h: 0.75, fontSize: 17 });
  const names = ["Format rule", "Logistic Regression", "Decision Tree", "Random Forest"];
  const keys = ["rule", "lr", "dt", "rf"];
  s.addChart(pres.charts.BAR, [{ name: "Phishing URLs caught", labels: names, values: keys.map((k) => num(FC[k].bare_tp)) }], {
    x: MX, y: 2.2, w: 7.2, h: 4.5, barDir: "col", ...AXIS, showTitle: true, title: `Bare-homepage phishing URLs caught (of ${FC.bare_phishing})`,
    chartColors: [C.slate, C.teal, C.teal, C.teal], showValue: true, dataLabelPosition: "outEnd", valAxisMinVal: 0, valAxisMaxVal: num(FC.bare_phishing),
    valAxisMajorUnit: 50, barGapWidthPct: 55, objectName: "bare-homepage-recall" });
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.deep }, align: "center" } });
  const rows = [[hdr("On these URLs"), hdr("Caught"), hdr("False alarms")],
    ...keys.map((k, i) => [{ text: names[i], options: { align: "left" } }, FC[k].bare_tp, FC[k].bare_fp])];
  s.addTable(rows, { x: 8.1, y: 2.3, w: 4.6, colW: [2.2, 1.1, 1.3], fontFace: BODY, fontSize: 14, color: C.ink, align: "center",
    valign: "middle", rowH: 0.42, border: { type: "solid", pt: 0.75, color: C.grid }, objectName: "bare-homepage-table" });
  card(s, 8.1, 4.65, 4.6, 2.05, C.amberSoft, "consequence");
  T(s, bullets([`${FC.lr.bare_fn} of LR's ${M.lr.fn} missed phishing URLs are in this subset.`,
    "No real-world performance claim: PhiUSIIL has no legitimate URLs with paths or queries to test it."], 14),
    { x: 8.3, y: 4.8, w: 4.25, h: 1.8 });
  s.addNotes(`[1:30] Central finding, part 2.
Now look only at the ${FC.bare_rows} held-out URLs that have the legitimate format. Here the format cue cannot help.
The rule finds none of the ${FC.bare_phishing} phishing URLs. LR finds ${FC.lr.bare_tp} with ${FC.lr.bare_fp} false alarms, DT ${FC.dt.bare_tp} with ${FC.dt.bare_fp}, RF ${FC.rf.bare_tp} with ${FC.rf.bare_fp}. So the models learned some real signal beyond the format, but on these hard cases they catch only about half.
${FC.lr.bare_fn} of Logistic Regression's ${M.lr.fn} false negatives are exactly these URLs.
Consequence: we do not claim real-world performance, and we cannot test generalisation inside PhiUSIIL because it contains no legitimate deep links. That is why an independently collected dataset is our main future work.

Likely viva question: Could you fix this by dropping a column? No. The bias is in which URLs were collected, not in one feature; it can only be tested on differently collected data.`);
}

// ---------------------------------------------------------------- 9. risk score
pres.addSection({ title: "Risk and explainability" });
{
  const s = content("Calibrated risk score and bands", "Risk and explainability");
  T(s, bullets([`Served score: Random Forest probability, ${K.calibration}-calibrated on training rows with the grouped folds.`,
    `Held-out Brier score ${N.cal_test_brier}.`,
    `Bands: Low < ${K.low_max} ≤ Medium < ${K.high_min} ≤ High.`], 16), { x: MX, y: 1.4, w: 5.6, h: 1.9 });
  const bands = [["Low", RK.low, C.teal], ["Medium", RK.medium, C.mist], ["High", RK.high, C.amber]];
  bands.forEach(([name, b, col], i) => {
    const x = 6.55 + i * 2.1, y = 1.4;
    card(s, x, y, 1.95, 2.15, C.paper, `band-${name}`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: y + 0.25, w: 0.3, h: 0.3, fill: { color: col }, line: { color: col } });
    T(s, name, { x: x + 0.6, y: y + 0.2, w: 1.3, h: 0.4, fontSize: 17, bold: true, color: C.deep });
    T(s, b.share, { x: x + 0.2, y: y + 0.7, w: 1.6, h: 0.55, fontFace: HEAD, fontSize: 26, bold: true, color: C.ink });
    T(s, `of URLs\n${b.phishing_rate} phishing`, { x: x + 0.2, y: y + 1.27, w: 1.6, h: 0.75, fontSize: 13, color: C.slate });
  });
  card(s, MX, 3.8, 12.1, 1.35, C.amberSoft, "disclosure");
  T(s, runs([[`${RK.low.n_phishing} of the ${RK.total_phishing_test} held-out phishing URLs fall in Low, and ${RK.high.n_legit} legitimate URLs fall in High. `, { bold: true }],
    ["“Low” means low estimated risk, not “safe”.", {}]]), { x: MX + 0.3, y: 3.9, w: 11.5, h: 1.15, fontSize: 18, valign: "middle" });
  T(s, runs([["Calibration caveat: ", { bold: true, color: C.deep }],
    ["sigmoid was re-checked on a 20,000-row development sample, and the internal calibration CV folds of that check were ungrouped. It was not repeated on the full training split.", {}]]),
    { x: MX, y: 5.4, w: 12.1, h: 0.75, fontSize: 15 });
  T(s, `The ${RK.low.n_phishing} Low-band phishing URLs are a band count from the calibrated forest, not LR's ${M.lr.fn} false negatives; the counts are equal by coincidence. Calibration method and bands: project-owner decision, group confirmation pending.`,
    { x: MX, y: 6.2, w: 12.1, h: 0.6, fontSize: 12, color: C.slate });
  s.addNotes(`[1:30] Risk score.
The score users see is the Random Forest probability after sigmoid calibration, fitted only on training rows with the grouped folds. On the held-out rows, Brier score ${N.cal_test_brier}.
The bands match their names: Low is ${RK.low.phishing_rate} phishing, High ${RK.high.phishing_rate} phishing, Medium holds the genuinely uncertain ${RK.medium.share} of URLs.
Say this explicitly: ${RK.low.n_phishing} held-out phishing URLs (${RK.low_phishing_share_of_phishing} of them) are in Low, and ${RK.high.n_legit} legitimate URLs are in High. Low is not "safe".
Do not mix this ${RK.low.n_phishing} up with LR's ${M.lr.fn} false negatives: different model, different measure; the sets only partly overlap.
Caveat: sigmoid was re-checked on a 20,000-row development sample whose internal calibration folds were ungrouped.

Likely viva questions:
- How is the score calibrated without touching the test set? CalibratedClassifierCV fitted on training rows with the grouped folds. The method check was on a 20,000-row development sample with ungrouped inner calibration folds; we disclose that.
- Why use bands if ${RK.low.n_phishing} phishing URLs land in Low? We disclose it; the bands communicate estimated risk, not a safety guarantee. The method and bands still need group confirmation.`);
}

// ---------------------------------------------------------------- 10. explainability
{
  const s = content("What drives the score: SHAP explanations", "Risk and explainability");
  const ih = 5.45, iw = ih * 1185 / 1110;
  s.addImage({ path: FIG("shap_summary.png"), x: MX, y: 1.35, w: iw, h: ih, altText: "SHAP summary of the Random Forest on held-out URLs; letter ratio, special-character ratio and other special characters rank highest" });
  const x = MX + iw + 0.45, w = W - MX - x;
  T(s, "Top features", { x, y: 1.4, w, h: 0.4, fontSize: 17, bold: true, color: C.deep });
  T(s, RK.shap_top.slice(0, 5).map((f, i) => ({ text: f, options: { bullet: { type: "number" }, breakLine: i < 4, fontSize: 14 } })), { x, y: 1.85, w, h: 1.7 });
  T(s, "These character-composition features also encode “bare homepage vs URL with a path or query”, so the explanations agree with the bias finding.",
    { x, y: 3.45, w, h: 1.0, fontSize: 15 });
  card(s, x, 4.75, w, 1.65, C.tealSoft, "demo");
  T(s, runs([["demo.py, held-out phishing URL", { bold: true, color: C.deep, breakLine: true }],
    [`risk ${D.score} → ${D.band}; predicted ${D.predicted}`, { breakLine: true }],
    ...D.top.map((t, i) => [`${t.feature}  +${t.contribution}`, { breakLine: i < D.top.length - 1, color: C.slate }])]),
    { x: x + 0.2, y: 4.85, w: w - 0.4, h: 1.45, fontSize: 14 });
  s.addNotes(`[1:15] Explainability.
SHAP values of the Random Forest on 500 held-out URLs (TreeExplainer). The top features are character-composition features: ${RK.shap_top.slice(0, 5).join(", ")}.
Be clear that these features are exactly the ones that separate a bare homepage from a URL with a path or query, so the explanations support the bias finding rather than contradict it.
Demo (saved output in ${D.source}): for one held-out phishing URL the score is ${D.score}, band ${D.band}, with the top three contributions shown. The URL itself is not shown on the slide because it may still be live.
python demo.py takes about 90 seconds to start; run it before the slot if the panel asks for it live. It scores held-out test URLs only.

Likely viva question: Why can't the demo score any URL? Several inputs (TLDLegitimateProb, URLCharProb) come from the dataset authors' feature extractor, which we do not reproduce, so scoring new URLs would be an unsupported claim.`);
}

// ---------------------------------------------------------------- 11. limitations and future work
pres.addSection({ title: "Wrap-up" });
{
  const s = content("Limitations and next steps", "Wrap-up");
  card(s, MX, 1.4, 7.1, 5.3, C.paper, "limitations");
  T(s, "Limitations", { x: MX + 0.3, y: 1.55, w: 6.5, h: 0.45, fontSize: 19, bold: true, color: C.deep });
  T(s, bullets([
    "Main: URL-format collection bias. The URL-only scores mostly measure how the dataset was collected.",
    "extended set: IsHTTPS and page features carry the same bias.",
    `LR's C = ${TU.lr_C} is at the edge of the range searched.`,
    "Random Forest tuned on a training subset.",
    "Calibration re-checked on a 20,000-row sample with ungrouped inner folds.",
    "Domain-level grouping deferred; SVM still provisional.",
  ], 17), { x: MX + 0.3, y: 2.1, w: 6.55, h: 4.5 });
  card(s, 8.0, 1.4, 4.7, 2.55, C.tealSoft, "remaining");
  T(s, "Remaining for the final report", { x: 8.25, y: 1.55, w: 4.3, h: 0.45, fontSize: 17, bold: true, color: C.deep });
  T(s, bullets(["Grouped retuning of the SVM", "Decision on domain-level grouping", "Final analysis and report"], 15), { x: 8.25, y: 2.05, w: 4.3, h: 1.8 });
  card(s, 8.0, 4.15, 4.7, 2.55, C.amberSoft, "future");
  T(s, "Future work", { x: 8.25, y: 4.3, w: 4.3, h: 0.45, fontSize: 17, bold: true, color: C.amber });
  T(s, "Evaluate the frozen models on an independently collected dataset: legitimate URLs with paths and queries, and phishing from another source and period.",
    { x: 8.25, y: 4.8, w: 4.3, h: 1.8, fontSize: 15 });
  s.addNotes(`[1:00] Limitations and next steps.
The main limitation is the collection bias from the previous slides. The others are smaller and all disclosed: LR's C at the edge, RF tuned on a subset, the calibration check on a 20,000-row sample with ungrouped inner folds, domain-level grouping deferred, and the SVM still provisional.
Remaining before the final report: SVM grouped retune, the domain-grouping decision and the final analysis. Future work, beyond this project's scope: an independently collected dataset to test generalisation.

Likely viva question: What would you do next? Evaluate the frozen models on independently collected data; PhiUSIIL itself has no legitimate non-homepage URLs, so the bias cannot be tested inside it.`);
}

// ---------------------------------------------------------------- 12. contributions
{
  const s = content("Contributions and status", "Wrap-up");
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.deep }, align: "left" } });
  const rows = [[hdr("Member"), hdr("Model"), hdr("Original contribution (own commits)")],
    ["Madushanka D.M.C", "Logistic Regression", "Data inspection and cleaning, first LR tuning, workflow documentation, branch integration"],
    ["Madushan S.C", "Decision Tree", "URL feature engineering, feature selection, tuned Decision Tree experiment"],
    ["Dharmasiri S.O.R", "SVM", "Shared evaluation utilities (ROC, confusion matrices, feature-set comparison), calibrated linear SVM"],
    ["Thanthrige D.P", "Random Forest", "Scaffold and rules, URLSimilarityIndex leakage investigation, RF tuning, risk scoring and calibration, SHAP, main.py"]];
  s.addTable(rows, { x: MX, y: 1.4, w: 12.1, colW: [2.4, 2.2, 7.5], fontFace: BODY, fontSize: 14, color: C.ink, valign: "middle",
    rowH: 0.5, border: { type: "solid", pt: 0.75, color: C.grid }, objectName: "contributions" });
  card(s, MX, 4.45, 7.3, 2.25, C.paper, "integration");
  T(s, "Final integration and grouped-CV retuning, 24 Nov 2026", { x: MX + 0.3, y: 4.6, w: 6.8, h: 0.4, fontSize: 16, bold: true, color: C.deep });
  T(s, "Madushanka D.M.C merged the member branches and re-ran the agreed protocol across all modules: URL grouping, grouped split and CV, feature-set freeze, LR/DT/RF retuning, format baseline, calibration, SHAP, demo and unit tests. Not yet committed.",
    { x: MX + 0.3, y: 5.05, w: 6.8, h: 1.55, fontSize: 14 });
  card(s, 8.15, 4.45, 4.55, 2.25, C.amberSoft, "pending");
  T(s, "Pending group confirmation", { x: 8.4, y: 4.6, w: 4.1, h: 0.4, fontSize: 16, bold: true, color: C.amber });
  T(s, bullets(["url_engineered as primary set", "Calibration method and risk bands", "This contribution table", "Who presents which slide"], 14), { x: 8.4, y: 5.05, w: 4.1, h: 1.55 });
  s.addNotes(`[0:45] Contributions.
Original work, as each member's own commits: Madushanka (e86f144, 78ae724, 6ceada1, 3eefb14, 0888f07, 88804ba), Madushan (7973c24), Dharmasiri (e6cbe11, 4392061, 643c94e), Thanthrige (25e26f5, a74ec83, dc5e34a, 87e91ca, 752c23d, bd3e196, 2569c27, b90a58c, 736ee33, 2593aec).
The 24 November integration and grouped-CV retuning was done by Madushanka D.M.C and is not committed yet. The members have not yet confirmed this table, the speaker allocation, the primary-set choice or the calibration and bands; do not present any of these as agreed until they are.
Close: thank the panel and invite questions.`);
}

pres.writeFile({ fileName: OUT }).then(async (file) => {
  if (process.env.APPLY_THEME) {
    const { applyTheme } = require(process.env.APPLY_THEME);
    await applyTheme(file, THEME);
  }
  console.log("wrote " + file);
});
