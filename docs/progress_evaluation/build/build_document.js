// Two-page progress document (A4) for the URLSentra progress evaluation.
//
//   python docs/progress_evaluation/build/collect_numbers.py   # numbers.json from reports/
//   node docs/progress_evaluation/build/build_document.js      # writes ../URLSentra_progress_document.docx
//
// Needs the npm package "docx" (9.x). Every number comes from numbers.json; none is typed here.
// Not part of the ML pipeline.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, AlignmentType,
  BorderStyle, ShadingType, LevelFormat, TableLayoutType,
} = require("docx");

const N = JSON.parse(fs.readFileSync(path.join(__dirname, "numbers.json"), "utf8"));
const OUT = path.join(__dirname, "..", "URLSentra_progress_document.docx");
const M = N.models, FC = N.format_controlled, FS = N.feature_sets, TU = N.tuning, RK = N.risk, K = N.constants;

const FONT = "Calibri";
const BODY = 22;      // half-points: 11 pt
const SMALL = 18;     // tables 9 pt
const INK = "1F2A30", ACCENT = "0B5563", MUTED = "5A6670", RULE = "B8C4CA", HEAD_FILL = "E3EEF0", RULE_FILL = "F4F4F4";
const CONTENT_W = 11906 - 2 * 1080;   // A4 width minus 0.75" margins

// "**bold**" and "`code`" inline markup -> TextRuns
function runs(text, opts = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  const push = (t, extra) => t && out.push(new TextRun({ text: t, font: FONT, size: opts.size || BODY, color: opts.color || INK, bold: opts.bold, italics: opts.italics, ...extra }));
  while ((m = re.exec(text))) {
    push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) push(tok.slice(2, -2), { bold: true });
    else push(tok.slice(1, -1), { font: "Consolas", size: (opts.size || BODY) - 2 });
    last = m.index + tok.length;
  }
  push(text.slice(last));
  return out;
}

const para = (text, o = {}) => new Paragraph({
  children: runs(text, o), alignment: o.align || AlignmentType.JUSTIFIED,
  spacing: { after: o.after ?? 70, before: o.before ?? 0, line: 247 },
});
const heading = (text) => new Paragraph({
  children: [new TextRun({ text, font: FONT, size: 23, bold: true, color: ACCENT })],
  spacing: { before: 110, after: 40 }, keepNext: true,
});
const bullet = (text) => new Paragraph({
  children: runs(text), numbering: { reference: "dots", level: 0 },
  alignment: AlignmentType.LEFT, spacing: { after: 25, line: 247 },
});

function table(widths, header, rows, { shadeLast = false, textAlign = AlignmentType.CENTER } = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: RULE };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (text, w, { head = false, fill, align } = {}) => new TableCell({
    width: { size: w, type: WidthType.DXA }, borders,
    shading: fill ? { type: ShadingType.CLEAR, color: "auto", fill } : undefined,
    margins: { top: 25, bottom: 25, left: 60, right: 60 },
    children: [new Paragraph({ alignment: align || AlignmentType.LEFT, spacing: { after: 0, line: 228 },
      children: runs(text, { size: SMALL, bold: head, color: head ? ACCENT : INK }) })],
  });
  const tr = (cells, opts) => new TableRow({ cantSplit: true, children: cells.map((c, i) =>
    cell(c, widths[i], { ...opts, align: i === 0 ? AlignmentType.LEFT : opts.align })) });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED,
    rows: [tr(header, { head: true, fill: HEAD_FILL, align: AlignmentType.CENTER }),
      ...rows.map((r, i) => tr(r, { fill: shadeLast && i === rows.length - 1 ? RULE_FILL : undefined, align: textAlign }))],
  });
}
const caption = (text) => new Paragraph({ children: runs(text, { size: 18, color: MUTED, italics: true }),
  spacing: { before: 30, after: 70 }, keepNext: false });
const capAbove = (text) => new Paragraph({ children: runs(text, { size: 18, color: MUTED, bold: true }),
  spacing: { before: 40, after: 30 }, keepNext: true });

// ---------------------------------------------------------------- content
const c = [];
c.push(new Paragraph({ children: [new TextRun({ text: "URLSentra: Explainable Phishing URL Risk Scoring", font: FONT, size: 32, bold: true, color: ACCENT })], spacing: { after: 20 } }));
c.push(para("Progress report · EC5203 Machine Learning mini project · Group 42 · University of Ruhuna · 24 November 2026", { color: MUTED, size: 19, after: 10, align: AlignmentType.LEFT }));
c.push(para("Members: Madushanka D.M.C · Madushan S.C · Dharmasiri S.O.R · Thanthrige D.P", { color: MUTED, size: 19, after: 60, align: AlignmentType.LEFT }));

c.push(heading("1. Problem and objective"));
c.push(para(`Users and mail or browser filters must judge a link before it is opened. URLSentra trains classical models on URL features, turns the phishing probability into a calibrated risk score with Low / Medium / High bands, and explains each score with SHAP. The aim of this stage is a **defensible evaluation**, not the highest possible score.`));

c.push(heading("2. Data, preprocessing and leakage control"));
c.push(para(`PhiUSIIL (UCI #967) has ${N.raw_rows} URLs, ${N.raw_cols} columns and ${N.missing_values} missing values. The label is flipped so that 1 = phishing (the positive class). Removing ${N.repeated_urls_removed} repeated URLs (first copy kept) leaves ${N.rows_clean} rows (${N.clean_phishing} phishing, ${N.clean_legit} legitimate). URLs that differ only in scheme, a leading “www.”, hostname case or a trailing slash share a **URL group**: ${N.groups} groups, ${N.multi_groups} with several rows and ${N.conflict_groups} with conflicting labels (${N.conflict_rows} rows, kept as label noise). The test set is one fold of a StratifiedGroupKFold (seed ${K.seed}): ${N.train_rows} training and ${N.test_rows} test rows, ${N.shared_groups} shared groups, phishing share ${N.phish_share_train} in both. All tuning and cross-validation use the same ${K.cv_folds} grouped folds of the training rows, and the test rows were used once. \`URLSimilarityIndex\` is excluded from every feature set because it equals 100 for every legitimate URL, which makes it a proxy for the label. Domain-level grouping is deferred for this evaluation, not rejected.`));

c.push(heading("3. Feature sets and models"));
c.push(para(`**url_base** (${FS.n_features.url_base} URL-string features, without \`IsHTTPS\`) is the control. **url_engineered** (${FS.n_features.url_engineered}) adds path depth, longest path segment and query-field count and is the **primary** set. **extended** (${FS.n_features.extended}) adds \`IsHTTPS\` and ${FS.n_page_features} page-content features and is secondary only. The engineered features are computed from each URL string alone. A label-free redundancy filter (|r| ≥ ${K.corr_threshold}) removed \`${FS.removed_feature}\` (r = ${FS.removed_r}); it is refitted inside every CV fold, and all ${TU.fs_folds_checked} fold selections matched. Models were tuned on the training rows with phishing F1 and a one-standard-error rule fixed before each run. **Logistic Regression:** L2, C ${TU.lr_cmin}–${TU.lr_cmax}, ${TU.lr_n_fits} fits, all converged; C = ${TU.lr_C} (F1 ${TU.lr_f1} ± ${TU.lr_f1_std}) lies at the upper edge of the range, and an L1 search was stopped because it did not converge. **Decision Tree:** max_depth ${TU.dt_depth} (F1 ${TU.dt_f1}, ${TU.dt_leaves} leaves; the unlimited tree has F1 ${TU.dt_unl_f1} with ${TU.dt_unl_leaves} leaves). **Random Forest:** ${TU.rf_trees} trees, depth ${TU.rf_depth}, max_features ${TU.rf_max_features}, from ${TU.rf_n_candidates} candidates on a ${K.rf_subset_rows}-row group-preserving training subset (subset F1 ${TU.rf_subset_f1} ± ${TU.rf_subset_f1_std}). The **SVM is provisional** (C from an earlier ungrouped 2,000-row tuning; test F1 ${FS.svm_test_f1}) and is not part of the official comparison.`));

c.push(heading("4. Results"));
c.push(capAbove(`Table 1. Official comparison on url_engineered: grouped ${K.cv_folds}-fold CV (${N.train_rows} training rows) and held-out test (${N.test_rows} rows)`));
const mrow = (name, m, cvAuc = true) => [name, `${m.cv_f1} ± ${m.cv_f1_std}`, m.test_acc, m.test_precision, m.test_recall, `**${m.test_f1}**`, m.test_auc ?? "—", m.fp, m.fn];
c.push(table([2330, 1640, 880, 880, 880, 800, 880, 700, 756],
  ["Model", "CV F1 (mean ± std)", "Test acc.", "Precision", "Recall", "F1", "ROC-AUC", "FP", "FN"],
  [mrow("Logistic Regression", M.lr), mrow("Decision Tree", M.dt), mrow("Random Forest", M.rf),
   mrow("Format rule (no learning)", { ...M.rule, test_auc: "—" })], { shadeLast: true }));
c.push(caption(`Sources: reports/official_comparison.csv, reports/format_baseline.csv. SVM (provisional) is reported separately in reports/results.csv.`));
c.push(para(`Grouped CV F1 for url_base / url_engineered / extended: LR ${FS.lr.url_base} / ${FS.lr.url_engineered} / ${FS.lr.extended}, DT ${FS.dt.url_base} / ${FS.dt.url_engineered} / ${FS.dt.extended}, RF ${FS.rf.url_base} / ${FS.rf.url_engineered} / ${FS.rf.extended}. **The engineered features gave no measurable improvement**: DT and RF are unchanged, and LR scores higher without them (${FS.lr.url_base} vs ${FS.lr.url_engineered}, ${FS.lr_base_gain_in_fold_std} fold standard deviations). url_engineered remains primary because it was fixed as primary before the results; switching afterwards would be selection on results (project-owner decision, group confirmation pending). The extended set reaches ${FS.dt.extended}–${FS.rf.extended} only through the collection-biased \`IsHTTPS\` and page features.`));

c.push(heading("5. Main limitation: dataset collection bias"));
c.push(para(`This is an evidence-based finding, not a group vote. **Every legitimate training URL is a bare https://www.<domain> homepage**, while only ${M.rule.phish_bare_share_train} of phishing URLs are. The fixed rule “not a bare https://www homepage → phishing” learns nothing, yet it reaches grouped CV F1 ${M.rule.cv_f1} and test F1 ${M.rule.test_f1} (precision ${M.rule.test_precision}, ${M.rule.fn} missed phishing URLs). On test F1, LR beats it by only ${M.lr_minus_rule_test_f1}, RF by ${M.rf_minus_rule_test_f1}, and DT falls ${M.dt_minus_rule_test_f1.replace("-", "")} below it. On the ${FC.bare_rows} held-out bare-homepage URLs (${FC.bare_legit} legitimate, ${FC.bare_phishing} phishing), where the format cue is absent, the rule finds no phishing, while LR finds ${FC.lr.bare_tp} (${FC.lr.bare_fp} false alarms), DT ${FC.dt.bare_tp} (${FC.dt.bare_fp}) and RF ${FC.rf.bare_tp} (${FC.rf.bare_fp}); ${FC.lr.bare_fn} of LR's ${M.lr.fn} misses are in this subset. So the models learn some signal beyond the format but catch only about half of these hard cases, and the near-perfect scores mostly measure how the dataset was collected. They are **not evidence of real-world performance**. This cannot be tested inside PhiUSIIL, which has no legitimate URLs with paths or queries.`));

c.push(heading("6. Risk score and explainability"));
c.push(para(`The served score is the Random Forest probability, ${K.calibration}-calibrated on training rows with the grouped folds (held-out Brier ${N.cal_test_brier}). Bands are Low < ${K.low_max} ≤ Medium < ${K.high_min} ≤ High. On the test rows, Low holds ${RK.low.share} of URLs (${RK.low.phishing_rate} phishing), Medium ${RK.medium.share} (${RK.medium.phishing_rate}) and High ${RK.high.share} (${RK.high.phishing_rate}). **${RK.low.n_phishing} of the ${RK.total_phishing_test} held-out phishing URLs fall in Low and ${RK.high.n_legit} legitimate URLs in High**, so “Low” means low estimated risk, not “safe”. These ${RK.low.n_phishing} are a band count from the calibrated forest, not LR's ${M.lr.fn} false negatives; the counts are equal by coincidence and the sets only partly overlap. Sigmoid was chosen over isotonic and no calibration by out-of-fold Brier score and log loss, re-checked on a 20,000-row development sample whose **internal calibration CV folds were ungrouped**; it was not repeated on the full training split. SHAP ranks ${RK.shap_top.slice(0, 5).map((f) => `\`${f}\``).join(", ")} highest. These character-composition features also encode the homepage format, so the explanations agree with the bias finding. \`demo.py\` shows the score, band and top SHAP contributions for held-out URLs only.`));

c.push(heading("7. Individual contributions"));
c.push(table([1700, 1500, 6546],
  ["Member", "Model", "Original contribution (commits preserved as authored)"],
  [["Madushanka D.M.C", "Logistic Regression", "Data inspection and cleaning, first LR tuning, workflow documentation, branch integration (e86f144, 78ae724, 6ceada1, 3eefb14, 0888f07, 88804ba)"],
   ["Madushan S.C", "Decision Tree", "URL feature engineering, feature selection, tuned Decision Tree experiment (7973c24)"],
   ["Dharmasiri S.O.R", "SVM", "Shared evaluation utilities (ROC, confusion matrices, feature-set comparison), calibrated linear SVM with train-only CV tuning (e6cbe11, 4392061, 643c94e)"],
   ["Thanthrige D.P", "Random Forest", "Project scaffold and rules, leakage investigation (URLSimilarityIndex), RF tuning, risk scoring and calibration, SHAP explanations, main.py integration (25e26f5, a74ec83, dc5e34a, 87e91ca, 752c23d, bd3e196, 2569c27, b90a58c, 736ee33, 2593aec)"]],
  { textAlign: AlignmentType.LEFT }));
c.push(para(`**Final integration and grouped-CV retuning (24 November 2026):** Madushanka D.M.C merged the member branches and re-ran the agreed protocol across all modules: URL grouping and grouped split/CV, feature-set freeze, LR/DT/RF retuning, format baseline, calibration re-run, SHAP, \`demo.py\` and 64 unit tests. This work is not yet committed, and the members have not yet confirmed this table.`, { before: 60 }));

c.push(heading("8. Status and next steps"));
c.push(bullet(`**Done:** cleaning, URL grouping, grouped split and CV, feature sets, three tuned models evaluated on the full data, format baseline, calibrated risk score, SHAP, demo and unit tests.`));
c.push(bullet(`**Remaining:** grouped retuning of the SVM, the decision on domain-level grouping, and the final report.`));
c.push(bullet(`**Future work:** evaluate the frozen models on an independently collected dataset (legitimate deep links, phishing from another source and period) before any real-world claim.`));
c.push(bullet(`**Pending group confirmation:** url_engineered as the primary set, the calibration method and risk bands, and the contribution table.`));

const doc = new Document({
  creator: "URLSentra Group 42", title: "URLSentra progress report", description: "EC5203 progress evaluation, two-page document",
  styles: { default: { document: { run: { font: FONT, size: BODY } } } },
  numbering: { config: [{ reference: "dots", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 300, hanging: 200 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } } }, children: c }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote " + OUT); });
