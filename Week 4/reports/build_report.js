const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow,
  TableCell, WidthType, ShadingType, AlignmentType, BorderStyle, ImageRun,
} = require("docx");
const fs = require("fs");

const PAGE_WIDTH_DXA = 12240; // US Letter
const PAGE_HEIGHT_DXA = 15840;
const MARGIN = 1080; // 0.75in

function heading(text, level = HeadingLevel.HEADING_1) {
  return new Paragraph({ text, heading: level, spacing: { before: 200, after: 100 } });
}

function bodyText(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...opts })],
    spacing: { after: 120 },
  });
}

function bullet(text) {
  return new Paragraph({
    text,
    bullet: { level: 0 },
    spacing: { after: 60 },
  });
}

function cell(text, { bold = false, width, shade } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text, bold })] })],
  });
}

const colWidths = [2400, 1700, 1600, 1700, 1500, 1240];

const headerRow = new TableRow({
  tableHeader: true,
  children: [
    cell("Model", { bold: true, width: colWidths[0], shade: "D9E2F3" }),
    cell("CV F1 (mean±std)", { bold: true, width: colWidths[1], shade: "D9E2F3" }),
    cell("Test Acc.", { bold: true, width: colWidths[2], shade: "D9E2F3" }),
    cell("Test Prec.", { bold: true, width: colWidths[3], shade: "D9E2F3" }),
    cell("Test Recall", { bold: true, width: colWidths[4], shade: "D9E2F3" }),
    cell("Test F1", { bold: true, width: colWidths[5], shade: "D9E2F3" }),
  ],
});

function dataRow(vals, shade) {
  return new TableRow({
    children: vals.map((v, i) => cell(v, { width: colWidths[i], shade })),
  });
}

const rows = [
  headerRow,
  dataRow(["LogReg (Week 3 baseline, single split)", "—", "—", "—", "—", "0.761"]),
  dataRow(["LogReg (Week 4, 5-fold CV)", "0.739 ± 0.019", "0.793", "0.722", "0.754", "0.738"], "F2F2F2"),
  dataRow(["Random Forest (untuned, 5-fold CV)", "0.757 ± 0.025", "0.805", "0.774", "0.696", "0.733"]),
  dataRow(["Random Forest (GridSearchCV-tuned)", "0.763", "0.793", "0.750", "0.696", "0.722"], "F2F2F2"),
  dataRow(["Gradient Boosting (RandomizedSearchCV-tuned) — FINAL", "0.769", "0.816", "0.810", "0.681", "0.740"], "C6E0B4"),
];

const resultsTable = new Table({
  width: { size: colWidths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
  columnWidths: colWidths,
  rows,
});

const cvImg = fs.readFileSync("../outputs/figures/cv_f1_comparison.png");
const cmImg = fs.readFileSync("../outputs/figures/confusion_matrix_best_model.png");
const fiImg = fs.readFileSync("../outputs/figures/feature_importance_best_model.png");

const conclusionText = fs.readFileSync("conclusion.md", "utf-8")
  .split("\n")
  .slice(2)
  .join("\n")
  .trim();

const conclusionParagraphs = conclusionText
  .split("\n\n")
  .map((p) => bodyText(p.replace(/\n/g, " ")));

const doc = new Document({
  sections: [
    {
      properties: {
        page: {
          size: { width: PAGE_WIDTH_DXA, height: PAGE_HEIGHT_DXA },
          margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN },
        },
      },
      children: [
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Advance Soft Logics — AI/ML Internship", bold: true, size: 20 })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
          children: [new TextRun({ text: "Week 4 Report: Model Evaluation, Hyperparameter Tuning & Ensemble Methods", bold: true, size: 32 })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 300 },
          children: [new TextRun({ text: "Hashir Ahmed", italics: true, size: 22 })],
        }),

        heading("What I Built", HeadingLevel.HEADING_1),
        bodyText(
          "I extended the Week 3 Titanic survival-classification project with a proper model-selection workflow: 5-fold stratified cross-validation for the baseline models, a GridSearchCV hyperparameter search over Random Forest, a RandomizedSearchCV search over Gradient Boosting, and a final results table comparing every model — including the Week 3 baseline — on the same held-out test set. All preprocessing (imputation, scaling, one-hot encoding) stays inside a single scikit-learn Pipeline, so every cross-validation fold and every search candidate refits preprocessing on training data only, with no leakage from validation or test data."
        ),
        bodyText(
          "The final model was selected using cross-validated F1 only, before the test set was touched — Gradient Boosting (mean CV F1 = 0.769) was chosen over the tuned Random Forest (0.763). The chosen pipeline was then evaluated once on the untouched test set (accuracy 0.816, F1 0.740), saved with joblib, and its hyperparameters, test metrics, and feature/permutation importances were written to a JSON record for reproducibility."
        ),

        heading("Results", HeadingLevel.HEADING_1),
        resultsTable,
        new Paragraph({ text: "", spacing: { after: 200 } }),

        new Paragraph({
          children: [new ImageRun({ data: cvImg, type: "png", transformation: { width: 380, height: 235 } })],
          alignment: AlignmentType.CENTER,
        }),

        new Paragraph({
          children: [
            new ImageRun({ data: cmImg, type: "png", transformation: { width: 230, height: 230 } }),
            new TextRun({ text: "   " }),
            new ImageRun({ data: fiImg, type: "png", transformation: { width: 300, height: 188 } }),
          ],
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
        }),

        heading("Challenges Faced & How I Solved Them", HeadingLevel.HEADING_1),
        bullet("Mixed-dtype categorical columns (int, bool, str) broke the ColumnTransformer's categorical imputer with a 'could not convert string to float' error. Solved by explicitly casting all categorical feature columns to string dtype before the Pipeline sees them, so pandas no longer hands the transformer a block it silently coerces to a single numeric dtype."),
        bullet("GradientBoostingClassifier has no class_weight parameter, unlike Logistic Regression and Random Forest. Since the imbalance (62/38) is only moderate, I relied on precision/recall/F1 rather than accuracy alone for Gradient Boosting instead of adding manual sample-weighting complexity, and documented that decision rather than silently ignoring it."),
        bullet("RandomizedSearchCV needed a genuine distribution, not a fixed list, to sample efficiently across a wide parameter space — used scipy.stats.randint/uniform instead of a coarse grid, which let 40 sampled configurations cover a much larger space than an equivalent GridSearchCV run would in reasonable time."),
        bullet("Avoiding test-set peeking during model selection required discipline: all model comparison during tuning used cross_val_score / GridSearchCV/RandomizedSearchCV's internal CV, and the test set was only evaluated once, after the final model was already chosen."),

        heading("Conclusion", HeadingLevel.HEADING_1),
        ...conclusionParagraphs,
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync("Week4_Report.docx", buffer);
  console.log("Wrote Week4_Report.docx");
});
