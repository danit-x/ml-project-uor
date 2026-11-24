"""Check that every number in the progress document and slides traces to a source file.

    python docs/progress_evaluation/build/verify_outputs.py

Reads the generated .docx and .pptx (slide text, tables, chart caches and speaker notes) with the
standard library only, extracts every numeric token and looks it up in numbers.json (which
collect_numbers.py builds from reports/) and in the official CSVs. Tokens that are not data
(section numbers, dates, course codes, documented protocol facts) are listed in KNOWN with the
reason and the source. Prints any token it cannot trace and exits 1 if there is one.
Not part of the ML pipeline.
"""
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DOCX = HERE.parent / "URLSentra_progress_document.docx"
PPTX = HERE.parent / "URLSentra_progress_presentation.pptx"
TOKEN = re.compile(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?")
HASH = re.compile(r"\b[0-9a-f]{7}\b")
TIMING = re.compile(r"\[(\d+):(\d{2})\]")   # speaker-note timing markers, summed separately

# Numbers that are not results, each with where it is documented.
KNOWN = {
    "2026": "date", "24": "date (24 November 2026)", "42": "group number / config.RANDOM_STATE",
    "5203": "course code EC5203", "967": "UCI dataset id", "1": "label value / list numbering / min leaf",
    "0": "label value / axis", "2": "list numbering", "3": "list numbering", "4": "list numbering", "5": "list numbering",
    "6": "section number", "7": "section number", "8": "section number", "10": "slide number", "11": "slide number",
    "12": "slide number", "100": "URLSimilarityIndex value for every legitimate URL (README §14)",
    "0.99": "rounded summary of the ~0.99 scores (official_comparison.csv)",
    "0.999": "demo risk score (reports/demo_full_2026-11-24.txt)",
    "2,000": "SVM tuning sample size (src/models/svm.py comment)",
    "20,000": "calibration development sample (notebook 04 SAMPLE = 20000; re-run reproduced)",
    "64": "unit tests (python -m unittest discover -s tests -t .)",
    "0.1": "L1 C value that did not converge (reports/tuning/lr_full_run_INCOMPLETE_2026-11-24.log)",
    "33": "minutes of the non-converged L1 fold (src/models/logreg.py comment)",
    "45": "group size when the query was dropped (src/data.py docstring)",
    "300": "LR C = 300 candidate / RF trees (reports/tuning/*.csv)",
    "90": "seconds demo.py takes to start (presentation_outline.md logistics)",
    "500": "SHAP rows (main_full_run log: 'SHAP for 500 test URLs')",
    "50": "chart axis step", "150": "chart axis tick", "200": "chart axis tick",
    "0.001": "chart axis step / LR C lower bound", "0.002": "chart axis tick", "0.003": "chart axis tick",
    "0.004": "chart axis tick", "0.005": "chart axis bound", "-0.001": "chart axis tick", "-0.002": "chart axis tick",
    "-0.003": "chart axis tick", "-0.004": "chart axis tick", "-0.005": "chart axis bound",
    "+0.001": "chart axis tick", "+0.002": "chart axis tick", "+0.003": "chart axis tick", "+0.004": "chart axis tick",
    "0.30": "risk.LOW_MAX", "0.70": "risk.HIGH_MIN",
}


def flatten(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from flatten(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from flatten(v)
    else:
        yield str(obj)


def norm(tok):
    return tok.lstrip("+").rstrip(",")


def source_tokens():
    numbers = json.loads((HERE / "numbers.json").read_text(encoding="utf-8"))
    toks = set()
    for value in flatten(numbers):
        toks.update(norm(t) for t in TOKEN.findall(value))
    # every value of the official CSVs, in the formats the documents use
    for f in ["official_comparison.csv", "format_baseline.csv", "format_controlled.csv", "cv_results.csv",
              "results.csv", "risk_bands.csv"]:
        frame = pd.read_csv(ROOT / "reports" / f)
        for v in frame.select_dtypes("number").to_numpy().ravel():
            if pd.isna(v):
                continue
            toks.update({f"{v:.4f}", f"{v:.5f}", f"{v:.3f}", f"{v:.2f}", f"{v:,.0f}" if float(v).is_integer() else f"{v:.4f}"})
    return toks


def texts(path, kind):
    out = []
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith(".xml")]
        if kind == "docx":
            names = [n for n in names if n == "word/document.xml"]
        else:
            names = [n for n in names if re.match(r"ppt/(slides/slide|notesSlides/notesSlide|charts/chart)\d+\.xml", n)]
        for n in names:
            xml = z.read(n).decode("utf-8")
            parts = re.findall(r"<(?:a|w):t(?: [^>]*)?>([^<]*)</(?:a|w):t>", xml) + re.findall(r"<c:v>([^<]*)</c:v>", xml)
            out.append((n, " ".join(parts)))
    return out


def main():
    known_src = source_tokens()
    unresolved, counted, seconds, slides_timed = [], 0, 0, 0
    for path, kind in [(DOCX, "docx"), (PPTX, "pptx")]:
        for part, text in texts(path, kind):
            for mm, ss in TIMING.findall(text):
                seconds += 60 * int(mm) + int(ss)
                slides_timed += 1
            text = TIMING.sub(" ", HASH.sub(" ", text)).replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
            for m in TOKEN.finditer(text):
                tok = norm(m.group())
                counted += 1
                if tok in known_src or tok in KNOWN or m.group() in KNOWN:
                    continue
                # chart caches store raw floats such as 0.0025 or 102
                try:
                    if f"{float(tok.replace(',', '')):.4f}" in known_src or f"{float(tok.replace(',', '')):,.0f}" in known_src:
                        continue
                except ValueError:
                    pass
                unresolved.append((path.name, part, m.group(), text[max(0, m.start() - 50): m.end() + 30]))
    print(f"numeric tokens checked: {counted}")
    print(f"planned delivery from speaker notes: {slides_timed} timed slides, {seconds // 60}:{seconds % 60:02d} total")
    if seconds != 15 * 60:
        unresolved.append(("timing", "notes", f"{seconds} s", "planned delivery is not 15:00"))
    for name, part, tok, ctx in unresolved:
        print(f"UNTRACED {tok!r} in {name}:{part}: ...{ctx}...")
    print("all numbers traced to numbers.json, the official CSVs or a documented constant" if not unresolved
          else f"{len(unresolved)} untraced tokens")
    sys.exit(1 if unresolved else 0)


if __name__ == "__main__":
    main()
