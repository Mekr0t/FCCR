r"""Build results/report_tables.md for the stage-1 report.

Built only from results/<run>/metrics.json (our runs) and the organizers' baseline predictions
in data/raw/semeval2025_task7_data/baselines/*.json, which are read as data and scored with
src/evaluate.py against the official test reference. Overwrites results/report_tables.md.

Usage (from the project root):
    .venv\Scripts\python.exe scripts\make_report_tables.py

Bold convention: the best value per column is bold within each track group of Table A and
within each language column of Table B (ties are all bold; values compared after rounding to
the 3 decimals shown). Missing runs are shown as "–" and are not eligible for bold.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data import eval_units, load_reference  # noqa: E402
from src.evaluate import evaluate  # noqa: E402

RAW_DIR = os.path.join(ROOT, "data", "raw", "semeval2025_task7_data")
TEST_DIR = os.path.join(RAW_DIR, "test_set")
BASELINE_DIR = os.path.join(RAW_DIR, "baselines")
RESULTS_DIR = os.path.join(ROOT, "results")

# (label, baseline file stem) in the order shown in the tables.
ORGANIZERS = [
    ("BM25 (organizers)", "bm25"),
    ("mE5-large (organizers)", "multilingual-e5-large"),
    ("GTR-T5-large (organizers)", "gtr-t5-large"),
    ("mpnet-multi-v2 (organizers)", "paraphrase-multilingual-mpnet-base-v2"),
]
# Organizers' published test S@10 (monolingual = mean over languages; crosslingual = micro).
PUBLISHED_S10 = {
    "monolingual": {"bm25": 0.8123, "multilingual-e5-large": 0.8589, "gtr-t5-large": 0.7299, "paraphrase-multilingual-mpnet-base-v2": 0.5683},
    "crosslingual": {"bm25": 0.60275, "multilingual-e5-large": 0.63725, "gtr-t5-large": 0.70525, "paraphrase-multilingual-mpnet-base-v2": 0.41075},
}
# Our runs (test split) per track: (label, run name).
OURS_TEST = {
    "monolingual": [
        ("BM25 (ours, eng)", "bm25_mono_test_eng"),
        ("BM25 tuned (ours, both + stem + stop + Thai)", "bm25_mono_test_both_stem_stop_thai"),
        ("mE5-large (ours, both)", "dense_mono_test_both"),
    ],
    "crosslingual": [
        ("BM25 (ours, eng)", "bm25_cross_test_eng"),
        ("BM25 tuned (ours, eng + stem + stop)", "bm25_cross_test_eng_stem_stop"),
        ("mE5-large (ours, eng)", "dense_cross_test_eng"),
    ],
}
TRACK_TITLE = {
    "monolingual": "Monolingual (test, S@10 = mean over 10 languages)",
    "crosslingual": "Crosslingual (test, S@10 micro over posts)",
}


# ---- scoring ----------------------------------------------------------------------------
def score_organizers(test_dir=TEST_DIR, baseline_dir=BASELINE_DIR):
    """{track: {label: {s10, mrr, per_language, published_s10}}} from the baseline JSON files."""
    out = {}
    for track in ("monolingual", "crosslingual"):
        units = eval_units(test_dir, track, "test")
        post_langs = {str(p): u["lang"] for u in units for p in u["post_ids"]} if track == "monolingual" else None
        gold = {p: set(v) for p, v in load_reference(test_dir, track).items()}
        out[track] = {}
        for label, stem in ORGANIZERS:
            with open(os.path.join(baseline_dir, f"{stem}_{track}.json"), encoding="utf-8") as f:
                pred = json.load(f)
            m = evaluate(pred, gold, post_langs, k=10)
            if track == "monolingual":
                s10, mrr = m["macro_avg_s_at_k"], m["macro_mrr"]
                per_lang = {l: v["s_at_k"] for l, v in m["per_language"].items()}
            else:
                s10, mrr, per_lang = m["s_at_k"], m["mrr"], {}
            out[track][label] = {
                "s10": s10,
                "mrr": mrr,
                "per_language": per_lang,
                "published_s10": PUBLISHED_S10[track][stem],
            }
    return out


def load_run(results_dir, name):
    """Parsed metrics.json of one run, or None if the run folder has no metrics file."""
    path = os.path.join(results_dir, name, "metrics.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_scores(m):
    inner = m.get("metrics", {})
    s10 = m.get("headline_s_at_10", inner.get("s_at_k"))
    mrr = m.get("headline_mrr_at_10", inner.get("mrr"))
    per_lang = {l: v["s_at_k"] for l, v in inner.get("per_language", {}).items()}
    return {"s10": s10, "mrr": mrr, "per_language": per_lang}


# ---- bold / formatting -------------------------------------------------------------------
def bold_best(values, higher_is_better=True):
    """Bool per value: True for every value equal (after rounding to 3 decimals) to the best one.

    None values are never bold. If all values are None, nothing is bold.
    """
    rounded = [None if v is None else round(v, 3) for v in values]
    present = [v for v in rounded if v is not None]
    if not present:
        return [False] * len(values)
    best = max(present) if higher_is_better else min(present)
    return [v is not None and v == best for v in rounded]


def _fmt(v, bold=False):
    if v is None:
        return "–"
    s = f"{v:.3f}"
    return f"**{s}**" if bold else s


def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


# ---- sections -----------------------------------------------------------------------------
def table_a(results_dir, organizers):
    lines = ["## Table A - main test results", ""]
    for track in ("monolingual", "crosslingual"):
        entries = []
        for label, scores in organizers.get(track, {}).items():
            entries.append((label, scores["s10"], scores["mrr"], f"baselines/{track}"))
        for label, name in OURS_TEST[track]:
            m = load_run(results_dir, name)
            sc = _run_scores(m) if m else {"s10": None, "mrr": None}
            entries.append((label, sc["s10"], sc["mrr"], name if m else f"{name} (missing)"))
        bold_s = bold_best([e[1] for e in entries])
        bold_m = bold_best([e[2] for e in entries])
        rows = [
            [label, _fmt(s, bs), _fmt(mr, bm), src]
            for (label, s, mr, src), bs, bm in zip(entries, bold_s, bold_m)
        ]
        lines += [f"### {TRACK_TITLE[track]}", ""]
        lines += _table(["System", "S@10", "MRR@10", "Source"], rows)
        lines.append("")
    lines += [
        "Organizer rows are computed by `src/evaluate.py` from the baseline files. Their published S@10 values",
        "(for comparison) are: " + "; ".join(
            f"{track}: " + ", ".join(f"{l.split(' (')[0]} {PUBLISHED_S10[track][stem]}" for l, stem in ORGANIZERS)
            for track in ("monolingual", "crosslingual")
        ) + ".",
        "",
    ]
    diffs = [
        abs(scores["s10"] - scores["published_s10"])
        for track in organizers
        for scores in organizers[track].values()
    ]
    if diffs:
        lines.append(f"Max |computed - published| S@10 over organizer rows: {max(diffs):.5f}.")
        lines.append("")
    return lines


def table_b(results_dir, organizers):
    m_org = organizers.get("monolingual", {})
    rows_src = []
    for label, _ in ORGANIZERS[:2]:  # BM25 and mE5 only
        if label in m_org:
            rows_src.append((label, m_org[label]["s10"], m_org[label]["per_language"]))
    for label, name in OURS_TEST["monolingual"][1:]:  # tuned BM25 and mE5 (ours)
        m = load_run(results_dir, name)
        if m:
            sc = _run_scores(m)
            rows_src.append((label, sc["s10"], sc["per_language"]))
        else:
            rows_src.append((label + " (missing)", None, {}))

    langs = sorted({l for _, _, pl in rows_src for l in pl})
    lines = ["## Table B - per-language test S@10 (monolingual)", ""]
    if not rows_src:
        return lines + ["(no rows)", ""]
    cols = {l: bold_best([pl.get(l) for _, _, pl in rows_src]) for l in langs}
    cols["avg"] = bold_best([s for _, s, _ in rows_src])
    rows = []
    for i, (label, s, pl) in enumerate(rows_src):
        cells = [label] + [_fmt(pl.get(l), cols[l][i]) for l in langs] + [_fmt(s, cols["avg"][i])]
        rows.append(cells)
    lines += _table(["System"] + langs + ["avg"], rows)
    lines.append("")
    return lines


def _dev_runs(results_dir):
    runs = []
    if not os.path.isdir(results_dir):
        return runs
    for name in sorted(os.listdir(results_dir)):
        m = load_run(results_dir, name)
        if m and m.get("split") == "dev":
            runs.append((name, m))
    runs.sort(key=lambda x: (x[1].get("track", ""), x[0]))
    return runs


def table_c(results_dir):
    runs = _dev_runs(results_dir)
    lines = ["## Table C - dev ablations (all dev runs)", ""]
    if not runs:
        return lines + ["(no dev runs)", ""]
    mono_runs = [m for _, m in runs if m.get("track") == "monolingual"]
    langs = sorted({l for m in mono_runs for l in _run_scores(m)["per_language"]})
    headers = ["Run", "Model", "Variant", "Pool", "Track", "S@10", "MRR@10"] + langs
    rows = []
    for name, m in runs:
        sc = _run_scores(m)
        pl = sc["per_language"] if m.get("track") == "monolingual" else {}
        rows.append(
            [name, str(m.get("model", "–")), str(m.get("text_variant", "–")), str(m.get("pool", "–")), str(m.get("track", "–")),
             _fmt(sc["s10"]), _fmt(sc["mrr"])]
            + [_fmt(pl[l]) if l in pl else "–" for l in langs]
        )
    lines += _table(headers, rows)
    lines += ["", "Per-language columns are S@10 for monolingual runs; crosslingual runs have no per-language breakdown.", ""]
    return lines


def build_report(results_dir=RESULTS_DIR, organizers=None):
    """Return the full markdown text. `organizers` defaults to scoring the baseline files."""
    if organizers is None:
        organizers = score_organizers()
    lines = [
        "# Report tables",
        "",
        "Generated by `scripts/make_report_tables.py` from `results/*/metrics.json` and the organizers' baseline predictions. "
        "Do not edit by hand; rerun the script.",
        "",
    ]
    lines += table_a(results_dir, organizers)
    lines += table_b(results_dir, organizers)
    lines += table_c(results_dir)
    return "\n".join(lines).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    args = ap.parse_args()
    text = build_report(args.results_dir)
    out = os.path.join(args.results_dir, "report_tables.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
