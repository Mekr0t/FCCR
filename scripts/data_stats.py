r"""Dataset statistics for the stage-1 report.

Writes results/data_stats.json and results/data_stats.md for both official splits
(train_dev_sets, test_set). Output contains ONLY aggregates (counts, percentages, token
lengths, language codes). No post or fact-check text is ever printed or written.

Usage (from the project root):
    .venv\Scripts\python.exe scripts\data_stats.py

Conventions:
- Eval posts = posts in the split's eval units (dev for train_dev_sets, test for test_set).
- Post language: monolingual = the tasks.json key. Crosslingual = detected top language of
  the post text (src.data.post_language). FC language = detected top language of the claim.
- Lengths = whitespace tokens of src.data.post_text / fc_text, variants orig and eng.
  Post lengths cover the eval posts of each track; FC lengths cover every row of fact_checks.csv.
- Post composition: "text" = the translated text tuple has a non-empty original or English part.
  "OCR" = at least one non-empty OCR tuple.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data import (  # noqa: E402
    _select,
    _top_lang,
    eval_units,
    fc_text,
    load_reference,
    load_split,
    post_language,
    post_text,
    split_kind,
)

RAW_DIR = os.path.join(ROOT, "data", "raw", "semeval2025_task7_data")
SPLITS = {"train_dev_sets": os.path.join(RAW_DIR, "train_dev_sets"), "test_set": os.path.join(RAW_DIR, "test_set")}
RESULTS_DIR = os.path.join(ROOT, "results")
TOP_COMBOS = 15
VARIANTS = ("orig", "eng")


def _pct(n, d):
    return 100.0 * n / d if d else 0.0


def _len_summary(texts):
    lens = np.array([len(t.split()) for t in texts], dtype=float)
    if lens.size == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    return {
        "n": int(lens.size),
        "mean": round(float(lens.mean()), 2),
        "median": round(float(np.median(lens)), 2),
        "p90": round(float(np.percentile(lens, 90)), 2),
    }


def _n_train(section_tasks):
    return len(section_tasks["posts_train"]) if "posts_train" in section_tasks else None


def split_stats(split_dir):
    kind = split_kind(split_dir)
    posts_df, fcs_df, tasks, failures = load_split(split_dir)
    ref_mono = load_reference(split_dir, "monolingual")
    ref_cross = load_reference(split_dir, "crosslingual")
    mono_units = eval_units(split_dir, "monolingual", kind)
    cross_unit = eval_units(split_dir, "crosslingual", kind)[0]

    # ---- monolingual, per language --------------------------------------------------
    mono = {}
    mono_lang_of_post = {}
    for u in sorted(mono_units, key=lambda x: x["lang"]):
        lang = u["lang"]
        counts = [len(set(ref_mono[str(p)])) for p in u["post_ids"]]
        for p in u["post_ids"]:
            mono_lang_of_post[p] = lang
        mono[lang] = {
            "n_train_posts": _n_train(tasks["monolingual"][lang]),
            "n_eval_posts": len(u["post_ids"]),
            "pool_size": len(u["pool_fc_ids"]),
            "gold_pairs": int(sum(counts)),
            "n_multi_gold_posts": int(sum(c > 1 for c in counts)),
            "pct_multi_gold": _pct(sum(c > 1 for c in counts), len(counts)),
        }
    mono_eval_total = sum(v["n_eval_posts"] for v in mono.values())
    mono_totals = {
        "n_train_posts": sum(v["n_train_posts"] or 0 for v in mono.values()) if any(v["n_train_posts"] is not None for v in mono.values()) else None,
        "n_eval_posts": mono_eval_total,
        "pool_size": sum(v["pool_size"] for v in mono.values()),
        "gold_pairs": sum(v["gold_pairs"] for v in mono.values()),
        "n_multi_gold_posts": sum(v["n_multi_gold_posts"] for v in mono.values()),
        "pct_multi_gold": _pct(sum(v["n_multi_gold_posts"] for v in mono.values()), mono_eval_total),
        "pct_eng_posts": _pct(mono.get("eng", {}).get("n_eval_posts", 0), mono_eval_total),
    }

    # ---- crosslingual ----------------------------------------------------------------
    cpost_ids = cross_unit["post_ids"]
    post_lang = {p: post_language(posts_df.loc[p]) for p in cpost_ids}
    fc_lang_cache = {}
    missing_gold_fc = set()

    def fc_lang(fc):
        if fc not in fc_lang_cache:
            if fc not in fcs_df.index:
                missing_gold_fc.add(fc)
                fc_lang_cache[fc] = "missing"
            else:
                fc_lang_cache[fc] = _top_lang(fcs_df.loc[fc, "claim"]) or "unk"
        return fc_lang_cache[fc]

    pairs = []
    n_multi_cross = 0
    for p in cpost_ids:
        gold = set(ref_cross[str(p)])
        if len(gold) > 1:
            n_multi_cross += 1
        for fc in gold:
            pairs.append((post_lang[p], fc_lang(fc)))
    n_cross = len(cpost_ids)
    post_counts = Counter(post_lang.values())
    fc_counts = Counter(fl for _, fl in pairs)
    combo_counts = Counter(pairs).most_common(TOP_COMBOS)
    cross = {
        "n_train_posts": _n_train(tasks["crosslingual"]),
        "n_eval_posts": n_cross,
        "pool_size": len(cross_unit["pool_fc_ids"]),
        "gold_pairs": len(pairs),
        "n_multi_gold_posts": n_multi_cross,
        "pct_multi_gold": _pct(n_multi_cross, n_cross),
        "n_gold_fc_missing_from_fact_checks": len(missing_gold_fc),
        "post_language_counts": {k: {"posts": v, "pct": _pct(v, n_cross)} for k, v in post_counts.most_common()},
        "gold_fc_language_counts": {k: {"pairs": v, "pct": _pct(v, len(pairs))} for k, v in fc_counts.most_common()},
        "top_pair_languages": [
            {"post_lang": a, "fc_lang": b, "pairs": v, "pct": _pct(v, len(pairs))} for (a, b), v in combo_counts
        ],
        "pct_eng_posts_detected": _pct(post_counts.get("eng", 0), n_cross),
    }

    # ---- post composition over all eval posts ----------------------------------------
    all_ids = sorted(set(mono_lang_of_post) | set(cpost_ids))
    rows = posts_df.loc[all_ids, ["text", "ocr"]].to_dict("records")
    n_text = n_ocr = n_both = n_neither = 0
    ocr_counts = []
    for r in rows:
        has_text = bool(_select(r["text"], "eng"))
        k = sum(1 for o in (r["ocr"] or []) if _select(o, "eng"))
        ocr_counts.append(k)
        has_ocr = k > 0
        n_text += has_text
        n_ocr += has_ocr
        n_both += has_text and has_ocr
        n_neither += (not has_text) and (not has_ocr)
    n_all = len(all_ids)
    composition = {
        "n_posts": n_all,
        "pct_text_only": _pct(n_text - n_both, n_all),
        "pct_ocr_only": _pct(n_ocr - n_both, n_all),
        "pct_both": _pct(n_both, n_all),
        "pct_neither": _pct(n_neither, n_all),
        "ocr_items_mean": round(float(np.mean(ocr_counts)), 3) if ocr_counts else None,
        "ocr_items_median": float(np.median(ocr_counts)) if ocr_counts else None,
        "pct_posts_with_any_ocr": _pct(n_ocr, n_all),
    }

    # ---- lengths ---------------------------------------------------------------------
    mono_ids = sorted(mono_lang_of_post)
    post_recs = posts_df[["text", "ocr"]].to_dict("records")  # index order; looked up below
    post_index = posts_df.index
    pos = {pid: i for i, pid in enumerate(post_index)}
    mono_recs = [post_recs[pos[p]] for p in mono_ids]
    cross_recs = [post_recs[pos[p]] for p in cpost_ids]
    fc_recs = fcs_df[["claim", "title"]].to_dict("records")
    lengths = {"posts_monolingual": {}, "posts_crosslingual": {}, "fact_checks": {}}
    for v in VARIANTS:
        lengths["posts_monolingual"][v] = _len_summary(post_text(r, v) for r in mono_recs)
        lengths["posts_crosslingual"][v] = _len_summary(post_text(r, v) for r in cross_recs)
        lengths["fact_checks"][v] = _len_summary(fc_text(r, v) for r in fc_recs)

    return {
        "eval_split": kind,
        "n_parse_failures": len(failures),
        "monolingual": {"per_language": dict(sorted(mono.items())), "totals": mono_totals},
        "crosslingual": cross,
        "post_composition": composition,
        "lengths": lengths,
    }


# ---- markdown ----------------------------------------------------------------------------
def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def _f(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}"


def _split_md(name, s):
    kind = s["eval_split"]
    ev = "dev" if kind == "dev" else "test"
    L = [f"## {name} (eval split: {ev})", ""]
    L.append(f"Parse failures in posts/fact_checks CSVs: {s['n_parse_failures']}")
    L.append("")

    L.append("### Monolingual, per language")
    L.append("")
    mono = s["monolingual"]
    rows = []
    for lang, v in mono["per_language"].items():
        rows.append([lang, _f(v["n_train_posts"], 0), v["n_eval_posts"], v["pool_size"], v["gold_pairs"], _f(v["pct_multi_gold"])])
    t = mono["totals"]
    rows.append(["**total**", _f(t["n_train_posts"], 0), t["n_eval_posts"], t["pool_size"], t["gold_pairs"], _f(t["pct_multi_gold"])])
    L += _table(["Language", "#train posts", f"#{ev} posts", "Pool size", "Gold pairs", "% posts with >1 gold"], rows)
    L.append("")
    L.append(f"English share of monolingual eval posts (tasks.json key): {_f(t['pct_eng_posts'])}%")
    L.append("")

    c = s["crosslingual"]
    L.append("### Crosslingual")
    L.append("")
    L += _table(
        ["Quantity", "Value"],
        [
            ["#train posts", _f(c["n_train_posts"], 0)],
            [f"#{ev} posts", c["n_eval_posts"]],
            ["Pool size (fact-checks)", c["pool_size"]],
            ["Gold pairs", c["gold_pairs"]],
            ["% posts with >1 gold", _f(c["pct_multi_gold"])],
            ["Gold FC ids missing from fact_checks.csv", c["n_gold_fc_missing_from_fact_checks"]],
            ["% eval posts detected as English", _f(c["pct_eng_posts_detected"])],
        ],
    )
    L.append("")
    L.append("Post language distribution (detected):")
    L.append("")
    L += _table(
        ["Post language", "Posts", "%"],
        [[k, v["posts"], _f(v["pct"])] for k, v in c["post_language_counts"].items()],
    )
    L.append("")
    L.append("Gold fact-check language distribution (detected, per gold pair):")
    L.append("")
    L += _table(
        ["FC language", "Gold pairs", "%"],
        [[k, v["pairs"], _f(v["pct"])] for k, v in c["gold_fc_language_counts"].items()],
    )
    L.append("")
    L.append(f"Top {TOP_COMBOS} (post language -> fact-check language) combinations by gold pairs:")
    L.append("")
    L += _table(
        ["Post lang", "FC lang", "Gold pairs", "%"],
        [[r["post_lang"], r["fc_lang"], r["pairs"], _f(r["pct"])] for r in c["top_pair_languages"]],
    )
    L.append("")

    p = s["post_composition"]
    L.append("### Post composition (all eval posts of the split, mono + cross)")
    L.append("")
    L += _table(
        ["Quantity", "Value"],
        [
            ["Eval posts", p["n_posts"]],
            ["% text only", _f(p["pct_text_only"])],
            ["% OCR only", _f(p["pct_ocr_only"])],
            ["% both", _f(p["pct_both"])],
            ["% neither", _f(p["pct_neither"])],
            ["% with any OCR item", _f(p["pct_posts_with_any_ocr"])],
            ["OCR items per post: mean", _f(p["ocr_items_mean"], 2)],
            ["OCR items per post: median", _f(p["ocr_items_median"], 1)],
        ],
    )
    L.append("")

    L.append("### Lengths (whitespace tokens)")
    L.append("")
    lr = []
    for label, key in (
        ("Posts, monolingual track (eval)", "posts_monolingual"),
        ("Posts, crosslingual track (eval)", "posts_crosslingual"),
        ("Fact-checks (all rows of fact_checks.csv)", "fact_checks"),
    ):
        for v in VARIANTS:
            d = s["lengths"][key][v]
            lr.append([label, v, d["n"], _f(d["mean"]), _f(d["median"]), _f(d["p90"])])
    L += _table(["Units", "Variant", "n", "Mean", "Median", "p90"], lr)
    L.append("")
    return L


def render_md(stats, runtime_s):
    L = [
        "# Dataset statistics (aggregates only)",
        "",
        "Generated by `scripts/data_stats.py`. Counts, percentages and token lengths only; no text.",
        "Language codes are detected with `src.data._top_lang` (crosslingual) or taken from `tasks.json` (monolingual).",
        "",
    ]
    for name, s in stats["splits"].items():
        L += _split_md(name, s)
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    args = ap.parse_args()

    t0 = time.time()
    stats = {"splits": {}}
    for name, d in SPLITS.items():
        stats["splits"][name] = split_stats(d)
        print(f"{name}: done")
    runtime = time.time() - t0

    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "data_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.results_dir, "data_stats.md"), "w", encoding="utf-8") as f:
        f.write(render_md(stats, runtime))
    print(f"wrote {args.results_dir}/data_stats.json and data_stats.md in {runtime:.1f} s")


if __name__ == "__main__":
    main()
