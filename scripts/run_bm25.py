r"""Run the BM25 baseline from a YAML config.

Usage:
    .venv\Scripts\python.exe scripts\run_bm25.py --config configs\bm25_mono_dev_eng.yaml

Two config styles:
- official release: `data_dir` (train_dev_sets or test_set), `track` (monolingual|crosslingual),
  `eval_split` (dev|test), `text_variant`. Monolingual = one BM25 per language over that
  language's pool, queried with that language's eval posts; crosslingual = one BM25 over the
  full pool. Gold = the official <track>_reference.json.
- sample data (no `track`): `data_dir` + posts_file/fact_checks_file/mapping_file; whole
  pool, every post that has a gold fact-check.

Optional `analyzer:` mapping (bool keys stem, stopwords, thai_tokenizer) switches on the
language-aware analyzer in src/retrieval/bm25.py; without it (or all false) the default
lowercase + \w+ tokenizer is used and results are unchanged. Monolingual runs tag original
text with the tasks.json language key; crosslingual runs use each text's detected language.

Writes results/<run_name>/metrics.json and predictions.json (official format, top 100), and
upserts a row in results/results.md (a re-run with the same run_name replaces its row).
In the table, S@10 and MRR@10 use the official averaging (monolingual: mean over languages;
crosslingual: micro over posts); "macro S@10" is the per-language mean (monolingual/sample).
"""

import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.data import (  # noqa: E402
    VARIANTS,
    eval_units,
    fc_text,
    gold_map,
    load_dataset,
    load_reference,
    fc_segments,
    load_split,
    post_language,
    post_segments,
    post_text,
)
from src.checks import assert_predictions_in_pool  # noqa: E402
from src.evaluate import evaluate, official_score, save_predictions  # noqa: E402
from src.retrieval.bm25 import Analyzer, BM25Retriever  # noqa: E402

SEED = 42
MODEL = "bm25"
TABLE_HEADER = "| run | model | variant | pool | split | track | S@10 | MRR@10 | macro S@10 |"
TABLE_SEP = "|---|---|---|---|---|---|---|---|---|"


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _migrate_row(line):
    """Old 7-column rows (no split/track) -> 9 columns: split 'sample', track '-'."""
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) == 7:
        cells = cells[:4] + ["sample", "-"] + cells[4:]
    return "| " + " | ".join(cells) + " |"


def update_results_table(path, run, variant, pool, split, track, s, mrr, macro, model=MODEL):
    macro_s = "-" if macro is None else f"{macro:.4f}"
    row = f"| {run} | {model} | {variant} | {pool} | {split} | {track} | {s:.4f} | {mrr:.4f} | {macro_s} |"
    lines = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f if l.strip()]
    body = [l for l in lines if l.startswith("|") and not l.startswith("| run |") and not l.startswith("|---")]
    lines = [TABLE_HEADER, TABLE_SEP] + [_migrate_row(l) for l in body]
    replaced = False
    for i, line in enumerate(lines):
        if line.startswith(f"| {run} |"):
            lines[i] = row
            replaced = True
    if not replaced:
        lines.append(row)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_sample(cfg, variant, top_n, analyzer=None):
    """Sample-data path: whole pool, all posts with gold."""
    posts, fcs, mapping = load_dataset(
        resolve(cfg["data_dir"]),
        posts_file=cfg.get("posts_file", "posts.csv"),
        fact_checks_file=cfg.get("fact_checks_file", "fact_checks.csv"),
        mapping_file=cfg.get("mapping_file", "fact_check_post_mapping.csv"),
    )
    gold = gold_map(mapping)
    t0 = time.time()
    fc_item = (lambda row: fc_segments(row, variant)) if analyzer else (lambda row: fc_text(row, variant))
    retriever = BM25Retriever(analyzer=analyzer).fit(list(fcs.index), [fc_item(row) for _, row in fcs.iterrows()])
    fit_s = time.time() - t0

    post_ids = sorted(gold)
    queries, post_langs = [], {}
    for pid in post_ids:
        if pid in posts.index:
            row = posts.loc[pid]
            queries.append(post_segments(row, variant) if analyzer else post_text(row, variant))
            post_langs[pid] = post_language(row)
        else:
            queries.append([] if analyzer else "")
            post_langs[pid] = "unk"
    t0 = time.time()
    preds = dict(zip(post_ids, retriever.search(queries, k=top_n)))
    timing = {"units": {"all": {"fit_s": fit_s, "search_s": time.time() - t0, "n_posts": len(post_ids), "pool": len(fcs)}}}
    info = {"pool": "all", "pool_size": len(fcs), "split": "sample", "track": "-"}
    return preds, gold, post_langs, info, timing


def run_official(cfg, variant, top_n, analyzer=None):
    split_dir, track, eval_split = resolve(cfg["data_dir"]), cfg["track"], cfg["eval_split"]
    posts, fcs, _, failures = load_split(split_dir)
    if failures:
        print(f"WARNING: {len(failures)} unparsable cells (set to '')")
    units = eval_units(split_dir, track, eval_split)
    gold = {p: set(v) for p, v in load_reference(split_dir, track).items()}

    # Only build texts for fact-checks that appear in some pool.
    needed = {f for u in units for f in u["pool_fc_ids"]}
    t0 = time.time()
    if analyzer is None:
        texts = {fid: fc_text(row, variant) for fid, row in fcs.loc[sorted(needed)].iterrows()}
    elif track == "crosslingual":  # mixed languages: tag each text with its detected language
        texts = {fid: fc_segments(row, variant) for fid, row in fcs.loc[sorted(needed)].iterrows()}
    else:  # monolingual: tag with the tasks.json language, so build per unit below
        texts = None
    print(f"built {len(needed)} fact-check texts in {time.time() - t0:.1f}s", flush=True)

    preds, post_langs, timing = {}, {}, {"units": {}}
    for u in units:
        lang, pool = u["lang"], u["pool_fc_ids"]
        t0 = time.time()
        if analyzer is not None and track == "monolingual":
            unit_texts = {f: fc_segments(fcs.loc[f], variant, lang) for f in pool}
            queries = [post_segments(posts.loc[p], variant, lang) for p in u["post_ids"]]
        elif analyzer is not None:
            unit_texts = texts
            queries = [post_segments(posts.loc[p], variant) for p in u["post_ids"]]
        else:
            unit_texts = texts
            queries = [post_text(posts.loc[p], variant) for p in u["post_ids"]]
        retriever = BM25Retriever(analyzer=analyzer).fit(pool, [unit_texts[f] for f in pool])
        fit_s = time.time() - t0
        t0 = time.time()
        ranked = retriever.search(queries, k=top_n)
        search_s = time.time() - t0
        for p, r in zip(u["post_ids"], ranked):
            preds[p] = r
            post_langs[p] = lang
        timing["units"][lang] = {"fit_s": fit_s, "search_s": search_s, "n_posts": len(queries), "pool": len(pool)}
        print(f"  {lang}: pool={len(pool)} posts={len(queries)} fit={fit_s:.1f}s search={search_s:.1f}s", flush=True)

    if track == "monolingual":
        pool_label, pool_size = "per-language", {u["lang"]: len(u["pool_fc_ids"]) for u in units}
    else:
        pool_label, pool_size = "all", len(units[0]["pool_fc_ids"])
    info = {"pool": pool_label, "pool_size": pool_size, "split": eval_split, "track": track}
    return preds, gold, post_langs, info, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    with open(resolve(args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_name = cfg["run_name"]
    variant = cfg.get("text_variant", "eng")
    if variant not in VARIANTS:
        raise ValueError(f"text_variant must be one of {VARIANTS}, got {variant!r}")
    k = int(cfg.get("k", 10))
    top_n = int(cfg.get("top_n", 100))

    analyzer = Analyzer.from_config(cfg.get("analyzer"))
    t_start = time.time()
    official = "track" in cfg
    preds, gold, post_langs, info, timing = (run_official if official else run_sample)(cfg, variant, top_n, analyzer)
    timing["total_s"] = time.time() - t_start

    metrics = evaluate(preds, gold, post_langs, k=k)
    track = info["track"]
    if official:
        headline = official_score(metrics, track)
        mrr = metrics["macro_mrr"] if track == "monolingual" else metrics["mrr"]
        macro = metrics["macro_avg_s_at_k"] if track == "monolingual" else None
    else:
        headline, mrr, macro = metrics["s_at_k"], metrics["mrr"], metrics["macro_avg_s_at_k"]

    out_dir = resolve(os.path.join("results", run_name))
    os.makedirs(out_dir, exist_ok=True)
    record = {
        "run_name": run_name,
        "model": MODEL,
        "text_variant": variant,
        **info,
        "seed": SEED,
        "headline_s_at_10": headline,
        "headline_mrr_at_10": mrr,
        "headline_note": "monolingual: mean over languages; crosslingual: micro over posts" if official else "micro over posts",
        "metrics": metrics,
        "timing": timing,
        "config": cfg,
    }
    if official:  # every predicted id must come from the post's pool before anything is written
        assert_predictions_in_pool(preds, eval_units(resolve(cfg["data_dir"]), track, cfg["eval_split"]))
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    save_predictions(preds, os.path.join(out_dir, "predictions.json"), k=100)

    update_results_table(
        resolve(os.path.join("results", "results.md")),
        run_name, variant, info["pool"], info["split"], track, headline, mrr, macro,
    )

    print(f"run:        {run_name}  (model={MODEL}, variant={variant}, split={info['split']}, track={track}, pool={info['pool']})")
    print(f"posts:      {metrics['n_posts']}")
    print(f"S@{k}:       {headline:.4f}  ({'macro over languages' if track == 'monolingual' else 'micro'})")
    print(f"MRR@{k}:     {mrr:.4f}")
    print("per language:")
    for lang, v in metrics["per_language"].items():
        print(f"  {lang:>5}  S@{k}={v['s_at_k']:.4f}  MRR={v['mrr']:.4f}  n={v['n']}")
    print(f"time:       {timing['total_s']:.1f}s")
    print(f"wrote {os.path.relpath(out_dir, ROOT)}/metrics.json and predictions.json")


if __name__ == "__main__":
    main()
