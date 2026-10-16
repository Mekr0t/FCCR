r"""Run the dense (bi-encoder) baseline from a YAML config.

Usage:
    .venv\Scripts\python.exe scripts\run_dense.py --config configs\dense_mono_dev_eng.yaml

Config mirrors run_bm25.py (official release: data_dir/track/eval_split; sample: data_dir +
posts_file/fact_checks_file/mapping_file) plus: `model` (HF id), `model_label` (results.md
column, e.g. mE5-large), `text_variant` (eng|orig|both), `max_seq_length` (512), `batch_size`
(32), `query_prefix`/`passage_prefix` (e5: "query: " / "passage: ").

Fact-check embeddings of every pool fact-check are cached in
cache/<model-slug>/<split dir name>/<variant>/ and post embeddings in .../<variant>_posts/
(see src/retrieval/dense.py); a re-run with a warm cache only does the matmul search.
Monolingual: the split's fact-checks are embedded once, each language searches its own pool
subset. Outputs are the same as run_bm25.py (metrics.json, predictions.json, results.md row).
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from run_bm25 import resolve, update_results_table  # noqa: E402
from src.data import (  # noqa: E402
    VARIANTS,
    eval_units,
    fc_text,
    gold_map,
    load_dataset,
    load_reference,
    load_split,
    post_language,
    post_text,
)
from src.checks import assert_predictions_in_pool  # noqa: E402
from src.evaluate import evaluate, official_score, save_predictions  # noqa: E402
from src.retrieval.dense import (  # noqa: E402
    DEFAULT_MODEL,
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    DenseEncoder,
    EmbeddingStore,
    model_slug,
    search,
    seed_everything,
    truncation_stats,
)

SEED = 42


def embed_all(cfg, variant, split_name, post_ids, post_texts, fc_ids, fc_texts):
    """Embed (or load from cache) posts and fact-checks. Returns (q_emb, fc_emb, info)."""
    model = cfg.get("model", DEFAULT_MODEL)
    max_len = int(cfg.get("max_seq_length", 512))
    qp, pp = cfg.get("query_prefix", QUERY_PREFIX), cfg.get("passage_prefix", PASSAGE_PREFIX)
    encoder = DenseEncoder(model, max_len, int(cfg.get("batch_size", 32)))
    base = os.path.join(ROOT, "cache", model_slug(model), split_name)

    fc_store = EmbeddingStore(os.path.join(base, variant), model, max_len, pp)
    fc_emb, fc_info = fc_store.get(encoder, fc_ids, [pp + t for t in fc_texts], variant)
    post_store = EmbeddingStore(os.path.join(base, f"{variant}_posts"), model, max_len, qp)
    q_emb, post_info = post_store.get(encoder, post_ids, [qp + t for t in post_texts], variant)

    info = {}
    for name, i in (("fact_checks", fc_info), ("posts", post_info)):
        ntok = i.pop("ntok")
        info[name] = {**i, "tokens": truncation_stats(ntok, max_len)}
    info["device"] = encoder.device
    return q_emb, fc_emb, info


def run_sample(cfg, variant, top_n):
    posts, fcs, mapping = load_dataset(
        resolve(cfg["data_dir"]),
        posts_file=cfg.get("posts_file", "posts.csv"),
        fact_checks_file=cfg.get("fact_checks_file", "fact_checks.csv"),
        mapping_file=cfg.get("mapping_file", "fact_check_post_mapping.csv"),
    )
    gold = gold_map(mapping)
    post_ids = [p for p in sorted(gold) if p in posts.index]
    post_langs = {p: post_language(posts.loc[p]) for p in post_ids}
    missing = [p for p in sorted(gold) if p not in posts.index]  # gold posts without a row: empty prediction
    fc_ids = list(fcs.index)

    t0 = time.time()
    q_emb, fc_emb, einfo = embed_all(
        cfg, variant, os.path.basename(os.path.normpath(cfg["data_dir"])),
        post_ids, [post_text(posts.loc[p], variant) for p in post_ids],
        fc_ids, [fc_text(row, variant) for _, row in fcs.iterrows()],
    )
    embed_s = time.time() - t0
    t0 = time.time()
    ranked = search(q_emb, fc_emb, fc_ids, top_n)
    search_s = time.time() - t0
    preds = dict(zip(post_ids, ranked))
    for p in missing:
        preds[p] = []
        post_langs[p] = "unk"
    timing = {"embed_total_s": embed_s, "search_total_s": search_s, "units": {"all": {"n_posts": len(post_ids), "pool": len(fc_ids)}}}
    info = {"pool": "all", "pool_size": len(fc_ids), "split": "sample", "track": "-", "embedding": einfo}
    return preds, gold, post_langs, info, timing


def run_official(cfg, variant, top_n):
    split_dir, track, eval_split = resolve(cfg["data_dir"]), cfg["track"], cfg["eval_split"]
    posts, fcs, _, failures = load_split(split_dir)
    if failures:
        print(f"WARNING: {len(failures)} unparsable cells (set to '')")
    units = eval_units(split_dir, track, eval_split)
    gold = {p: set(v) for p, v in load_reference(split_dir, track).items()}

    fc_ids = sorted({f for u in units for f in u["pool_fc_ids"]})
    post_ids = [p for u in units for p in u["post_ids"]]
    t0 = time.time()
    fc_texts = [fc_text(row, variant) for _, row in fcs.loc[fc_ids].iterrows()]
    post_texts = [post_text(posts.loc[p], variant) for p in post_ids]
    print(f"built {len(fc_ids)} fact-check and {len(post_ids)} post texts in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    q_emb, fc_emb, einfo = embed_all(
        cfg, variant, os.path.basename(os.path.normpath(split_dir)), post_ids, post_texts, fc_ids, fc_texts
    )
    embed_s = time.time() - t0
    print(f"embeddings ready in {embed_s:.1f}s (fc: {einfo['fact_checks']['n_embedded']} new, "
          f"{einfo['fact_checks']['n_cached']} cached; posts: {einfo['posts']['n_embedded']} new)", flush=True)

    fc_row = {f: i for i, f in enumerate(fc_ids)}
    post_row = {p: i for i, p in enumerate(post_ids)}
    preds, post_langs, timing = {}, {}, {"embed_total_s": embed_s, "units": {}}
    search_total = 0.0
    for u in units:
        lang, pool = u["lang"], u["pool_fc_ids"]
        t0 = time.time()
        ranked = search(q_emb[[post_row[p] for p in u["post_ids"]]], fc_emb[[fc_row[f] for f in pool]], pool, top_n)
        search_s = time.time() - t0
        search_total += search_s
        for p, r in zip(u["post_ids"], ranked):
            preds[p] = r
            post_langs[p] = lang
        timing["units"][lang] = {"search_s": search_s, "n_posts": len(u["post_ids"]), "pool": len(pool)}
        print(f"  {lang}: pool={len(pool)} posts={len(u['post_ids'])} search={search_s:.1f}s", flush=True)
    timing["search_total_s"] = search_total

    if track == "monolingual":
        pool_label, pool_size = "per-language", {u["lang"]: len(u["pool_fc_ids"]) for u in units}
    else:
        pool_label, pool_size = "all", len(units[0]["pool_fc_ids"])
    info = {"pool": pool_label, "pool_size": pool_size, "split": eval_split, "track": track, "embedding": einfo}
    return preds, gold, post_langs, info, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    seed_everything(SEED)
    with open(resolve(args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_name = cfg["run_name"]
    variant = cfg.get("text_variant", "eng")
    if variant not in VARIANTS:
        raise ValueError(f"text_variant must be one of {VARIANTS}, got {variant!r}")
    model = cfg.get("model", DEFAULT_MODEL)
    label = cfg.get("model_label", model)
    k = int(cfg.get("k", 10))
    top_n = int(cfg.get("top_n", 100))

    t_start = time.time()
    official = "track" in cfg
    preds, gold, post_langs, info, timing = (run_official if official else run_sample)(cfg, variant, top_n)
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
        "model": label,
        "model_id": model,
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
        run_name, variant, info["pool"], info["split"], track, headline, mrr, macro, model=label,
    )

    print(f"run:        {run_name}  (model={label}, variant={variant}, split={info['split']}, track={track}, pool={info['pool']})")
    print(f"posts:      {metrics['n_posts']}")
    print(f"S@{k}:       {headline:.4f}  ({'macro over languages' if track == 'monolingual' else 'micro'})")
    print(f"MRR@{k}:     {mrr:.4f}")
    print("per language:")
    for lang, v in metrics["per_language"].items():
        print(f"  {lang:>5}  S@{k}={v['s_at_k']:.4f}  MRR={v['mrr']:.4f}  n={v['n']}")
    for name in ("fact_checks", "posts"):
        t = info["embedding"][name]["tokens"]
        print(f"truncation {name}: {t.get('n_truncated')}/{t.get('n')} > max_seq_length ({t.get('frac_truncated', 0):.3%})")
    print(f"time:       embed {timing['embed_total_s']:.1f}s  search {timing['search_total_s']:.1f}s  total {timing['total_s']:.1f}s")
    print(f"wrote {os.path.relpath(out_dir, ROOT)}/metrics.json and predictions.json")


if __name__ == "__main__":
    main()
