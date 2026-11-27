r"""Fuse existing runs with weighted reciprocal rank fusion (nothing is re-retrieved).

Usage:
    .venv\Scripts\python.exe scripts\run_fusion.py --config configs\rrf_mono_dev_....yaml
    .venv\Scripts\python.exe scripts\run_fusion.py --sweep configs\sweeps\rrf_mono_dev.yaml [--write-configs]

Config keys: run_name, data_dir, track, eval_split (official release) or the sample keys
(data_dir, posts_file, fact_checks_file, mapping_file); `inputs` (list of {run, weight}, read from
results/<run>/predictions.json); `rrf_k`; text_variant and model_label (results.md columns only);
k (10); top_n (100).

A sweep YAML (configs/sweeps/, not matched by the config tests) has `name_prefix`, `base` (config
keys shared by all runs), `ks`, `bm25_weights` and `input_sets` ({set: {label, members: [{run, kind}]}},
kind = dense gets weight 1.0, kind = bm25 gets the swept weight). The sweep writes one config per
grid point to configs/<run_name>.yaml and, unless --write-configs, runs each of them.

eval_split test is refused unless --allow-test is passed (test runs are only for the final frozen
configuration). Outputs: results/<run_name>/metrics.json (also candidate recall S@20/50/100),
predictions.json (top_n, gitignored) and one row in results/results.md.
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

import yaml  # noqa: E402

from run_bm25 import resolve, update_results_table  # noqa: E402
from src.checks import assert_predictions_in_pool  # noqa: E402
from src.data import eval_units, gold_map, load_dataset, load_reference, post_language  # noqa: E402
from src.evaluate import evaluate, official_score, save_predictions  # noqa: E402
from src.retrieval.fusion import rrf_fuse  # noqa: E402

RECALL_KS = (20, 50, 100)


def load_inputs(cfg):
    """Read the input runs' predictions.json; returns (runs, weights)."""
    runs, weights = [], []
    for inp in cfg["inputs"]:
        with open(resolve(os.path.join("results", inp["run"], "predictions.json")), encoding="utf-8") as f:
            runs.append(json.load(f))
        weights.append(float(inp["weight"]))
    return runs, weights


def _scores(preds, gold, post_langs, track, k):
    """(headline S@k, MRR@k, full metrics) with the track's official aggregation."""
    m = evaluate(preds, gold, post_langs, k=k)
    if track == "monolingual":
        return official_score(m, track), m["macro_mrr"], m
    return m["s_at_k"], m["mrr"], m


def run_one(cfg, allow_test=False):
    split = cfg.get("eval_split")
    if split == "test" and not allow_test:
        raise SystemExit("refusing to run fusion on the test split (final frozen config only; pass --allow-test)")
    runs, weights = load_inputs(cfg)
    top_n, k = int(cfg.get("top_n", 100)), int(cfg.get("k", 10))
    t0 = time.time()
    preds = rrf_fuse(runs, weights, k=cfg["rrf_k"], top_n=top_n)
    fuse_s = time.time() - t0

    official = "track" in cfg
    if official:
        split_dir, track = resolve(cfg["data_dir"]), cfg["track"]
        units = eval_units(split_dir, track, split)
        gold = {p: set(v) for p, v in load_reference(split_dir, track).items()}
        post_langs = {p: u["lang"] for u in units for p in u["post_ids"]}
        ref_posts = {str(p) for p in gold}
        for inp, r in zip(cfg["inputs"], runs):
            if {str(p) for p in r} != ref_posts:
                raise ValueError(f"input run {inp['run']} does not cover exactly the reference post set")
        assert_predictions_in_pool(preds, units)
        pool = "per-language" if track == "monolingual" else "all"
    else:
        posts, _, mapping = load_dataset(
            resolve(cfg["data_dir"]), posts_file=cfg.get("posts_file", "posts.csv"),
            fact_checks_file=cfg.get("fact_checks_file", "fact_checks.csv"),
            mapping_file=cfg.get("mapping_file", "fact_check_post_mapping.csv"),
        )
        gold = gold_map(mapping)
        post_langs = {p: (post_language(posts.loc[p]) if p in posts.index else "unk") for p in gold}
        track, pool, split = "-", "all", "sample"

    headline, mrr, metrics = _scores(preds, gold, post_langs, track, k)
    macro = metrics.get("macro_avg_s_at_k") if (track == "monolingual" or not official) else None
    recall = {f"s_at_{kk}": _scores(preds, gold, post_langs, track, kk)[0] for kk in RECALL_KS}

    run_name, label, variant = cfg["run_name"], cfg.get("model_label", "RRF"), cfg.get("text_variant", "mixed")
    out_dir = resolve(os.path.join("results", run_name))
    os.makedirs(out_dir, exist_ok=True)
    record = {
        "run_name": run_name, "model": label, "model_id": "rrf", "text_variant": variant,
        "pool": pool, "split": split, "track": track,
        "inputs": cfg["inputs"], "weights": weights, "rrf_k": cfg["rrf_k"],
        "headline_s_at_10": headline, "headline_mrr_at_10": mrr,
        "headline_note": "monolingual: mean over languages; crosslingual: micro over posts" if official else "micro over posts",
        "candidate_recall": recall,
        "metrics": metrics, "timing": {"fuse_s": fuse_s}, "config": cfg,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    save_predictions(preds, os.path.join(out_dir, "predictions.json"), k=top_n)
    update_results_table(resolve(os.path.join("results", "results.md")), run_name, variant, pool, split,
                         track, headline, mrr, macro, model=label)
    print(f"{run_name}: S@10={headline:.4f} MRR={mrr:.4f} "
          + " ".join(f"S@{kk}={recall[f's_at_{kk}']:.4f}" for kk in RECALL_KS))
    return record


def expand_sweep(sw):
    """Expand a sweep mapping into a list of run configs (one per grid point)."""
    cfgs = []
    for set_name, spec in sw["input_sets"].items():
        for k in sw["ks"]:
            for wb in sw["bm25_weights"]:
                inputs = [{"run": m["run"], "weight": 1.0 if m["kind"] == "dense" else wb} for m in spec["members"]]
                cfg = {"run_name": f"{sw['name_prefix']}_{set_name}_k{k}_wb{wb}", **sw["base"],
                       "text_variant": spec["label"], "model_label": spec["model_label"],
                       "inputs": inputs, "rrf_k": k}
                cfgs.append(cfg)
    return cfgs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--config")
    g.add_argument("--sweep")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--write-configs", action="store_true", help="with --sweep: only write the expanded configs")
    args = ap.parse_args()
    if args.config:
        with open(resolve(args.config), encoding="utf-8") as f:
            run_one(yaml.safe_load(f), args.allow_test)
        return
    with open(resolve(args.sweep), encoding="utf-8") as f:
        cfgs = expand_sweep(yaml.safe_load(f))
    for cfg in cfgs:
        with open(resolve(os.path.join("configs", cfg["run_name"] + ".yaml")), "w", encoding="utf-8", newline="\n") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {len(cfgs)} configs to configs/")
    if not args.write_configs:
        for cfg in cfgs:
            run_one(cfg, args.allow_test)


if __name__ == "__main__":
    main()
