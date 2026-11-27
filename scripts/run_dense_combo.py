r"""Dense retrieval with score combination / chunking to counter truncation (issue S2-4).

Usage:
    .venv\Scripts\python.exe scripts\run_dense_combo.py --config configs\dense_mono_dev_origeng_max.yaml

Same config keys as run_dense.py plus `combo`:
  origeng_max | origeng_mean | origeng_rrf   Variant A: cos(post_orig, fc_orig) and
                                             cos(post_eng, fc_eng) combined over the FULL pool
                                             (rrf: ranks of the two full rankings, k=60, equal weights)
  origeng_both_mean                          3-way mean of the orig, eng and both scores
  both_chunkmax | both_chunkmean             Variant B: posts whose `both` text exceeds
                                             max_seq_length tokens are split into windows
                                             (`chunk_size` 450 tokens, `chunk_overlap` 50), each
                                             window is embedded with the `query: ` prefix and scored
                                             against the existing `both` fact-check embeddings
                                             (max over chunks, or mean chunk embedding re-normalised).
                                             Other posts keep their `both` embedding.
Optional: `rrf_k` (60). Only post chunks are ever embedded here; fact-check and full-post
embeddings come from the caches written by run_dense.py (cache/<model>/<split>/<variant>[_posts]).
Chunk embeddings go to cache/<model>/<split>/both_chunks_posts/ (new directory).

A test split is refused unless --allow-test is passed (and the issue says "final test run").
Outputs: results/<run_name>/metrics.json (with truncation subset metrics), predictions.json
(top-100, gitignored) and a results/results.md row.
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
from run_dense import embed_all  # noqa: E402
from src.checks import assert_predictions_in_pool  # noqa: E402
from src.data import (  # noqa: E402
    eval_units,
    fc_text,
    gold_map,
    load_dataset,
    load_reference,
    load_split,
    post_language,
    post_text,
)
from src.evaluate import evaluate, official_score, save_predictions  # noqa: E402
from src.retrieval.combine import (  # noqa: E402
    chunk_windows,
    combine_scores,
    mean_normalise,
    segment_max,
    subset_metrics,
)
from src.retrieval.dense import (  # noqa: E402
    DEFAULT_MODEL,
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    DenseEncoder,
    EmbeddingStore,
    model_slug,
    seed_everything,
)

SEED = 42
CHUNK_ID_STRIDE = 1000  # chunk id = post_id * 1000 + chunk index
COMBOS = {
    "origeng_max": ("orig+eng", ["orig", "eng"], "max"),
    "origeng_mean": ("orig+eng", ["orig", "eng"], "mean"),
    "origeng_rrf": ("orig+eng", ["orig", "eng"], "rrf"),
    "origeng_both_mean": ("orig+eng+both", ["orig", "eng", "both"], "mean"),
    "both_chunkmax": ("both+chunkmax", ["both"], "chunkmax"),
    "both_chunkmean": ("both+chunkmean", ["both"], "chunkmean"),
}


def load_inputs(cfg):
    """(units, gold, post_langs, texts, fc_texts, split_name, info) for the official or sample layout.

    units: [{lang, post_ids, pool_fc_ids}]; texts[variant] / fc_texts[variant] are aligned with
    all_post_ids / all_fc_ids (the same ordering run_dense.py uses).
    """
    variants = ("orig", "eng", "both")
    if "track" in cfg:
        split_dir, track, eval_split = resolve(cfg["data_dir"]), cfg["track"], cfg["eval_split"]
        posts, fcs, _, failures = load_split(split_dir)
        if failures:
            print(f"WARNING: {len(failures)} unparsable cells (set to '')")
        units = eval_units(split_dir, track, eval_split)
        gold = {p: set(v) for p, v in load_reference(split_dir, track).items()}
        fc_ids = sorted({f for u in units for f in u["pool_fc_ids"]})
        post_ids = [p for u in units for p in u["post_ids"]]
        post_langs = {p: u["lang"] for u in units for p in u["post_ids"]}
        fc_rows = [row for _, row in fcs.loc[fc_ids].iterrows()]
        post_rows = [posts.loc[p] for p in post_ids]
        split_name, split_label = os.path.basename(os.path.normpath(split_dir)), eval_split
    else:
        posts, fcs, mapping = load_dataset(
            resolve(cfg["data_dir"]),
            posts_file=cfg.get("posts_file", "posts.csv"),
            fact_checks_file=cfg.get("fact_checks_file", "fact_checks.csv"),
            mapping_file=cfg.get("mapping_file", "fact_check_post_mapping.csv"),
        )
        gold = gold_map(mapping)
        post_ids = [p for p in sorted(gold) if p in posts.index]
        fc_ids = list(fcs.index)
        units = [{"lang": "all", "post_ids": post_ids, "pool_fc_ids": fc_ids}]
        post_langs = {p: post_language(posts.loc[p]) for p in post_ids}
        fc_rows = [row for _, row in fcs.iterrows()]
        post_rows = [posts.loc[p] for p in post_ids]
        split_name, split_label, track = os.path.basename(os.path.normpath(cfg["data_dir"])), "sample", "-"
    p_txt = {v: [post_text(r, v) for r in post_rows] for v in variants}
    f_txt = {v: [fc_text(r, v) for r in fc_rows] for v in variants}
    info = {"track": track, "split": split_label}
    return units, gold, post_langs, post_ids, fc_ids, p_txt, f_txt, split_name, info


def chunk_posts(cfg, encoder, split_name, post_ids, both_texts, ntok):
    """Embed windows of the posts whose `both` text exceeds max_seq_length.

    Returns ({post_id: (k x d float16 chunk embeddings)}, info). Windows are cut on the raw
    token ids (no prefix, no special tokens) and decoded back to text, so each chunk with the
    `query: ` prefix and the two special tokens stays far below max_seq_length.
    """
    model = cfg.get("model", DEFAULT_MODEL)
    max_len = int(cfg.get("max_seq_length", 512))
    qp = cfg.get("query_prefix", QUERY_PREFIX)
    size, overlap = int(cfg.get("chunk_size", 450)), int(cfg.get("chunk_overlap", 50))
    tok = encoder.model.tokenizer
    ids, texts, owner = [], [], []
    for i, (pid, text, n) in enumerate(zip(post_ids, both_texts, ntok)):
        if n <= max_len:
            continue
        raw = tok(text, add_special_tokens=False)["input_ids"]
        for j, (s, e) in enumerate(chunk_windows(len(raw), size, overlap)):
            assert j < CHUNK_ID_STRIDE
            ids.append(int(pid) * CHUNK_ID_STRIDE + j)
            texts.append(qp + tok.decode(raw[s:e], skip_special_tokens=True))
            owner.append(int(pid))
    base = os.path.join(ROOT, "cache", model_slug(model), split_name)
    store = EmbeddingStore(os.path.join(base, "both_chunks_posts"), model, max_len, qp)
    if ids:
        emb, einfo = store.get(encoder, ids, texts, "both_chunks")
        cn = einfo.pop("ntok")
    else:
        emb, einfo, cn = np.zeros((0, 1), dtype=np.float16), {"n": 0, "n_embedded": 0, "n_cached": 0, "embed_s": 0.0}, np.zeros(0, dtype=np.int32)
    out = {}
    for pid, e in zip(owner, emb):
        out.setdefault(pid, []).append(e)
    out = {p: np.stack(v) for p, v in out.items()}
    info = {
        **einfo,
        "n_chunked_posts": len(out),
        "n_chunks": len(ids),
        "chunk_size": size,
        "chunk_overlap": overlap,
        "max_chunk_tokens_with_prefix": int(cn.max()) if len(cn) else 0,
        "n_chunks_over_max_seq_length": int((cn > max_len).sum()),
    }
    return out, info


def run(cfg, combo, allow_test):
    if cfg.get("eval_split") == "test" and not allow_test:
        raise SystemExit("refusing eval_split=test without --allow-test (test runs only for a frozen configuration)")
    label, comps, mode = COMBOS[combo]
    units, gold, post_langs, post_ids, fc_ids, p_txt, f_txt, split_name, info = load_inputs(cfg)
    model = cfg.get("model", DEFAULT_MODEL)
    max_len = int(cfg.get("max_seq_length", 512))
    qp = cfg.get("query_prefix", QUERY_PREFIX)

    t0 = time.time()
    Q, P, einfo = {}, {}, {}
    for v in comps:
        Q[v], P[v], einfo[v] = embed_all(cfg, v, split_name, post_ids, p_txt[v], fc_ids, f_txt[v])
        n_new = einfo[v]["fact_checks"]["n_embedded"] + einfo[v]["posts"]["n_embedded"]
        print(f"  {v}: embeddings ready ({n_new} newly embedded)", flush=True)
    encoder = DenseEncoder(model, max_len, int(cfg.get("batch_size", 32)))
    # truncation flag = same definition as the post stores' ntok.npy: tokens of "query: " + both text,
    # special tokens (<s>, </s>) included, before truncation; affected = ntok > max_seq_length
    ntok_both = encoder.token_lengths([qp + t for t in p_txt["both"]])
    affected = [int(p) for p, n in zip(post_ids, ntok_both) if n > max_len]

    post_rows_of = {int(p): [i] for i, p in enumerate(post_ids)}  # rows into q_rows per post
    chunk_info = None
    if mode in ("chunkmax", "chunkmean"):
        chunks, chunk_info = chunk_posts(cfg, encoder, split_name, post_ids, p_txt["both"], ntok_both)
        print(f"  chunked {chunk_info['n_chunked_posts']} posts into {chunk_info['n_chunks']} windows "
              f"(max {chunk_info['max_chunk_tokens_with_prefix']} tokens)", flush=True)
        q = Q["both"].copy()
        if mode == "chunkmean":
            for p, e in chunks.items():
                q[post_rows_of[p][0]] = mean_normalise(e)
            Q["both"] = q
        else:
            extra, rows = [], {p: [r[0]] for p, r in post_rows_of.items()}
            for p, e in chunks.items():
                rows[p] = []
                for vec in e:
                    rows[p].append(len(q) + len(extra))
                    extra.append(vec)
            Q["both"] = np.concatenate([q, np.stack(extra)]) if extra else q
            post_rows_of = rows

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fc_row = {f: i for i, f in enumerate(fc_ids)}
    post_row = {p: i for i, p in enumerate(post_ids)}
    rrf_k = int(cfg.get("rrf_k", 60))
    top_n, batch = int(cfg.get("top_n", 100)), 128
    preds, t_search = {}, time.time()
    for u in units:
        pool = np.asarray(u["pool_fc_ids"])
        sel = [fc_row[f] for f in pool]
        Pm = {v: torch.from_numpy(np.ascontiguousarray(P[v][sel])).to(device).float() for v in comps}
        k = min(top_n, len(pool))
        upids = u["post_ids"]
        for s in range(0, len(upids), batch):
            bp = upids[s : s + batch]
            mats = []
            with torch.inference_mode():
                for v in comps:
                    if mode == "chunkmax":
                        rows = [r for p in bp for r in post_rows_of[int(p)]]
                        owner = torch.tensor([i for i, p in enumerate(bp) for _ in post_rows_of[int(p)]], device=device)
                        Qb = torch.from_numpy(np.ascontiguousarray(Q[v][rows])).to(device).float()
                        mats.append(segment_max(Qb @ Pm[v].T, owner, len(bp)))
                    else:
                        Qb = torch.from_numpy(np.ascontiguousarray(Q[v][[post_row[p] for p in bp]])).to(device).float()
                        mats.append(Qb @ Pm[v].T)
                sc = mats[0] if len(mats) == 1 else combine_scores(mats, mode, rrf_k=rrf_k)
                top = torch.topk(sc, k, dim=1).indices.cpu().numpy()
            for p, t in zip(bp, top):
                preds[p] = pool[t].tolist()
        print(f"  {u['lang']}: pool={len(pool)} posts={len(upids)}", flush=True)
        del Pm
    search_s = time.time() - t_search
    embed_s = time.time() - t0 - search_s
    return preds, gold, post_langs, post_ids, affected, units, label, chunk_info, einfo, {"embed_total_s": embed_s, "search_total_s": search_s}, info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-test", action="store_true", help="permit eval_split=test (frozen configuration only)")
    args = parser.parse_args()

    seed_everything(SEED)
    with open(resolve(args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    combo = cfg["combo"]
    if combo not in COMBOS:
        raise ValueError(f"combo must be one of {sorted(COMBOS)}, got {combo!r}")
    run_name, k = cfg["run_name"], int(cfg.get("k", 10))
    model_label = cfg.get("model_label", cfg.get("model", DEFAULT_MODEL))

    t_start = time.time()
    preds, gold, post_langs, post_ids, affected, units, label, chunk_info, einfo, timing, info = run(cfg, combo, args.allow_test)
    timing["total_s"] = time.time() - t_start
    track = info["track"]
    official = "track" in cfg

    metrics = evaluate(preds, gold, post_langs, k=k)
    if official:
        headline = official_score(metrics, track)
        mrr = metrics["macro_mrr"] if track == "monolingual" else metrics["mrr"]
        macro = metrics["macro_avg_s_at_k"] if track == "monolingual" else None
    else:
        headline, mrr, macro = metrics["s_at_k"], metrics["mrr"], metrics["macro_avg_s_at_k"]

    aff = set(affected)
    unaff = [p for p in post_ids if int(p) not in aff]
    per_lang_aff = {}
    for p in post_ids:
        d = per_lang_aff.setdefault(post_langs[p], {"n": 0, "n_affected": 0})
        d["n"] += 1
        d["n_affected"] += int(p) in aff
    truncation = {
        "definition": "posts whose `both` text, prefixed with 'query: ' and including <s>/</s>, has more than "
        "max_seq_length tokens (same ntok as cache/<model>/<split>/both_posts/ntok.npy)",
        "n_posts": len(post_ids),
        "n_affected": len(affected),
        "frac_affected": len(affected) / max(1, len(post_ids)),
        "per_language": per_lang_aff,
        "subset_note": "subset metrics are micro over the posts of the subset (also for monolingual)",
        "affected": subset_metrics(preds, gold, post_langs, affected, k=k),
        "unaffected": subset_metrics(preds, gold, post_langs, unaff, k=k),
    }

    pool_label = "per-language" if (official and track == "monolingual") else "all"
    pool_size = {u["lang"]: len(u["pool_fc_ids"]) for u in units} if pool_label == "per-language" else len(units[0]["pool_fc_ids"])
    out_dir = resolve(os.path.join("results", run_name))
    os.makedirs(out_dir, exist_ok=True)
    record = {
        "run_name": run_name,
        "model": model_label,
        "model_id": cfg.get("model", DEFAULT_MODEL),
        "text_variant": label,
        "combo": combo,
        "pool": pool_label,
        "pool_size": pool_size,
        "split": info["split"],
        "track": track,
        "seed": SEED,
        "headline_s_at_10": headline,
        "headline_mrr_at_10": mrr,
        "headline_note": "monolingual: mean over languages; crosslingual: micro over posts" if official else "micro over posts",
        "metrics": metrics,
        "truncation": truncation,
        "chunking": chunk_info,
        "embedding_cache_use": {v: {n: {kk: vv for kk, vv in e[n].items() if kk != "tokens"} for n in ("fact_checks", "posts")} for v, e in einfo.items()},
        "timing": timing,
        "config": cfg,
    }
    if official:
        assert_predictions_in_pool(preds, eval_units(resolve(cfg["data_dir"]), track, cfg["eval_split"]))
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    save_predictions(preds, os.path.join(out_dir, "predictions.json"), k=100)
    update_results_table(
        resolve(os.path.join("results", "results.md")),
        run_name, label, pool_label, info["split"], track, headline, mrr, macro, model=model_label,
    )

    print(f"run:        {run_name}  (combo={combo}, split={info['split']}, track={track}, pool={pool_label})")
    print(f"S@{k}:       {headline:.4f}   MRR@{k}: {mrr:.4f}")
    print(f"affected:   {len(affected)}/{len(post_ids)}  S@{k}={truncation['affected']['s_at_k']:.4f}  "
          f"unaffected S@{k}={truncation['unaffected']['s_at_k']:.4f}")
    print(f"time:       embed {timing['embed_total_s']:.1f}s  search {timing['search_total_s']:.1f}s")


if __name__ == "__main__":
    main()
