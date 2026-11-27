"""Weighted reciprocal rank fusion (RRF) of ranked retrieval runs.

score(d) = sum_i w_i / (k + rank_i(d)) with 1-based ranks; a document missing from list i
contributes nothing from it. Pure Python on {post_id: [fc_id, ...]} dicts, so it works on the
predictions.json of any run (nothing is recomputed).
"""


def rrf_fuse(runs, weights=None, k=60, top_n=100):
    """Fuse ranked runs with weighted RRF.

    runs: list of {post_id: [fc_id ranked best first]}; every run must cover the same posts.
    weights: one weight per run (default all 1.0). k: RRF constant (> 0). top_n: list length kept.
    Returns {str(post_id): [int(fc_id), ...]} (official prediction format).
    Ties are broken by: higher fused score, then lower (better) best individual rank, then
    smaller fc_id, so the output is deterministic.
    """
    if not runs:
        raise ValueError("need at least one run to fuse")
    weights = [1.0] * len(runs) if weights is None else [float(w) for w in weights]
    if len(weights) != len(runs):
        raise ValueError(f"{len(weights)} weights for {len(runs)} runs")
    if k <= 0:
        raise ValueError("k must be positive")
    norm = [{str(p): [int(f) for f in fcs] for p, fcs in run.items()} for run in runs]
    posts = set(norm[0])
    for i, run in enumerate(norm[1:], start=1):
        if set(run) != posts:
            raise ValueError(f"run {i} covers a different post set than run 0 ({len(set(run) ^ posts)} posts differ)")

    out = {}
    for p in sorted(posts, key=int):
        score, best = {}, {}
        for w, run in zip(weights, norm):
            seen = set()
            for rank, fc in enumerate(run[p], start=1):
                if fc in seen:  # a duplicate id inside one list counts once, at its best rank
                    continue
                seen.add(fc)
                score[fc] = score.get(fc, 0.0) + w / (k + rank)
                best[fc] = min(best.get(fc, rank), rank)
        ordered = sorted(score, key=lambda fc: (-score[fc], best[fc], fc))
        out[p] = ordered[:top_n]
    return out
