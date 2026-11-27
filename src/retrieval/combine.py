"""Score combination and chunking helpers for the truncation experiments (issue S2-4).

All functions work on L2-normalised embeddings (cosine = dot product) and score the FULL pool
of a unit, not only a shortlist.

- Variant A (separate orig + eng): `combine_scores` merges per-variant score matrices with
  max, mean or RRF (ranks of the full pool, equal weights).
- Variant B (chunking): `chunk_windows` splits a long token sequence into overlapping windows;
  `segment_max` takes the max cosine over a post's chunks; `mean_normalise` averages chunk
  embeddings and re-normalises.
- `subset_metrics` evaluates predictions on a subset of posts through src/evaluate.evaluate.
"""

from __future__ import annotations

import numpy as np

COMBINE_MODES = ("max", "mean", "rrf")


def chunk_windows(n_tokens, size=450, overlap=50):
    """(start, end) token windows covering [0, n_tokens), each at most `size` long.

    Consecutive windows overlap by `overlap` tokens; the last window ends exactly at n_tokens.
    A sequence of at most `size` tokens (or an empty one) gives a single window.
    """
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    if n_tokens <= size:
        return [(0, int(n_tokens))]
    stride = size - overlap
    out, start = [], 0
    while True:
        end = min(start + size, n_tokens)
        out.append((start, int(end)))
        if end >= n_tokens:
            return out
        start += stride


def mean_normalise(emb):
    """Mean of the rows of `emb` (k x d), re-normalised to unit length (float32)."""
    m = np.asarray(emb, dtype=np.float32).mean(axis=0)
    n = np.linalg.norm(m)
    return m / n if n > 0 else m


def ranks_desc(scores):
    """0-based rank of every column per row (torch tensor, rows x pool); rank 0 = highest score."""
    import torch

    order = torch.argsort(scores, dim=1, descending=True)
    ranks = torch.empty_like(order)
    cols = torch.arange(scores.shape[1], device=scores.device).expand_as(order)
    ranks.scatter_(1, order, cols)
    return ranks


def combine_scores(mats, mode, rrf_k=60, weights=None):
    """Combine equally shaped score matrices (rows x pool torch tensors) into one.

    mode 'max' / 'mean': elementwise over the cosine scores (weights apply to mean only).
    mode 'rrf': sum_i w_i / (rrf_k + rank_i + 1) with 1-based ranks over the whole pool.
    Returns a matrix to be ranked descending.
    """
    import torch

    if mode not in COMBINE_MODES:
        raise ValueError(f"mode must be one of {COMBINE_MODES}, got {mode!r}")
    w = [1.0] * len(mats) if weights is None else list(weights)
    if mode == "max":
        return torch.stack(mats).amax(dim=0)
    if mode == "mean":
        return sum(wi * m for wi, m in zip(w, mats)) / sum(w)
    return sum(wi / (rrf_k + ranks_desc(m).float() + 1.0) for wi, m in zip(w, mats))


def segment_max(scores, owner, n_owners):
    """Max over rows that share an owner. scores: R x N, owner: R long tensor in [0, n_owners)."""
    import torch

    out = torch.full((n_owners, scores.shape[1]), float("-inf"), device=scores.device, dtype=scores.dtype)
    return out.scatter_reduce_(0, owner.unsqueeze(1).expand_as(scores), scores, "amax", include_self=True)


def subset_metrics(preds, gold, post_langs, subset_ids, k=10):
    """evaluate() on the posts in `subset_ids` only (micro over those posts, plus per-language n)."""
    from src.evaluate import evaluate

    keep = {str(p) for p in subset_ids}
    p = {a: b for a, b in preds.items() if str(a) in keep}
    g = {a: b for a, b in gold.items() if str(a) in keep}
    return evaluate(p, g, {a: v for a, v in post_langs.items() if str(a) in keep}, k=k)
