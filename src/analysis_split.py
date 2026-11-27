"""Deterministic 50/50 split of dev posts into an error-analysis half and a held-out half.

Ids only. Reuses src.data.eval_units (no second loader). Stratum = the monolingual language
if the post is in the monolingual dev set, else "cross_only". A post present in both tracks
gets one half (it is stratified by its monolingual language).
"""

from __future__ import annotations

import math
import random

from src.data import eval_units

DEFAULT_SEED = 20261127
CROSS_ONLY = "cross_only"


def split_dev_posts(split_dir, seed=DEFAULT_SEED):
    """Return {"analysis": [ids], "heldout": [ids], "counts": {stratum: {"analysis": n, "heldout": n}}}."""
    strata = {}
    for unit in eval_units(split_dir, "monolingual", "dev"):
        for p in unit["post_ids"]:
            strata[p] = unit["lang"]
    for unit in eval_units(split_dir, "crosslingual", "dev"):
        for p in unit["post_ids"]:
            strata.setdefault(p, CROSS_ONLY)

    by_stratum = {}
    for p, s in strata.items():
        by_stratum.setdefault(s, []).append(p)

    analysis, heldout, counts = [], [], {}
    for s in sorted(by_stratum):
        ids = sorted(by_stratum[s])
        # string seeds are hashed with sha512 by random.Random: stable across runs and platforms
        random.Random(f"{seed}:{s}").shuffle(ids)
        k = math.ceil(len(ids) / 2)
        analysis += ids[:k]
        heldout += ids[k:]
        counts[s] = {"analysis": k, "heldout": len(ids) - k}
    return {"analysis": sorted(analysis), "heldout": sorted(heldout), "counts": counts}
