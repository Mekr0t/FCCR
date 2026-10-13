"""The single evaluation function used by every experiment.

Metrics:
- S@k (Success@k): fraction of gold posts with >= 1 gold fact-check in the top k.
- MRR (= MRR@k, k=10 by default): mean reciprocal rank of the first gold fact-check, counted
  as 0 when it is not within the top k.
- Per-language S@k / MRR, and macro S@k (mean over languages), which is how the
  leaderboard averages.

Official scoring parity: the monolingual score is `macro_avg_s_at_k` (mean of per-language
S@10, languages = tasks.json keys); the crosslingual score is the micro `s_at_k`. Use
`official_score` to pick the right one for a track.

Only posts present in `gold` (with a non-empty gold set) are scored. A gold post
with no predictions counts as a miss. Predictions for other posts are ignored.
Ids are compared as strings, so int and str ids are interchangeable.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict


def _norm(x):
    return str(x)


def _score_one(ranked, gold_set, k):
    """Return (hit_at_k, reciprocal_rank_at_k) for one post."""
    for i, fc in enumerate(ranked[:k]):
        if fc in gold_set:
            return 1.0, 1.0 / (i + 1)
    return 0.0, 0.0


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def evaluate(predictions, gold, post_langs=None, k=10):
    """Evaluate ranked predictions against gold.

    predictions: {post_id: [fc_id, ...]} ranked, best first.
    gold: {post_id: set(fc_id)}.
    post_langs: optional {post_id: lang}; posts missing from it go to 'unk'.
    Returns a dict with s_at_k, mrr, n_posts and, if post_langs is given,
    per_language ({lang: {s_at_k, mrr, n}}), macro_avg_s_at_k and macro_mrr.
    """
    preds = {_norm(p): [_norm(f) for f in fcs] for p, fcs in predictions.items()}
    gold_n = {_norm(p): {_norm(f) for f in fs} for p, fs in gold.items() if fs}

    per_post = {
        pid: _score_one(preds.get(pid, []), gset, k) for pid, gset in gold_n.items()
    }
    result = {
        "k": k,
        "s_at_k": _mean(s for s, _ in per_post.values()),
        "mrr": _mean(r for _, r in per_post.values()),
        "n_posts": len(per_post),
    }

    if post_langs is not None:
        langs = {_norm(p): lang for p, lang in post_langs.items()}
        groups = defaultdict(list)
        for pid in per_post:
            groups[langs.get(pid, "unk")].append(pid)

        per_language = {}
        for lang in sorted(groups):
            ids = groups[lang]
            per_language[lang] = {
                "s_at_k": _mean(per_post[p][0] for p in ids),
                "mrr": _mean(per_post[p][1] for p in ids),
                "n": len(ids),
            }
        result["per_language"] = per_language
        result["macro_avg_s_at_k"] = _mean(v["s_at_k"] for v in per_language.values())
        result["macro_mrr"] = _mean(v["mrr"] for v in per_language.values())

    return result


def official_score(metrics, track):
    """The headline S@k of the official scorers: macro over languages (monolingual) or micro (crosslingual)."""
    if track == "monolingual":
        return metrics["macro_avg_s_at_k"]
    if track == "crosslingual":
        return metrics["s_at_k"]
    raise ValueError(f"unknown track {track!r}")


def save_predictions(preds, path, k=100):
    """Write {"<post_id>": [top-k fc ids as int]} in the official submission format.

    The official scorers need >= 10 ids per post and compare against int ids (string ids
    silently score 0), hence int conversion and k=100 like the organizers' baselines.
    """
    out = {str(p): [int(f) for f in fcs[:k]] for p, fcs in preds.items()}
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
