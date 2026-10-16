"""Runtime sanity checks for prediction dicts, run before a prediction file is saved.

`assert_predictions_in_pool` is the guard against the most expensive silent mistake in this
project: a prediction file whose fact-check ids come from outside a post's retrieval pool, or
that has too few / duplicated ids, which the official scorer would either reject or score
inconsistently. Ids are compared as strings, so int and str ids are interchangeable.
"""

from __future__ import annotations


def assert_predictions_in_pool(preds, units, min_per_post=10):
    """Raise ValueError if `preds` is not a valid official-style prediction set for `units`.

    preds: {post_id: [fc_id, ...]} ranked lists (int or str ids).
    units: output of src.data.eval_units, i.e. [{"lang", "post_ids", "pool_fc_ids"}, ...].

    Checks, for every unit and every post in it:
      - the post has predictions at all, and at least `min_per_post` of them;
      - no fact-check id is repeated within the post's list;
      - every predicted fact-check id is in that unit's pool (the post's own language pool
        for monolingual, the full pool for crosslingual).
    Also checks that the set of predicted posts equals the set of posts in the units.
    Returns None on success; raises ValueError listing up to 10 problems otherwise.
    """
    problems = []
    preds_n = {str(p): [str(f) for f in fcs] for p, fcs in preds.items()}
    expected_posts = set()

    for u in units:
        lang = u["lang"]
        pool = {str(f) for f in u["pool_fc_ids"]}
        for p in u["post_ids"]:
            pid = str(p)
            expected_posts.add(pid)
            if pid not in preds_n:
                problems.append(f"[{lang}] post {pid}: no predictions")
                continue
            fcs = preds_n[pid]
            if len(fcs) < min_per_post:
                problems.append(f"[{lang}] post {pid}: {len(fcs)} predictions, need >= {min_per_post}")
            if len(set(fcs)) != len(fcs):
                problems.append(f"[{lang}] post {pid}: duplicated fact-check ids")
            outside = sorted({f for f in fcs if f not in pool})
            if outside:
                problems.append(f"[{lang}] post {pid}: {len(outside)} ids outside the pool, e.g. {outside[:3]}")

    extra = sorted(set(preds_n) - expected_posts)
    if extra:
        problems.append(f"{len(extra)} predicted posts are not evaluated, e.g. {extra[:3]}")

    if problems:
        shown = problems[:10]
        more = f" (+{len(problems) - 10} more)" if len(problems) > 10 else ""
        raise ValueError("invalid predictions: " + "; ".join(shown) + more)
