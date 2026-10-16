"""Toy parity: src/evaluate.py vs the copied official logic in scripts/score_official.py.

Overlap with tests/test_evaluate.py: the rank-10 / rank-11 boundary, multi-gold first hit,
macro vs micro, official_score selection and the save_predictions format are already unit-tested
there. This file adds what test_evaluate.py does not: agreement with the organizers' functions
(calculate_success_at_k incl. their asserts), a JSON round trip through save_predictions with
int fact-check ids, and the one intended difference (a post without predictions).
"""

import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import score_official as so  # noqa: E402
from src.evaluate import evaluate, official_score, save_predictions  # noqa: E402

K = 10

# Per-language fact-check pools of different sizes (monolingual) and their union (crosslingual).
POOLS = {"eng": list(range(1000, 1014)), "spa": list(range(2000, 2012)), "deu": list(range(3000, 3016))}

# post -> (lang, gold ids, {gold id: 1-based rank it must occupy}). Every other pool id fills
# the remaining ranks in pool order. Covers: multi-gold with a hit on a non-first gold, hit at
# rank 10, miss at rank 11, miss at rank 12, hit at rank 1.
SPEC = {
    "101": ("eng", [1002, 1003], {1002: 3, 1003: 7}),
    "102": ("eng", [1009], {1009: 10}),
    "103": ("eng", [1010], {1010: 11}),
    "104": ("eng", [1001], {1001: 1}),
    "201": ("spa", [2011], {2011: 12}),
    "202": ("spa", [2000, 2005], {2000: 1, 2005: 6}),
    "203": ("spa", [2008], {2008: 9}),
    "301": ("deu", [3015, 3002], {3015: 9, 3002: 2}),
    "302": ("deu", [3010], {3010: 11}),
    "303": ("deu", [3000], {3000: 1}),
    "304": ("deu", [3009], {3009: 10}),
}
EXPECTED_HIT = {"101": True, "102": True, "103": False, "104": True, "201": False, "202": True,
                "203": True, "301": True, "302": False, "303": True, "304": True}


def _ranking(post):
    lang, _, placed = SPEC[post]
    fill = iter(f for f in POOLS[lang] if f not in placed)
    out = []
    for rank in range(1, len(POOLS[lang]) + 1):
        out.append(next(f for f, r in placed.items() if r == rank) if rank in placed.values() else next(fill))
    return out


def _posts():
    return {p: spec[0] for p, spec in SPEC.items()}


def _truth():
    """Reference-style: string post ids, int fc ids (as load_reference returns)."""
    return {p: list(spec[1]) for p, spec in SPEC.items()}


def _round_trip(tmp_path, preds):
    """Write with save_predictions (int ids, int post keys) and load back as the scorers do."""
    path = tmp_path / "pred.json"
    save_predictions({int(p): fcs for p, fcs in preds.items()}, str(path))
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_fixture_ranks_are_as_documented():
    for post, (_, gold, _) in SPEC.items():
        ranked = _ranking(post)
        rank = min(ranked.index(g) + 1 for g in gold)
        hit = rank <= K
        assert hit == EXPECTED_HIT[post], (post, rank)


def test_monolingual_parity_with_official_logic(tmp_path):
    pred = _round_trip(tmp_path, {p: _ranking(p) for p in SPEC})
    m = evaluate(pred, {p: set(g) for p, g in _truth().items()}, _posts(), k=K)

    tasks = {"monolingual": {l: {"posts_train": [], "posts_dev": [int(p) for p, pl in _posts().items() if pl == l]}
                             for l in POOLS}}
    off = so.official_mono(_truth(), pred, K, so.official_post_id_to_lang(tasks, "train_dev"))
    avg_off = float(off.pop("avg"))

    assert official_score(m, "monolingual") == pytest.approx(avg_off, abs=1e-12)
    for lang in POOLS:
        assert m["per_language"][lang]["s_at_k"] == pytest.approx(off[lang], abs=1e-12), lang
    assert off == pytest.approx({"eng": 3 / 4, "spa": 2 / 3, "deu": 3 / 4})
    # macro (mean over languages) differs from micro (8 hits of 11 posts) on this toy set
    assert avg_off == pytest.approx((3 / 4 + 2 / 3 + 3 / 4) / 3)
    assert avg_off != pytest.approx(m["s_at_k"])


def test_crosslingual_parity_with_official_logic(tmp_path):
    pred = _round_trip(tmp_path, {p: _ranking(p) for p in SPEC})
    m = evaluate(pred, {p: set(g) for p, g in _truth().items()}, _posts(), k=K)
    off = so.official_cross(_truth(), pred, K)
    assert official_score(m, "crosslingual") == pytest.approx(off, abs=1e-12)
    assert off == pytest.approx(8 / 11)


def test_missing_post_official_raises_ours_counts_zero(tmp_path):
    # Intended difference. The official scorers refuse a submission that lacks a gold post:
    # the length assert in calculate_success_at_k fires first, and if the lengths happen to
    # match, the 'Missing post id' raise does. Our evaluator is lenient on purpose: a gold post
    # without predictions is a miss (so an incomplete run scores lower, it does not crash).
    pred = _round_trip(tmp_path, {p: _ranking(p) for p in SPEC if p != "101"})
    with pytest.raises(AssertionError, match="number of prediction"):
        so.official_cross(_truth(), pred, K)

    full = _round_trip(tmp_path, {p: _ranking(p) for p in SPEC})
    same_len = {("999999" if p == "101" else p): fcs for p, fcs in full.items()}  # same count, 101 absent
    with pytest.raises(Exception, match="Missing post id"):
        so.official_cross(_truth(), same_len, K)

    m = evaluate(pred, {p: set(g) for p, g in _truth().items()}, _posts(), k=K)
    assert m["n_posts"] == len(SPEC)
    assert official_score(m, "crosslingual") == pytest.approx(7 / 11)  # 101 counted as a miss
