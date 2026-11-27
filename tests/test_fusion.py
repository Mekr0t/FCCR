"""Unit tests for weighted RRF (src/retrieval/fusion.py) on hand-computed toy examples, plus a
sample-data smoke test of scripts/run_fusion.py (skipped when the sample predictions are absent)."""

import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src.retrieval.fusion import rrf_fuse  # noqa: E402


def test_rank_math_two_lists():
    a = {"1": [10, 20, 30]}
    b = {"1": [20, 10, 40]}
    # k=1: 10 -> 1/2 + 1/3 = 5/6; 20 -> 1/3 + 1/2 = 5/6 (tie); 30 -> 1/4; 40 -> 1/4 (tie)
    # ties: equal score and equal best rank -> smaller id first
    out = rrf_fuse([a, b], k=1)
    assert out["1"] == [10, 20, 30, 40]


def test_scores_exact_ordering():
    a = {"1": [1, 2, 3]}
    b = {"1": [3, 2, 1]}
    # k=60: 1 and 3 -> 1/61 + 1/63 = 0.032266 beat 2 -> 2/62 = 0.032258
    s2 = 2 / 62
    s13 = 1 / 61 + 1 / 63
    assert s13 > s2
    assert rrf_fuse([a, b], k=60)["1"] == [1, 3, 2]


def test_weights_change_order():
    a = {"1": [1, 2]}
    b = {"1": [2, 1]}
    assert rrf_fuse([a, b], [1.0, 1.0], k=10)["1"] == [1, 2]  # tie, smaller id
    assert rrf_fuse([a, b], [1.0, 0.5], k=10)["1"] == [1, 2]
    assert rrf_fuse([a, b], [0.5, 1.0], k=10)["1"] == [2, 1]


def test_document_missing_from_one_list_contributes_zero():
    a = {"1": [1, 2]}
    b = {"1": [3]}
    # k=1: 1 -> 1/2, 2 -> 1/3, 3 -> 1/2 ; 1 and 3 tie on score and best rank -> smaller id first
    assert rrf_fuse([a, b], k=1)["1"] == [1, 3, 2]
    # doc only in the weak list loses to a doc ranked 1st in the strong list
    assert rrf_fuse([a, b], [1.0, 0.1], k=1)["1"] == [1, 2, 3]


def test_tie_break_prefers_better_best_rank_then_id():
    a = {"1": [5, 9]}
    b = {"1": [9, 5]}
    c = {"1": [7]}
    # k=1: 5 -> 1/2 + 1/3, 9 -> 1/3 + 1/2 (equal), 7 -> 1/2; 5 and 9 both best rank 1 -> id order
    assert rrf_fuse([a, b], k=1)["1"] == [5, 9]
    p = {"1": [0, 8, 6]}  # 8 at rank 2
    q = {"1": [1, 8]}  # 8 at rank 2
    out = rrf_fuse([p, q], k=1)["1"]
    assert out[0] == 8  # 2/3 beats 1/2
    # deterministic: repeated call and input list order do not change ties
    assert rrf_fuse([p, q], k=1) == rrf_fuse([p, q], k=1)


def test_tie_break_by_best_rank():
    # doc 50: rank 1 in A (w=1, k=1) -> 1/2; doc 10: rank 2 in B (w=1.5) -> 1.5/3 = 1/2 (exact in floats).
    # Equal fused score; 50 has the better best rank so it beats the smaller id 10.
    a = {"1": [50]}
    b = {"1": [99, 10]}
    assert rrf_fuse([a, b], [1.0, 1.5], k=1)["1"] == [99, 50, 10]


def test_k_effect():
    # small k rewards a top rank, large k rewards appearing in several lists
    x = {"1": [1, 7, 2]}
    y = {"1": [3, 8, 2]}
    # doc 1: 1/(k+1); doc 2: 2/(k+3)
    assert rrf_fuse([x, y], k=1)["1"][0] == 1  # 1/2 vs 2/4: tie on score, best rank 1 beats 3
    assert rrf_fuse([x, y], k=100)["1"][0] == 2  # 1/101 < 2/103
    assert rrf_fuse([x, y], k=10)["1"][0] == 2  # 1/11 < 2/13


def test_top_n_and_output_format():
    a = {"7": list(range(10, 0, -1))}
    out = rrf_fuse([a, a], k=60, top_n=3)
    assert list(out) == ["7"] and out["7"] == [10, 9, 8]
    assert all(isinstance(f, int) for f in out["7"])


def test_ids_are_normalised():
    a = {1: ["5", "6"]}
    b = {"1": [6, 5]}
    out = rrf_fuse([a, b], k=60)
    assert list(out) == ["1"] and out["1"] == [5, 6]


def test_mismatched_post_sets_raise():
    with pytest.raises(ValueError):
        rrf_fuse([{"1": [1]}, {"1": [1], "2": [1]}])
    with pytest.raises(ValueError):
        rrf_fuse([{"1": [1]}, {"2": [1]}])


def test_bad_arguments_raise():
    with pytest.raises(ValueError):
        rrf_fuse([])
    with pytest.raises(ValueError):
        rrf_fuse([{"1": [1]}], weights=[1.0, 2.0])
    with pytest.raises(ValueError):
        rrf_fuse([{"1": [1]}], k=0)


def test_single_run_keeps_order():
    assert rrf_fuse([{"1": [4, 2, 9]}], k=60)["1"] == [4, 2, 9]


def test_duplicates_inside_a_list_count_once():
    assert rrf_fuse([{"1": [1, 1, 2]}], k=1)["1"] == [1, 2]


# ---- smoke test on the public sample ------------------------------------------------------
SAMPLE_PREDS = [os.path.join(ROOT, "results", r, "predictions.json") for r in ("bm25_sample_eng", "dense_sample_eng")]


@pytest.mark.skipif(
    not (all(os.path.isfile(p) for p in SAMPLE_PREDS) and os.path.isdir(os.path.join(ROOT, "data", "raw", "sample"))),
    reason="sample data or sample predictions not present locally",
)
def test_sample_fusion_smoke(monkeypatch, tmp_path):
    import run_fusion
    import yaml

    with open(os.path.join(ROOT, "configs", "rrf_sample_eng.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    real_resolve = run_fusion.resolve

    def resolve(path):  # outputs of this test go to tmp_path, inputs are read from the real results/
        norm = path.replace("\\", "/")
        if norm.startswith("results/") and not norm.endswith("/predictions.json") or cfg["run_name"] in norm:
            return str(tmp_path / norm.replace("/", "_"))
        return real_resolve(path)

    monkeypatch.setattr(run_fusion, "resolve", resolve)
    rec = run_fusion.run_one(cfg)
    assert 0.0 <= rec["headline_s_at_10"] <= 1.0
    r = rec["candidate_recall"]
    assert r["s_at_20"] <= r["s_at_50"] <= r["s_at_100"]
    assert rec["headline_s_at_10"] <= r["s_at_20"]
    assert rec["track"] == "-" and rec["split"] == "sample"
    assert (tmp_path / ("results_" + cfg["run_name"]) / "metrics.json").is_file()


def test_test_split_is_refused():
    import run_fusion

    with pytest.raises(SystemExit):
        run_fusion.run_one({"eval_split": "test", "inputs": [], "rrf_k": 60})
