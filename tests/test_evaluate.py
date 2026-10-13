import json

from src.evaluate import evaluate, official_score, save_predictions


def _filler(n, prefix="x"):
    return [f"{prefix}{i}" for i in range(n)]


def test_s_at_10_hit_at_rank_10_counts():
    gold = {"p1": {"g"}}
    preds = {"p1": _filler(9) + ["g"]}  # rank 10
    m = evaluate(preds, gold, k=10)
    assert m["s_at_k"] == 1.0
    assert abs(m["mrr"] - 0.1) < 1e-12


def test_s_at_10_hit_at_rank_11_does_not_count():
    gold = {"p1": {"g"}, "p2": {"g"}}
    preds = {
        "p1": _filler(9) + ["g"],       # rank 10 -> hit
        "p2": _filler(10) + ["g"],      # rank 11 -> miss
    }
    m = evaluate(preds, gold, k=10)
    assert m["n_posts"] == 2
    assert m["s_at_k"] == 0.5
    assert abs(m["mrr"] - 0.1 / 2) < 1e-12  # MRR@10: rank 11 contributes 0


def test_mrr_hit_at_rank_2_is_half():
    gold = {"p1": {"g"}}
    m = evaluate({"p1": ["x", "g"]}, gold, k=10)
    assert m["mrr"] == 0.5
    assert m["s_at_k"] == 1.0


def test_mrr_no_hit_is_zero():
    gold = {"p1": {"g"}}
    m = evaluate({"p1": ["x", "y", "z"]}, gold, k=10)
    assert m["mrr"] == 0.0
    assert m["s_at_k"] == 0.0


def test_multiple_gold_first_hit_counts():
    gold = {"p1": {"a", "b"}}
    assert evaluate({"p1": ["x", "b", "a"]}, gold)["mrr"] == 0.5
    assert evaluate({"p1": ["a", "x", "b"]}, gold)["mrr"] == 1.0


def test_missing_prediction_counts_as_miss():
    gold = {"p1": {"g"}, "p2": {"g"}}
    m = evaluate({"p1": ["g"]}, gold, k=10)
    assert m["n_posts"] == 2
    assert m["s_at_k"] == 0.5
    assert m["mrr"] == 0.5


def test_predictions_for_non_gold_posts_are_ignored():
    gold = {"p1": {"g"}}
    m = evaluate({"p1": ["g"], "other": ["g"]}, gold, k=10)
    assert m["n_posts"] == 1
    assert m["s_at_k"] == 1.0


def test_string_and_int_ids_are_interchangeable():
    gold = {1: {10}}
    m = evaluate({"1": ["10"]}, gold, k=10)
    assert m["s_at_k"] == 1.0


def test_per_language_macro_differs_from_micro():
    # eng: 1 post, miss. spa: 3 posts, all hit.
    gold = {"e1": {"g"}, "s1": {"g"}, "s2": {"g"}, "s3": {"g"}}
    preds = {"e1": ["x"], "s1": ["g"], "s2": ["g"], "s3": ["g"]}
    langs = {"e1": "eng", "s1": "spa", "s2": "spa", "s3": "spa"}
    m = evaluate(preds, gold, langs, k=10)

    assert m["s_at_k"] == 0.75  # micro: 3/4 posts
    assert m["per_language"]["eng"] == {"s_at_k": 0.0, "mrr": 0.0, "n": 1}
    assert m["per_language"]["spa"]["s_at_k"] == 1.0
    assert m["per_language"]["spa"]["n"] == 3
    assert m["macro_avg_s_at_k"] == 0.5  # macro: mean over languages
    assert m["macro_avg_s_at_k"] != m["s_at_k"]


def test_missing_lang_goes_to_unk():
    gold = {"p1": {"g"}}
    m = evaluate({"p1": ["g"]}, gold, {}, k=10)
    assert list(m["per_language"]) == ["unk"]


def test_official_score_picks_macro_for_mono_and_micro_for_cross():
    gold = {"e1": {"g"}, "s1": {"g"}, "s2": {"g"}, "s3": {"g"}}
    preds = {"e1": ["x"], "s1": ["g"], "s2": ["g"], "s3": ["g"]}
    langs = {"e1": "eng", "s1": "spa", "s2": "spa", "s3": "spa"}
    m = evaluate(preds, gold, langs)
    assert official_score(m, "monolingual") == 0.5
    assert official_score(m, "crosslingual") == 0.75


def test_save_predictions_official_format(tmp_path):
    path = tmp_path / "p.json"
    save_predictions({7: [str(i) for i in range(150)]}, str(path))
    data = json.loads(path.read_text())
    assert list(data) == ["7"]
    assert data["7"] == list(range(100))  # top 100, int ids
