import json

from src.analysis_split import split_dev_posts

MONO = {"eng": [1, 2, 3, 4, 5, 6, 7], "deu": [10, 11, 12, 13], "fra": [20]}
CROSS = [3, 4, 30, 31, 32, 33, 34, 35]  # 3 and 4 are also in the mono dev set


def _write(tmp_path):
    tasks = {
        "monolingual": {l: {"fact_checks": [100], "posts_train": [], "posts_dev": ids} for l, ids in MONO.items()},
        "crosslingual": {"fact_checks": [100, 101], "posts_train": [], "posts_dev": CROSS},
    }
    (tmp_path / "tasks.json").write_text(json.dumps(tasks))
    mono_ref = {str(p): [100] for ids in MONO.values() for p in ids}
    (tmp_path / "monolingual_reference.json").write_text(json.dumps(mono_ref))
    (tmp_path / "crosslingual_reference.json").write_text(json.dumps({str(p): [100] for p in CROSS}))
    return str(tmp_path)


def _all_posts():
    return {p for ids in MONO.values() for p in ids} | set(CROSS)


def test_disjoint_and_complete(tmp_path):
    r = split_dev_posts(_write(tmp_path))
    a, h = set(r["analysis"]), set(r["heldout"])
    assert not a & h
    assert a | h == _all_posts()
    assert len(r["analysis"]) + len(r["heldout"]) == len(_all_posts())
    assert r["analysis"] == sorted(r["analysis"]) and r["heldout"] == sorted(r["heldout"])


def test_shared_post_gets_one_half(tmp_path):
    r = split_dev_posts(_write(tmp_path))
    # 3 and 4 appear in both tracks: exactly one half, counted once, stratum = mono language
    for p in (3, 4):
        assert (p in r["analysis"]) != (p in r["heldout"])
    assert sum(c["analysis"] + c["heldout"] for c in r["counts"].values()) == len(_all_posts())
    assert "cross_only" in r["counts"]
    assert r["counts"]["cross_only"]["analysis"] + r["counts"]["cross_only"]["heldout"] == 6


def test_deterministic(tmp_path):
    d = _write(tmp_path)
    assert split_dev_posts(d) == split_dev_posts(d)


def test_stratum_balance(tmp_path):
    r = split_dev_posts(_write(tmp_path))
    for s, c in r["counts"].items():
        assert 0 <= c["analysis"] - c["heldout"] <= 1, s
    assert r["counts"]["fra"] == {"analysis": 1, "heldout": 0}


def test_seed_changes_assignment(tmp_path):
    d = _write(tmp_path)
    base = split_dev_posts(d, seed=1)["analysis"]
    assert any(split_dev_posts(d, seed=s)["analysis"] != base for s in range(2, 12))
