"""Unit tests for src/retrieval/combine.py on toy arrays (no models, no data)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.retrieval.combine import (  # noqa: E402
    chunk_windows,
    combine_scores,
    mean_normalise,
    ranks_desc,
    segment_max,
    subset_metrics,
)

A = torch.tensor([[0.9, 0.1, 0.5, 0.3], [0.2, 0.8, 0.4, 0.6]])
B = torch.tensor([[0.1, 0.7, 0.5, 0.2], [0.3, 0.9, 0.1, 0.5]])


def test_max_and_mean():
    assert torch.allclose(combine_scores([A, B], "max"), torch.maximum(A, B))
    assert torch.allclose(combine_scores([A, B], "mean"), (A + B) / 2)
    assert torch.allclose(combine_scores([A, B, A], "mean"), (2 * A + B) / 3)


def test_ranks_desc():
    assert ranks_desc(A).tolist() == [[0, 3, 1, 2], [3, 0, 2, 1]]


def test_rrf_math():
    out = combine_scores([A, B], "rrf", rrf_k=60)
    # row 0, column 0: rank 0 in A, rank 3 in B -> 1/61 + 1/64
    assert out[0, 0].item() == pytest.approx(1 / 61 + 1 / 64)
    # row 0, column 1: rank 3 in A, rank 0 in B -> same value, so a tie
    assert out[0, 1].item() == pytest.approx(out[0, 0].item())
    # column 2: rank 1 in both -> 2/62
    assert out[0, 2].item() == pytest.approx(2 / 62)
    # weights: only the second ranking counts
    w = combine_scores([A, B], "rrf", weights=[0.0, 1.0])
    assert torch.argmax(w[0]).item() == 1


def test_unknown_mode():
    with pytest.raises(ValueError):
        combine_scores([A], "sum")


def test_chunk_windows_short_is_single():
    assert chunk_windows(10, 450, 50) == [(0, 10)]
    assert chunk_windows(450, 450, 50) == [(0, 450)]
    assert chunk_windows(0, 450, 50) == [(0, 0)]


@pytest.mark.parametrize("n", [451, 500, 900, 901, 1234, 5000])
def test_chunk_windows_cover_and_overlap(n):
    size, ov = 450, 50
    w = chunk_windows(n, size, ov)
    assert all(e - s <= size for s, e in w)
    assert w[0][0] == 0 and w[-1][1] == n
    covered = np.zeros(n, dtype=bool)
    for s, e in w:
        covered[s:e] = True
    assert covered.all()
    for (s1, e1), (s2, e2) in zip(w, w[1:]):
        assert s2 < e1 and e1 - s2 >= ov  # consecutive windows overlap by at least `overlap`
        assert s2 > s1


def test_chunk_windows_bad_args():
    with pytest.raises(ValueError):
        chunk_windows(10, 5, 5)


def test_mean_normalise():
    e = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float16)
    m = mean_normalise(e)
    assert np.linalg.norm(m) == pytest.approx(1.0, abs=1e-6)
    assert m == pytest.approx(np.array([1, 1, 0]) / np.sqrt(2), abs=1e-3)


def test_segment_max():
    s = torch.tensor([[1.0, 5.0], [3.0, 2.0], [0.5, 9.0]])
    out = segment_max(s, torch.tensor([0, 0, 1]), 2)
    assert out.tolist() == [[3.0, 5.0], [0.5, 9.0]]


def test_subset_metrics():
    preds = {"1": [10, 11], "2": [20, 21], "3": [30, 31]}
    gold = {"1": {10}, "2": {21}, "3": {99}}
    langs = {"1": "a", "2": "a", "3": "b"}
    m = subset_metrics(preds, gold, langs, [1, 2], k=1)
    assert m["n_posts"] == 2 and m["s_at_k"] == pytest.approx(0.5)
    assert set(m["per_language"]) == {"a"}
