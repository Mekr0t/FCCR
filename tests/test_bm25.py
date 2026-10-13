import os

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from src.data import fc_text, load_dataset, post_text
from src.retrieval.bm25 import BM25Retriever, tokenize

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sample")


@pytest.fixture(scope="module")
def sample():
    return load_dataset(
        SAMPLE_DIR,
        posts_file="trial_posts.csv",
        fact_checks_file="trial_fact_checks.csv",
        mapping_file="trial_data_mapping.csv",
    )


@pytest.mark.parametrize("variant", ["eng", "orig"])
def test_top10_identical_to_rank_bm25(sample, variant):
    posts, fcs, _ = sample
    ids = list(fcs.index)
    texts = [fc_text(r, variant) for _, r in fcs.iterrows()]
    queries = [post_text(r, variant) for _, r in posts.iterrows()]

    ours = BM25Retriever().fit(ids, texts).search(queries, k=10)

    ref = BM25Okapi([tokenize(t) for t in texts])
    for q, got in zip(queries, ours):
        scores = np.asarray(ref.get_scores(tokenize(q)))
        order = np.lexsort((np.arange(len(ids)), -scores))[:10]  # ties by pool order
        assert got == [ids[i] for i in order]


def test_scores_match_rank_bm25_numerically(sample):
    _, fcs, _ = sample
    texts = [fc_text(r, "eng") for _, r in fcs.iterrows()]
    r = BM25Retriever().fit(list(range(len(texts))), texts)
    ref = BM25Okapi([tokenize(t) for t in texts])
    q = "the vaccine is a hoax the the video"
    ours = (r._query_matrix([q]) @ r.W).toarray()[0]
    assert np.allclose(ours, ref.get_scores(tokenize(q)), rtol=1e-9, atol=1e-12)


def test_always_returns_k_padded_in_pool_order():
    r = BM25Retriever().fit([10, 11, 12, 13], ["alpha beta", "beta gamma", "delta", "epsilon zeta"])
    out = r.search(["alpha", "nomatch", ""], k=3)
    assert out[0][0] == 10 and len(out[0]) == 3
    assert out[0][1:] == [11, 12]  # zero scores padded by pool order
    assert out[1] == [10, 11, 12] and out[2] == [10, 11, 12]
    assert r.search(["alpha"], k=100) == [[10, 11, 12, 13]]  # k capped at pool size


def test_tie_break_by_pool_order_beyond_partition_boundary():
    docs = ["same words"] * 6 + [f"filler{i}" for i in range(10)]
    r = BM25Retriever().fit(list(range(16)), docs)
    assert r.search(["same words"], k=4) == [[0, 1, 2, 3]]
