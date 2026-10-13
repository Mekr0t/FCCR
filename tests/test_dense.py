import os

import numpy as np
import pytest

from src.data import fc_text, gold_map, load_dataset, post_text
from src.evaluate import evaluate
from src.retrieval.dense import (
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    DenseEncoder,
    EmbeddingStore,
    search,
    truncation_stats,
)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sample")


class FakeEncoder:
    """Deterministic CPU stand-in: embedding = normalised bag of character codes."""

    def __init__(self):
        self.calls = 0

    def token_lengths(self, texts):
        return np.array([len(t.split()) for t in texts], dtype=np.int32)

    def encode_sorted(self, texts, ntok, chunk=2):
        order = np.argsort(-ntok, kind="stable")
        for s in range(0, len(order), chunk):
            idx = order[s : s + chunk]
            self.calls += len(idx)
            e = np.zeros((len(idx), 8), dtype=np.float32)
            for r, i in enumerate(idx):
                for ch in texts[i]:
                    e[r, ord(ch) % 8] += 1
            e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-9
            yield idx, e.astype(np.float16)


def test_store_caches_grows_and_detects_changes(tmp_path):
    enc = FakeEncoder()
    d = str(tmp_path / "s")
    store = EmbeddingStore(d, "m", 512, "passage: ")
    emb, info = store.get(enc, [5, 3, 9], ["a b", "c d e", "f"], "eng", log=lambda *_: None)
    assert info["n_embedded"] == 3 and emb.dtype == np.float16 and emb.shape == (3, 8)
    for f in ("emb.npy", "ids.npy", "meta.json"):
        assert os.path.isfile(os.path.join(d, f))

    # a fresh store object reads the cache: nothing is re-embedded, rows follow the requested order
    enc2 = FakeEncoder()
    store2 = EmbeddingStore(d, "m", 512, "passage: ")
    emb2, info2 = store2.get(enc2, [9, 5], ["f", "a b"], "eng", log=lambda *_: None)
    assert enc2.calls == 0 and info2["n_cached"] == 2
    assert np.array_equal(emb2[0], emb[2]) and np.array_equal(emb2[1], emb[0])

    # new id -> only that one is embedded; changed text -> re-embedded
    _, info3 = store2.get(enc2, [9, 7, 5], ["f", "zzz", "changed text"], "eng", log=lambda *_: None)
    assert info3["n_embedded"] == 2 and enc2.calls == 2

    # different max_seq_length or prefix invalidates the store
    assert len(EmbeddingStore(d, "m", 256, "passage: ").ids) == 0
    assert len(EmbeddingStore(d, "m", 512, "query: ").ids) == 0
    assert len(EmbeddingStore(d, "m", 512, "passage: ").ids) == 4


def test_search_orders_by_dot_product_and_truncation_stats():
    pool = np.eye(4, dtype=np.float16)
    q = np.array([[0.1, 0.9, 0, 0], [0, 0, 0.2, 0.8]], dtype=np.float16)
    out = search(q, pool, [10, 11, 12, 13], top_n=3, device="cpu")
    assert out[0][:2] == [11, 10] and out[1][:2] == [13, 12] and len(out[0]) == 3
    s = truncation_stats([10, 600, 512, 513], 512)
    assert s["n_truncated"] == 2 and s["max_tokens"] == 600


@pytest.fixture(scope="module")
def e5():
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    enc = DenseEncoder()
    try:
        enc.model  # noqa: B018 - loads the model
    except Exception as e:  # noqa: BLE001 - offline / not downloaded
        pytest.skip(f"mE5-large not loadable: {e!r}")
    return enc


def _embed(enc, texts):
    ntok = enc.token_lengths(texts)
    out = np.zeros((len(texts), 1024), dtype=np.float16)
    for idx, e in enc.encode_sorted(texts, ntok):
        out[idx] = e
    return out


def test_embeddings_normalised_deterministic_and_retrieve_gold_on_sample(e5):
    if not os.path.isdir(SAMPLE_DIR):
        pytest.skip("sample data missing")
    posts, fcs, mapping = load_dataset(
        SAMPLE_DIR,
        posts_file="trial_posts.csv",
        fact_checks_file="trial_fact_checks.csv",
        mapping_file="trial_data_mapping.csv",
    )
    gold = gold_map(mapping)
    pids = [p for p in sorted(gold) if p in posts.index]
    q_texts = [QUERY_PREFIX + post_text(posts.loc[p], "eng") for p in pids]
    fc_ids = list(fcs.index)
    fc_texts = [PASSAGE_PREFIX + fc_text(r, "eng") for _, r in fcs.iterrows()]

    q1, f1 = _embed(e5, q_texts), _embed(e5, fc_texts)
    norms = np.linalg.norm(np.concatenate([q1, f1]).astype(np.float32), axis=1)
    assert np.allclose(norms, 1.0, atol=2e-3)
    q2 = _embed(e5, q_texts)
    assert np.array_equal(q1, q2), "fp16 encoding is not deterministic"

    ranked = search(q1, f1, fc_ids, top_n=10)
    m = evaluate(dict(zip(pids, ranked)), {p: gold[p] for p in pids}, k=10)
    assert m["s_at_k"] >= 0.9, m["s_at_k"]
