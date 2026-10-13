r"""BM25 retriever over fact-check texts, scipy.sparse implementation of rank_bm25's BM25Okapi.

Scoring matches BM25Okapi (k1=1.5, b=0.75, epsilon=0.25):
  idf(t)   = log(N - n_t + 0.5) - log(n_t + 0.5); idf < 0 is replaced by epsilon * mean(idf)
             (the mean runs over the raw idf of the whole vocabulary, like rank_bm25)
  score    = sum over query tokens (repeats count repeatedly) of
             idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
Tokenization: lowercase, then regex `\w+` (Unicode-aware). No stemming or stopwords.
The pool is whatever is passed to `fit`, so pool restriction is done by fitting on a subset.

The doc-term weights are precomputed once into a (vocab x docs) CSR matrix, so a batch of
queries is one sparse matmul. `search` always returns exactly min(k, pool size) ids: docs
are ordered by (-score, pool order), so zero-score docs pad the list deterministically.
"""

from __future__ import annotations

import re

import numpy as np
import scipy.sparse as sp

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(self, k1=1.5, b=0.75, epsilon=0.25, chunk=256):
        self.k1, self.b, self.epsilon, self.chunk = k1, b, epsilon, chunk
        self.ids = []
        self.vocab = None
        self.W = None  # (vocab x docs) csr, float64

    def fit(self, ids, texts):
        ids = list(ids)
        texts = list(texts)
        if len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length")
        if not ids:
            raise ValueError("cannot fit BM25 on an empty pool")
        self.ids = ids
        n_docs = len(ids)

        vocab = {}
        rows, cols = [], []  # term index, doc index (one entry per token occurrence)
        doc_len = np.zeros(n_docs, dtype=np.float64)
        for d, text in enumerate(texts):
            toks = tokenize(text)
            doc_len[d] = len(toks)
            for t in toks:
                i = vocab.get(t)
                if i is None:
                    i = vocab[t] = len(vocab)
                rows.append(i)
                cols.append(d)
        self.vocab = vocab
        n_terms = len(vocab)
        if n_terms == 0:
            self.W = sp.csr_matrix((0, n_docs))
            return self

        tf = sp.coo_matrix(
            (np.ones(len(rows), dtype=np.float64), (np.asarray(rows), np.asarray(cols))),
            shape=(n_terms, n_docs),
        ).tocsr()  # duplicates are summed -> term frequencies
        tf.sum_duplicates()

        df = np.diff(tf.indptr).astype(np.float64)  # docs containing each term
        idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
        eps = self.epsilon * idf.sum() / n_terms
        idf = np.where(idf < 0, eps, idf)

        avgdl = doc_len.sum() / n_docs
        # BM25Okapi has no guard for avgdl == 0 (all docs empty); avoid dividing by zero.
        norm = self.k1 * (1 - self.b + self.b * doc_len / avgdl) if avgdl > 0 else np.full(n_docs, self.k1 * (1 - self.b))
        data = tf.data
        term_of_entry = np.repeat(np.arange(n_terms), np.diff(tf.indptr))
        w = idf[term_of_entry] * data * (self.k1 + 1) / (data + norm[tf.indices])
        self.W = sp.csr_matrix((w, tf.indices, tf.indptr), shape=(n_terms, n_docs))
        return self

    def _query_matrix(self, query_texts):
        indptr, indices, data = [0], [], []
        for q in query_texts:
            counts = {}
            for t in tokenize(q):
                i = self.vocab.get(t)
                if i is not None:
                    counts[i] = counts.get(i, 0) + 1
            indices.extend(counts)
            data.extend(counts.values())
            indptr.append(len(indices))
        return sp.csr_matrix(
            (np.asarray(data, dtype=np.float64), np.asarray(indices, dtype=np.int64), np.asarray(indptr)),
            shape=(len(indptr) - 1, self.W.shape[0]),
        )

    @staticmethod
    def _top_k(scores, k):
        """Indices of the k best docs ordered by (-score, index); exact under ties."""
        n = scores.shape[0]
        if k >= n:
            return np.lexsort((np.arange(n), -scores))
        thresh = np.partition(scores, n - k)[n - k]  # k-th largest value
        above = np.flatnonzero(scores > thresh)
        above = above[np.lexsort((above, -scores[above]))]
        ties = np.flatnonzero(scores == thresh)[: k - len(above)]  # lowest doc index first
        return np.concatenate([above, ties])

    def search(self, query_texts, k=10):
        """For each query, return exactly min(k, pool size) pool ids ranked by BM25 score.

        Ties (including the all-zero padding) are broken by pool order, so results are
        deterministic.
        """
        if self.W is None:
            raise RuntimeError("call fit() before search()")
        query_texts = list(query_texts)
        k = min(int(k), len(self.ids))
        if k <= 0:
            return [[] for _ in query_texts]
        results = []
        for start in range(0, len(query_texts), self.chunk):
            Q = self._query_matrix(query_texts[start : start + self.chunk])
            S = (Q @ self.W).toarray() if self.W.shape[0] else np.zeros((Q.shape[0], len(self.ids)))
            for row in S:
                results.append([self.ids[i] for i in self._top_k(row, k)])
        return results
