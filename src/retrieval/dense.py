"""Dense retrieval with a bi-encoder (default: intfloat/multilingual-e5-large).

- fp16 inference on GPU, L2-normalised vectors (cosine = dot product), e5 prefixes
  (`query: ` for posts, `passage: ` for fact-checks), max_seq_length 512.
- Texts are sorted by token length (longest first) and encoded in chunks, so padding is minimal
  and an out-of-memory error would show up immediately.
- Embeddings are cached on disk per (model, split, variant) in an `EmbeddingStore`:
  `emb.npy` (float16), `ids.npy`, `thash.npy` (hash of the exact prefixed text, detects changed
  texts), `ntok.npy` (token count before truncation) and `meta.json`. A store grows: asking for
  ids it lacks embeds only those (and saves after every chunk, so a crash loses little).
  Changing model, max_seq_length or prefix invalidates the store.
- `search` = matmul of normalised vectors (torch, GPU if available) + top-k, scores in float32.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import time

import numpy as np

DEFAULT_MODEL = "intfloat/multilingual-e5-large"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


def model_slug(model_name):
    return model_name.replace("/", "--")


def _hash(text):
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little", signed=True)


def seed_everything(seed=42):
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DenseEncoder:
    """Lazy wrapper around a SentenceTransformer in eval mode (fp16 on GPU)."""

    def __init__(self, model_name=DEFAULT_MODEL, max_seq_length=512, batch_size=32, device=None):
        self.model_name, self.max_seq_length, self.batch_size = model_name, int(max_seq_length), int(batch_size)
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            m = SentenceTransformer(self.model_name, device=self.device)
            if self.device == "cuda":
                m = m.half()
            m.max_seq_length = self.max_seq_length
            m.eval()
            self._model = m
        return self._model

    def token_lengths(self, texts):
        """Token count of each text (special tokens included) before truncation."""
        tok = self.model.tokenizer
        out = np.empty(len(texts), dtype=np.int32)
        for s in range(0, len(texts), 8192):
            enc = tok(texts[s : s + 8192], add_special_tokens=True, truncation=False)["input_ids"]
            out[s : s + len(enc)] = [len(x) for x in enc]
        return out

    def encode_sorted(self, texts, ntok, chunk=32768):
        """Yield (indices, float16 normalised embeddings) chunks, longest texts first."""
        import torch

        order = np.argsort(-ntok, kind="stable")
        for s in range(0, len(order), chunk):
            idx = order[s : s + chunk]
            with torch.inference_mode():
                e = self.model.encode(
                    [texts[i] for i in idx],
                    batch_size=self.batch_size,
                    convert_to_tensor=True,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                )
                e = torch.nn.functional.normalize(e.float(), dim=-1).half()
            yield idx, e.cpu().numpy()


class EmbeddingStore:
    """On-disk, growable cache of id -> embedding for one (model, split, variant, kind)."""

    def __init__(self, dir, model, max_seq_length, prefix):
        self.dir = dir
        self.key = {"model": model, "max_seq_length": int(max_seq_length), "prefix": prefix}
        self.ids = np.zeros(0, dtype=np.int64)
        self.emb = None
        self.thash = np.zeros(0, dtype=np.int64)
        self.ntok = np.zeros(0, dtype=np.int32)
        self._load()

    def _path(self, name):
        return os.path.join(self.dir, name)

    def _load(self):
        try:
            with open(self._path("meta.json"), encoding="utf-8") as f:
                meta = json.load(f)
            if any(meta.get(k) != v for k, v in self.key.items()):
                return
            ids = np.load(self._path("ids.npy"))
            emb = np.load(self._path("emb.npy"))
            thash, ntok = np.load(self._path("thash.npy")), np.load(self._path("ntok.npy"))
            if not (len(ids) == len(emb) == len(thash) == len(ntok) == meta["n"]):
                return
        except (OSError, ValueError, KeyError):
            return
        self.ids, self.emb, self.thash, self.ntok = ids, emb, thash, ntok

    def save(self, variant):
        os.makedirs(self.dir, exist_ok=True)
        # meta is removed first and written last, so a crash mid-save leaves an invalid (ignored) store
        meta_path = self._path("meta.json")
        if os.path.exists(meta_path):
            os.remove(meta_path)
        np.save(self._path("emb.npy"), self.emb)
        np.save(self._path("ids.npy"), self.ids)
        np.save(self._path("thash.npy"), self.thash)
        np.save(self._path("ntok.npy"), self.ntok)
        meta = {
            **self.key,
            "variant": variant,
            "n": int(len(self.ids)),
            "dim": int(self.emb.shape[1]),
            "dtype": str(self.emb.dtype),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _add(self, ids, hashes, ntok, emb):
        """Insert or overwrite rows."""
        ids = np.asarray(ids)
        pos = {int(i): r for r, i in enumerate(self.ids)}
        if self.emb is None:
            self.emb = np.zeros((0, emb.shape[1]), dtype=np.float16)
        new_rows = []
        for k, i in enumerate(ids):
            r = pos.get(int(i))
            if r is None:
                new_rows.append(k)
            else:
                self.emb[r], self.thash[r], self.ntok[r] = emb[k], hashes[k], ntok[k]
        if new_rows:
            self.ids = np.concatenate([self.ids, ids[new_rows]])
            self.emb = np.concatenate([self.emb, emb[new_rows]])
            self.thash = np.concatenate([self.thash, np.asarray(hashes)[new_rows]])
            self.ntok = np.concatenate([self.ntok, np.asarray(ntok)[new_rows]])

    def get(self, encoder, ids, texts, variant, log=print):
        """Embeddings (float16, rows aligned with `ids`) plus info about cache use and timing.

        `texts` must already carry the e5 prefix. Missing or changed entries are embedded now.
        """
        ids = np.asarray(list(ids), dtype=np.int64)
        texts = list(texts)
        hashes = np.fromiter((_hash(t) for t in texts), dtype=np.int64, count=len(texts))
        pos = {int(i): r for r, i in enumerate(self.ids)}
        todo = np.array(
            [k for k, i in enumerate(ids) if int(i) not in pos or self.thash[pos[int(i)]] != hashes[k]], dtype=np.int64
        )
        info = {"n": int(len(ids)), "n_embedded": int(len(todo)), "n_cached": int(len(ids) - len(todo)), "embed_s": 0.0}
        if len(todo):
            t0 = time.time()
            sub_texts = [texts[k] for k in todo]
            ntok = encoder.token_lengths(sub_texts)
            done = 0
            for idx, e in encoder.encode_sorted(sub_texts, ntok):
                self._add(ids[todo[idx]], hashes[todo[idx]], ntok[idx], e)
                done += len(idx)
                log(f"    embedded {done}/{len(todo)} ({time.time() - t0:.0f}s)")
                self.save(variant)
            info["embed_s"] = time.time() - t0
        pos = {int(i): r for r, i in enumerate(self.ids)}
        rows = np.array([pos[int(i)] for i in ids], dtype=np.int64)
        info["ntok"] = self.ntok[rows]
        return self.emb[rows], info


def truncation_stats(ntok, max_len):
    ntok = np.asarray(ntok)
    if not len(ntok):
        return {"n": 0}
    return {
        "n": int(len(ntok)),
        "mean_tokens": float(ntok.mean()),
        "p99_tokens": float(np.percentile(ntok, 99)),
        "max_tokens": int(ntok.max()),
        "n_truncated": int((ntok > max_len).sum()),
        "frac_truncated": float((ntok > max_len).mean()),
    }


def search(q_emb, pool_emb, pool_ids, top_n=100, chunk=512, device=None):
    """Top-`top_n` pool ids per query by dot product of normalised vectors.

    q_emb (nq x d), pool_emb (np x d), pool_ids length np. Returns list of lists of pool ids,
    best first. Scores are computed in float32; ties are broken by torch.topk (arbitrary but
    deterministic for fixed inputs).
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pool_ids = np.asarray(pool_ids)
    k = min(int(top_n), len(pool_ids))
    P = torch.from_numpy(np.ascontiguousarray(pool_emb)).to(device).float()
    out = []
    with torch.inference_mode():
        for s in range(0, len(q_emb), chunk):
            Q = torch.from_numpy(np.ascontiguousarray(q_emb[s : s + chunk])).to(device).float()
            top = torch.topk(Q @ P.T, k, dim=1).indices.cpu().numpy()
            out.extend(pool_ids[top].tolist())
    return out
