"""Pinned BM25 results on the public sample data.

These numbers are the regression anchor for the BM25 text construction and tokenizer. If you
change either on purpose, the dev numbers will shift: re-run the official configs and update
the pins here (and the table in results/results.md) in the same change.

Uses the runner's sample path in memory (run_sample does not write anything), so nothing under
results/ is touched.
"""

import os
import sys

import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import run_bm25  # noqa: E402
from src.evaluate import evaluate  # noqa: E402

SAMPLE_DIR = os.path.join(ROOT, "data", "raw", "sample")
MSG = (
    "BM25 sample result changed: text construction or tokenization changed, so the dev numbers "
    "will shift. Re-run the official configs and update these pins deliberately."
)

# (config, text_variant, S@10, MRR@10)
PINS = [
    ("configs/bm25_sample_eng.yaml", "eng", 0.9574, 0.7794),
    ("configs/bm25_sample_orig.yaml", "orig", 0.7021, 0.5497),
]


@pytest.mark.skipif(not os.path.isdir(SAMPLE_DIR), reason="public sample data not present (data/raw/sample)")
@pytest.mark.parametrize("config_path,variant,s_pin,mrr_pin", PINS, ids=["eng", "orig"])
def test_bm25_sample_pinned(config_path, variant, s_pin, mrr_pin):
    with open(os.path.join(ROOT, config_path), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "track" not in cfg, "pinned test is for the sample path"
    assert cfg["text_variant"] == variant

    preds, gold, post_langs, _, _ = run_bm25.run_sample(cfg, variant, int(cfg.get("top_n", 100)), None)
    m = evaluate(preds, gold, post_langs, k=int(cfg.get("k", 10)))

    assert abs(m["s_at_k"] - s_pin) < 1e-4, f"{MSG} [{variant}] S@10 {m['s_at_k']:.4f} != {s_pin}"
    assert abs(m["mrr"] - mrr_pin) < 1e-4, f"{MSG} [{variant}] MRR@10 {m['mrr']:.4f} != {mrr_pin}"
