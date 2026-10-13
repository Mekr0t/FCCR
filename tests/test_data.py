import os

import pytest

from src.data import VARIANTS, fc_text, gold_map, load_dataset, post_language, post_text

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sample")


@pytest.fixture(scope="module")
def sample():
    posts, fcs, mapping = load_dataset(
        SAMPLE_DIR,
        posts_file="trial_posts.csv",
        fact_checks_file="trial_fact_checks.csv",
        mapping_file="trial_data_mapping.csv",
    )
    return posts, fcs, mapping


def test_sample_counts(sample):
    posts, fcs, mapping = sample
    assert len(posts) == 47
    assert len(fcs) == 50
    assert len(mapping) == 50


def test_post_text_non_empty_for_every_mapped_post(sample):
    posts, _, mapping = sample
    for variant in VARIANTS:
        for pid in set(mapping["post_id"]):
            assert post_text(posts.loc[pid], variant).strip(), (pid, variant)


def test_fc_text_non_empty(sample):
    _, fcs, _ = sample
    for variant in VARIANTS:
        for _, row in fcs.iterrows():
            assert fc_text(row, variant).strip()


def test_newline_text_parses(sample):
    posts, _, _ = sample
    found = False
    for _, row in posts.iterrows():
        for t in ([row["text"]] if row["text"] else []) + list(row["ocr"] or []):
            if any("\n" in s for s in t[:2] if isinstance(s, str)):
                found = True
    assert found, "expected at least one parsed text containing a newline"


def test_gold_map(sample):
    _, _, mapping = sample
    gold = gold_map(mapping)
    assert sum(len(v) for v in gold.values()) == 50
    assert all(isinstance(k, int) for k in gold)


def test_post_text_both_skips_identical_english():
    row = {"text": ("hola", "hola", [("spa", 1.0)]), "ocr": []}
    assert post_text(row, "both") == "hola"
    assert post_text(row, "orig") == "hola"
    row2 = {"text": ("hola", "hello", [("spa", 1.0)]), "ocr": [("ocr", "ocr en", [("eng", 1.0)])]}
    assert post_text(row2, "both") == "hola hello ocr ocr en"
    assert post_text(row2, "eng") == "hello ocr en"


def test_post_text_handles_empty_text_and_ocr_only():
    row = {"text": "", "ocr": [("uno", "one", [("spa", 0.9)])]}
    assert post_text(row, "eng") == "one"
    assert post_text({"text": "", "ocr": []}, "eng") == ""


def test_fc_text_empty_title():
    row = {"claim": ("a", "A", [("spa", 1.0)]), "title": ""}
    assert fc_text(row, "eng") == "A"


def test_post_language_fallback_to_ocr():
    row = {"text": "", "ocr": [("x", "x", [("tha", 0.5), ("eng", 0.9)])]}
    assert post_language(row) == "eng"
    assert post_language({"text": "", "ocr": []}) == "unk"
    row2 = {"text": ("a", "a", [("kor", 0.2), ("spa", 0.8)]), "ocr": []}
    assert post_language(row2) == "spa"


def test_invalid_variant_raises():
    with pytest.raises(ValueError):
        post_text({"text": "", "ocr": []}, "bogus")
