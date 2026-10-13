import os

import pytest

from src.data import fc_segments, load_dataset, post_segments
from src.retrieval.bm25 import Analyzer, BM25Retriever, tokenize

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sample")


def test_default_is_plain_tokenizer():
    assert Analyzer.from_config(None) is None
    assert Analyzer.from_config({"stem": False, "stopwords": False}) is None
    r = BM25Retriever().fit([1, 2], ["Vaccines work", "other text"])
    assert r._tok is tokenize


def test_analyzer_with_no_features_matches_default_tokens():
    a = Analyzer()
    for text in ["The Vaccines, are 5G!", "ภาษาไทย test"]:
        assert a(text) == tokenize(text)


def test_english_stemming():
    a = Analyzer(stem=True)
    toks = a("vaccines vaccinated vaccination")
    assert toks[0] == toks[1] == toks[2]
    assert a([("Vaccines", "eng")]) == a([("vaccinated", "eng")])
    assert a([("vaccines", "pol")]) == ["vaccines"]  # no stemmer for Polish


def test_stopwords_removed_per_language():
    a = Analyzer(stopwords=True)
    assert a([("the vaccine is a hoax", "eng")]) == ["vaccine", "hoax"]
    assert "der" not in a([("der Impfstoff", "deu")])
    assert a([("the vaccine", "unk")]) == ["the", "vaccine"]  # unknown language: untouched


def test_thai_split_into_words():
    text = "วัคซีนโควิดปลอดภัยหรือไม่"
    th = Analyzer(thai_tokenizer=True)(text)
    assert len(th) > 1
    assert "".join(th) == text  # whole words, no marks lost (the default \w+ regex shreds Thai)
    assert "".join(tokenize(text)) != text
    assert Analyzer(thai_tokenizer=False)(text) == tokenize(text)
    assert Analyzer(thai_tokenizer=True)("plain english") == ["plain", "english"]


def test_unknown_option_rejected():
    with pytest.raises(ValueError):
        Analyzer.from_config({"stem": True, "bogus": True})


def test_segments_variants():
    t = ("hola mundo", "hello world", [("spa", 0.9)])
    row = {"text": t, "ocr": [], "claim": t, "title": ""}
    assert post_segments(row, "eng") == [("hello world", "eng")]
    assert post_segments(row, "orig") == [("hola mundo", "spa")]
    assert post_segments(row, "orig", lang="por") == [("hola mundo", "por")]
    assert fc_segments(row, "both") == [("hola mundo", "spa"), ("hello world", "eng")]
    same = ("hi", "hi", [("eng", 1.0)])
    assert fc_segments({"claim": same, "title": ""}, "both") == [("hi", "eng")]


def test_retriever_with_segments():
    a = Analyzer(stem=True, stopwords=True)
    r = BM25Retriever(analyzer=a).fit([1, 2], [[("the vaccines", "eng")], [("masks", "eng")]])
    assert r.search([[("vaccinated", "eng")]], k=2)[0][0] == 1


def test_top_lang_accepts_flat_pair_and_list():
    from src.data import _top_lang

    assert _top_lang(("a", "b", [("spa", 0.6), ("eng", 0.2)])) == "spa"
    assert _top_lang(("a", "b", ["eng", 1.0])) == "eng"  # test_set format
