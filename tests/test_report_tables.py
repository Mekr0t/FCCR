import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import make_report_tables as mrt  # noqa: E402


def _write_run(results_dir, name, split, track, s10, mrr, per_language=None, model="bm25", variant="eng"):
    d = results_dir / name
    d.mkdir(parents=True)
    inner = {"k": 10, "s_at_k": s10, "mrr": mrr, "n_posts": 10}
    if per_language is not None:
        inner["per_language"] = {l: {"s_at_k": v, "mrr": v / 2, "n": 5} for l, v in per_language.items()}
    meta = {
        "run_name": name,
        "model": model,
        "text_variant": variant,
        "pool": "per-language" if track == "monolingual" else "all",
        "split": split,
        "track": track,
        "headline_s_at_10": s10,
        "headline_mrr_at_10": mrr,
        "metrics": inner,
    }
    (d / "metrics.json").write_text(json.dumps(meta), encoding="utf-8")


def test_bold_best_marks_all_ties_and_ignores_none():
    assert mrt.bold_best([0.5, 0.7, 0.7, None]) == [False, True, True, False]


def test_bold_best_lower_is_better_and_all_none():
    assert mrt.bold_best([0.5, 0.2, None], higher_is_better=False) == [False, True, False]
    assert mrt.bold_best([None, None]) == [False, False]


def test_bold_best_compares_after_rounding_to_three_decimals():
    # 0.8861 and 0.8864 both show as 0.886 -> both bold; 0.8849 is smaller.
    assert mrt.bold_best([0.8861, 0.8864, 0.8849]) == [True, True, False]


def test_build_report_bolds_best_per_track_and_skips_missing(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    # Test runs: ours mono (two present, one missing), ours cross.
    _write_run(res, "bm25_mono_test_eng", "test", "monolingual", 0.770, 0.62, {"eng": 0.7, "ara": 0.8})
    _write_run(res, "dense_mono_test_both", "test", "monolingual", 0.886, 0.72, {"eng": 0.81, "ara": 0.94}, model="mE5-large")
    _write_run(res, "bm25_cross_test_eng", "test", "crosslingual", 0.537, 0.34)
    _write_run(res, "dense_cross_test_eng", "test", "crosslingual", 0.681, 0.41, model="mE5-large")
    # Dev run must appear in Table C only.
    _write_run(res, "bm25_mono_dev_eng", "dev", "monolingual", 0.685, 0.49, {"eng": 0.55, "ara": 0.7})

    organizers = {
        "monolingual": {
            "Org A": {"s10": 0.900, "mrr": 0.6, "per_language": {"eng": 0.9, "ara": 0.9}, "published_s10": 0.9},
        },
        "crosslingual": {
            "Org A": {"s10": 0.600, "mrr": 0.3, "per_language": {}, "published_s10": 0.6},
        },
    }
    text = mrt.build_report(str(res), organizers)

    mono_part, rest = text.split("### Crosslingual", 1)
    cross_part = rest.split("## Table B", 1)[0]
    # Mono: organizer 0.900 beats ours (0.770, 0.886, missing) -> only organizer bold in S@10.
    assert "| Org A | **0.900** |" in mono_part
    assert "| mE5-large (ours, both) | 0.886 |" in mono_part  # not bold: lower than organizer
    assert "| BM25 (ours, eng) | 0.770 |" in mono_part
    # Missing tuned BM25 run is shown as dash.
    assert "| BM25 tuned (ours, both + stem + stop + Thai) | – | – |" in mono_part

    # Cross: ours mE5 0.681 is the best in its track -> bold, organizer 0.600 is not.
    assert "| mE5-large (ours, eng) | **0.681** |" in cross_part
    assert "| Org A | 0.600 |" in cross_part

    # Table C contains the dev run and no test runs.
    table_c = text.split("## Table C", 1)[1]
    assert "bm25_mono_dev_eng" in table_c
    assert "bm25_mono_test_eng" not in table_c
