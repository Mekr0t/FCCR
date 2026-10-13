r"""Score a prediction file two ways and check they agree.

1. src/evaluate.py (our evaluator), and
2. an independent copy of the official scoring logic (calculate_success_at_k from the
   organizers' monolingual_scoring.py / crosslingual_scoring.py, including its asserts).

Usage:
    .venv\Scripts\python.exe scripts\score_official.py --track monolingual \
        --split-dir data\raw\semeval2025_task7_data\test_set --pred path\to\pred.json

The split dir is only read as data (tasks.json and <track>_reference.json); nothing in it is
executed or imported. Exit code 1 if the two scorers disagree.
"""

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from src.data import eval_units, load_reference, split_kind  # noqa: E402
from src.evaluate import evaluate, official_score  # noqa: E402


# ---- independent copy of the official logic ------------------------------------------------
def official_post_id_to_lang(tasks, split):
    monolingual = tasks["monolingual"]
    out = {}
    for k, v in monolingual.items():
        if split == "train_dev":
            posts = list(v["posts_train"]) + list(v["posts_dev"])
        elif split == "test":
            posts = v["posts_test"]
        else:
            raise Exception("invalid split")
        for p in posts:
            if str(p) in out:  # (official code tested `p not in dict` with int p vs str keys, i.e. never fired)
                raise Exception("you have duplicated post ids")
            out[str(p)] = k
    return out


def official_mono(truth, prediction, k, post_id_to_lang):
    """calculate_success_at_k from monolingual_scoring.py."""
    assert len(truth) == len(prediction), "The number of prediction is not correct"
    success_by_lang = {}
    for key, t in truth.items():
        if key not in prediction:
            raise Exception("Missing post id: " + str(key))
        key_lang = post_id_to_lang[key]
        if key_lang not in success_by_lang:
            success_by_lang[key_lang] = []
        p = prediction[key]
        assert len(p) >= 10, "You should provide top 10 predictions for each example"
        if len(t) == 1:
            success_by_lang[key_lang].append(1 if t[0] in p[:k] else 0)
        elif len(t) > 1:
            flag = 0
            for t_ in t:
                if t_ in p[:k]:
                    flag += 1
                    break
            success_by_lang[key_lang].append(1 if flag > 0 else 0)
        else:
            raise Exception("Some examples have gold label less than 1")
    average = {lang: sum(s) / len(s) for lang, s in success_by_lang.items()}
    average["avg"] = np.mean(list(average.values()))
    return average


def official_cross(truth, prediction, k):
    """calculate_success_at_k from crosslingual_scoring.py."""
    assert len(truth) == len(prediction), "The number of prediction is not correct"
    success_at_k = 0
    for key, t in truth.items():
        if key not in prediction:
            raise Exception("Missing post id: " + str(key))
        p = prediction[key]
        assert len(p) >= 10, "You should provide top 10 predictions for each example"
        if len(t) == 1:
            if t[0] in p[:k]:
                success_at_k += 1
        elif len(t) > 1:
            for t_ in t:
                if t_ in p[:k]:
                    success_at_k += 1
                    break
        else:
            raise Exception("Some examples have gold label less than 1")
    return success_at_k / len(truth)


# ---- driver ---------------------------------------------------------------------------------
def score(track, split_dir, pred_path):
    """Return (ours, official): each {'avg': float, 'per_language': {...} or None}, plus our mrr."""
    kind = split_kind(split_dir)
    with open(pred_path, encoding="utf-8") as f:
        pred = json.load(f)
    truth = load_reference(split_dir, track)

    units = eval_units(split_dir, track, kind)
    post_langs = {str(p): u["lang"] for u in units for p in u["post_ids"]}
    m = evaluate(pred, {p: set(fs) for p, fs in truth.items()}, post_langs, k=10)
    ours = {
        "avg": official_score(m, track),
        "per_language": {l: v["s_at_k"] for l, v in m["per_language"].items()} if track == "monolingual" else None,
        "mrr_at_10": m["macro_mrr"] if track == "monolingual" else m["mrr"],
        "n_posts": m["n_posts"],
    }

    if track == "monolingual":
        with open(os.path.join(split_dir, "tasks.json"), encoding="utf-8") as f:
            tasks = json.load(f)
        res = official_mono(truth, pred, 10, official_post_id_to_lang(tasks, "test" if kind == "test" else "train_dev"))
        off = {"avg": float(res.pop("avg")), "per_language": res}
    else:
        off = {"avg": float(official_cross(truth, pred, 10)), "per_language": None}
    return ours, off


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True, choices=["monolingual", "crosslingual"])
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()

    ours, off = score(args.track, args.split_dir, args.pred)
    print(f"pred:  {args.pred}\ntrack: {args.track}   split dir: {args.split_dir}   posts: {ours['n_posts']}")
    print(f"ours     S@10 = {ours['avg']:.6f}   (MRR@10 = {ours['mrr_at_10']:.4f})")
    print(f"official S@10 = {off['avg']:.6f}")
    if ours["per_language"]:
        print("per language (ours / official):")
        for lang in sorted(off["per_language"]):
            print(f"  {lang}: {ours['per_language'][lang]:.4f} / {off['per_language'][lang]:.4f}")
    agree = abs(ours["avg"] - off["avg"]) < 1e-9 and (
        ours["per_language"] is None
        or all(abs(ours["per_language"][l] - v) < 1e-9 for l, v in off["per_language"].items())
    )
    print("AGREE" if agree else "MISMATCH")
    sys.exit(0 if agree else 1)


if __name__ == "__main__":
    main()
