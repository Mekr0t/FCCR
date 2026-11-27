r"""Write configs/splits/dev_error_analysis_split.json (post ids only, safe to commit).

Usage (from the project root):
    .venv\Scripts\python.exe scripts\make_dev_split.py [--split-dir DIR] [--seed N] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.analysis_split import DEFAULT_SEED, split_dev_posts  # noqa: E402

DEFAULT_DIR = os.path.join(ROOT, "data", "raw", "semeval2025_task7_data", "train_dev_sets")
DEFAULT_OUT = os.path.join(ROOT, "configs", "splits", "dev_error_analysis_split.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-dir", default=DEFAULT_DIR)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    if not os.path.isdir(args.split_dir):
        sys.exit(f"split dir not found: {args.split_dir}")

    res = split_dev_posts(args.split_dir, seed=args.seed)
    out = {
        "seed": args.seed,
        "created": "2026-11-27",
        "source": "train_dev_sets tasks.json posts_dev, both tracks",
        "analysis": res["analysis"],
        "heldout": res["heldout"],
        "counts": res["counts"],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")

    print(f"{'stratum':<12}{'analysis':>10}{'heldout':>10}")
    for s, c in res["counts"].items():
        print(f"{s:<12}{c['analysis']:>10}{c['heldout']:>10}")
    print(f"{'total':<12}{len(res['analysis']):>10}{len(res['heldout']):>10}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
