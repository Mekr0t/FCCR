"""Checks on saved official-run predictions, against the local official dataset.

Needs data/raw/semeval2025_task7_data (marker `needs_data`). Without it every test here
skips, which is the normal state in CI. Reads only: no retrieval, no model, nothing is written.
Set SEMEVAL_OFFICIAL_DIR to point the tests elsewhere (used to simulate a missing dataset).
"""

import glob
import json
import os

import pytest

from src.checks import assert_predictions_in_pool
from src.data import eval_units, load_reference

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OFFICIAL_DIR = os.environ.get("SEMEVAL_OFFICIAL_DIR") or os.path.join(ROOT, "data", "raw", "semeval2025_task7_data")
SPLIT_DIRS = {"dev": "train_dev_sets", "test": "test_set"}

pytestmark = pytest.mark.needs_data


def _data_available():
    return os.path.isdir(OFFICIAL_DIR)


requires_data = pytest.mark.skipif(
    not _data_available(), reason=f"official dataset not found at {OFFICIAL_DIR} (needs_data)"
)


def _official_runs():
    """(run dir name, split dir, track, eval_split) for results/* with predictions.json and an official track."""
    runs = []
    for mpath in sorted(glob.glob(os.path.join(ROOT, "results", "*", "metrics.json"))):
        run_dir = os.path.dirname(mpath)
        if not os.path.isfile(os.path.join(run_dir, "predictions.json")):
            continue
        with open(mpath, encoding="utf-8") as f:
            meta = json.load(f)
        track, split = meta.get("track"), meta.get("split")
        if track not in ("monolingual", "crosslingual") or split not in SPLIT_DIRS:
            continue  # sample runs and anything without an official track
        runs.append((os.path.basename(run_dir), os.path.join(OFFICIAL_DIR, SPLIT_DIRS[split]), track, split))
    return runs


_RUNS = _official_runs()
_PARAMS = [pytest.param(r, id=r[0]) for r in _RUNS] or [
    pytest.param(None, id="no-runs", marks=pytest.mark.skip(reason="no official results/*/predictions.json"))
]


@requires_data
@pytest.mark.parametrize("run", _PARAMS)
def test_predictions_inside_pools_and_post_set_matches_reference(run):
    name, split_dir, track, split = run
    with open(os.path.join(ROOT, "results", name, "predictions.json"), encoding="utf-8") as f:
        preds = json.load(f)
    units = eval_units(split_dir, track, split)
    assert_predictions_in_pool(preds, units)

    ref_posts = {str(p) for p in load_reference(split_dir, track)}
    assert {str(p) for p in preds} == ref_posts, f"{name}: predicted post set differs from the reference"
