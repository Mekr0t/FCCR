"""Loading, parsing and text construction for SemEval-2025 Task 7.

Parsing follows the official load.py (`parse_col`): newlines are escaped before
`ast.literal_eval`, because literal_eval chokes on raw newlines inside string literals.

Translated-text tuples look like `(original, english, [(lang, confidence), ...])`.
Tuple index 0 is the original text, index 1 the English translation.
"""

from __future__ import annotations

import ast
import os

import pandas as pd

VARIANTS = ("orig", "eng", "both")


def _parse_col(s):
    """Official parsing logic from load.py. Empty strings are returned unchanged."""
    return ast.literal_eval(s.replace("\n", "\\n")) if s else s


def _py(x):
    """Convert numpy/pandas scalars to plain Python scalars (for dict keys)."""
    return x.item() if hasattr(x, "item") else x


def load_dataset(
    dir,
    posts_file="posts.csv",
    fact_checks_file="fact_checks.csv",
    mapping_file="fact_check_post_mapping.csv",
):
    """Load posts, fact-checks and the post/fact-check mapping.

    Returns (posts_df, fcs_df, mapping_df). posts_df is indexed by post_id and
    fcs_df by fact_check_id; text-like columns are parsed into Python objects.
    """
    posts_path = os.path.join(dir, posts_file)
    fcs_path = os.path.join(dir, fact_checks_file)
    mapping_path = os.path.join(dir, mapping_file)
    for p in (posts_path, fcs_path, mapping_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    fcs_df = pd.read_csv(fcs_path).fillna("").set_index("fact_check_id")
    for col in ["claim", "instances", "title"]:
        fcs_df[col] = fcs_df[col].apply(_parse_col)

    posts_df = pd.read_csv(posts_path).fillna("").set_index("post_id")
    for col in ["instances", "ocr", "verdicts", "text"]:
        posts_df[col] = posts_df[col].apply(_parse_col)

    mapping_df = pd.read_csv(mapping_path)
    return posts_df, fcs_df, mapping_df


def _check_variant(variant):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")


def _select(t, variant):
    """Pick the text of one translated-text tuple for a variant.

    Missing pieces fall back to the other language: for "orig" an empty original
    uses the English text and vice versa, so a non-empty tuple never yields "".
    "both" joins original and English, skipping English when it equals the original.
    """
    _check_variant(variant)
    if not t:
        return ""
    orig = (t[0] if len(t) > 0 and t[0] else "") or ""
    eng = (t[1] if len(t) > 1 and t[1] else "") or ""
    if variant == "orig":
        return orig or eng
    if variant == "eng":
        return eng or orig
    parts = [orig]
    if eng and eng != orig:
        parts.append(eng)
    return " ".join(p for p in parts if p)


def post_text(row, variant):
    """Post text: `text` tuple plus every `ocr` tuple, joined by spaces. May be ''."""
    _check_variant(variant)
    pieces = [_select(row["text"], variant)]
    ocr = row["ocr"] or []
    pieces.extend(_select(t, variant) for t in ocr)
    return " ".join(p for p in pieces if p)


def fc_text(row, variant):
    """Fact-check text: `claim` plus `title` (title may be empty). May be ''."""
    _check_variant(variant)
    pieces = [_select(row["claim"], variant), _select(row["title"], variant)]
    return " ".join(p for p in pieces if p)


def _top_lang(t):
    """Language code with the highest confidence in a translated-text tuple, or None."""
    if not t or len(t) < 3 or not t[2]:
        return None
    return max(t[2], key=lambda lc: lc[1])[0]


def post_language(row):
    """Top language of the post text tuple, falling back to the first OCR tuple; 'unk' if none."""
    lang = _top_lang(row["text"])
    if lang is None:
        ocr = row["ocr"] or []
        if ocr:
            lang = _top_lang(ocr[0])
    return lang or "unk"


def gold_map(mapping_df):
    """Mapping dataframe -> {post_id: set(fact_check_id)}."""
    gold = {}
    for pid, fcid in zip(mapping_df["post_id"], mapping_df["fact_check_id"]):
        gold.setdefault(_py(pid), set()).add(_py(fcid))
    return gold


# ---------------------------------------------------------------------------
# Official release (Zenodo 10.5281/zenodo.14989176)
# ---------------------------------------------------------------------------

import json  # noqa: E402
import pickle  # noqa: E402

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(_ROOT, "data", "processed")
TRACKS = ("monolingual", "crosslingual")
EVAL_SPLITS = ("dev", "test")
_POST_COLS = ["instances", "ocr", "verdicts", "text"]
_FC_COLS = ["claim", "instances", "title"]
_CACHE_VERSION = 1


def _parse_col_safe(s):
    """_parse_col with a fallback that also escapes carriage returns."""
    try:
        return _parse_col(s)
    except (ValueError, SyntaxError):
        return ast.literal_eval(s.replace("\r", "\r").replace("\n", "\n"))


def _parse_frame(df, cols, id_col, failures, table):
    """Parse `cols` row by row; unparsable cells become '' and are recorded in `failures`."""
    for col in cols:
        out = []
        for rid, s in zip(df[id_col], df[col]):
            try:
                out.append(_parse_col_safe(s))
            except Exception as e:  # noqa: BLE001 - we want to record any failure
                failures.append({"table": table, "id": _py(rid), "column": col, "error": repr(e)[:200]})
                out.append("")
        df[col] = out
    return df.set_index(id_col)


def _split_name(split_dir):
    return os.path.basename(os.path.normpath(split_dir))


def load_split(split_dir, use_cache=True, cache_dir=None):
    """Load an official split dir (train_dev_sets or test_set).

    Returns (posts_df, fcs_df, tasks, failures). posts_df is indexed by post_id, fcs_df by
    fact_check_id, both parsed like load_dataset. `failures` lists every cell that could not
    be parsed (those cells are ''). Parsed frames are cached as a pickle in
    data/processed/<split>.pkl and rebuilt when either source CSV's mtime/size changes.
    """
    posts_path = os.path.join(split_dir, "posts.csv")
    fcs_path = os.path.join(split_dir, "fact_checks.csv")
    tasks_path = os.path.join(split_dir, "tasks.json")
    for p in (posts_path, fcs_path, tasks_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    with open(tasks_path, encoding="utf-8") as f:
        tasks = json.load(f)

    stamp = {
        "version": _CACHE_VERSION,
        "posts": (os.stat(posts_path).st_mtime_ns, os.stat(posts_path).st_size),
        "fcs": (os.stat(fcs_path).st_mtime_ns, os.stat(fcs_path).st_size),
    }
    cache_path = os.path.join(cache_dir or PROCESSED_DIR, f"{_split_name(split_dir)}.pkl")
    if use_cache and os.path.isfile(cache_path):
        try:
            with open(cache_path, "rb") as f:
                blob = pickle.load(f)
            if blob.get("stamp") == stamp:
                return blob["posts"], blob["fcs"], tasks, blob["failures"]
        except Exception:  # noqa: BLE001 - corrupt cache -> rebuild
            pass

    failures = []
    fcs_df = _parse_frame(pd.read_csv(fcs_path).fillna(""), _FC_COLS, "fact_check_id", failures, "fact_checks")
    posts_df = _parse_frame(pd.read_csv(posts_path).fillna(""), _POST_COLS, "post_id", failures, "posts")

    if use_cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp = cache_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(
                {"stamp": stamp, "posts": posts_df, "fcs": fcs_df, "failures": failures},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(tmp, cache_path)
    return posts_df, fcs_df, tasks, failures


def load_pairs(split_dir):
    """{filename: DataFrame(post_id, fact_check_id)} for every pairs*.csv in a split dir."""
    names = sorted(n for n in os.listdir(split_dir) if n.startswith("pairs") and n.endswith(".csv"))
    return {n: pd.read_csv(os.path.join(split_dir, n)) for n in names}


def split_kind(split_dir):
    """'test' if tasks.json carries posts_test keys, else 'dev' (train_dev_sets)."""
    with open(os.path.join(split_dir, "tasks.json"), encoding="utf-8") as f:
        tasks = json.load(f)
    return "test" if "posts_test" in tasks["crosslingual"] else "dev"


def load_reference(split_dir, track):
    """Official reference json of a split dir: {post_id (str): [fc_id (int), ...]}."""
    if track not in TRACKS:
        raise ValueError(f"track must be one of {TRACKS}, got {track!r}")
    with open(os.path.join(split_dir, f"{track}_reference.json"), encoding="utf-8") as f:
        return json.load(f)


def eval_units(split_dir, track, eval_split):
    """Evaluation units for a (split_dir, track, eval_split) combination.

    Monolingual: one unit per language in tasks.json, {lang, post_ids, pool_fc_ids}
    (pool = that language's `fact_checks` list). Crosslingual: a single unit with lang
    'cross' and the full crosslingual pool. eval_split 'dev' reads `posts_dev` (train_dev_sets),
    'test' reads `posts_test` (test_set). Gold = load_reference(split_dir, track); every unit's
    post ids are checked to be exactly the reference's post set.
    """
    if track not in TRACKS:
        raise ValueError(f"track must be one of {TRACKS}, got {track!r}")
    if eval_split not in EVAL_SPLITS:
        raise ValueError(f"eval_split must be one of {EVAL_SPLITS}, got {eval_split!r}")
    key = f"posts_{eval_split}"
    with open(os.path.join(split_dir, "tasks.json"), encoding="utf-8") as f:
        tasks = json.load(f)

    t = tasks[track]
    if track == "monolingual":
        units = []
        for lang in sorted(t):
            if key not in t[lang]:
                raise ValueError(f"{split_dir} has no {key!r} (is eval_split={eval_split!r} right for this dir?)")
            units.append(
                {"lang": lang, "post_ids": [int(p) for p in t[lang][key]], "pool_fc_ids": [int(x) for x in t[lang]["fact_checks"]]}
            )
    else:
        if key not in t:
            raise ValueError(f"{split_dir} has no {key!r} (is eval_split={eval_split!r} right for this dir?)")
        units = [
            {"lang": "cross", "post_ids": [int(p) for p in t[key]], "pool_fc_ids": [int(x) for x in t["fact_checks"]]}
        ]

    ref = load_reference(split_dir, track)
    unit_posts = [p for u in units for p in u["post_ids"]]
    if len(unit_posts) != len(set(unit_posts)) or {str(p) for p in unit_posts} != set(ref):
        raise ValueError("tasks.json eval posts do not match the reference file's posts")
    return units
