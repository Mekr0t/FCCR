"""Static checks for every configs/*.yaml (no data, no models needed).

The set of keys a config may use is derived from the runner sources (AST scan of the
`cfg.get("...")` / `cfg["..."]` reads in scripts/run_bm25.py and scripts/run_dense.py), so a key
the runner does not read is reported as unknown instead of silently ignored.
"""

import ast
import glob
import os

import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_FILES = sorted(glob.glob(os.path.join(ROOT, "configs", "*.yaml")))
COMMON_REQUIRED = ["run_name", "data_dir", "text_variant", "k", "top_n"]
TRACKS = {"monolingual", "crosslingual"}
SPLITS = {"dev", "test"}
SPLIT_DIR = {"dev": "train_dev_sets", "test": "test_set"}
SAMPLE_REQUIRED = ["posts_file", "fact_checks_file", "mapping_file"]
DENSE_REQUIRED = ["model", "model_label"]


def _cfg_keys_read(runner_path):
    """String keys the runner reads from its `cfg` dict (cfg.get("x", ...) and cfg["x"])."""
    with open(runner_path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=runner_path)
    keys = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "cfg"
                and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            keys.add(node.args[0].value)
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "cfg"
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
            keys.add(node.slice.value)
    return keys


BM25_KEYS = _cfg_keys_read(os.path.join(ROOT, "scripts", "run_bm25.py"))
DENSE_KEYS = _cfg_keys_read(os.path.join(ROOT, "scripts", "run_dense.py"))


def _load(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_dense(name, cfg):
    return "model" in cfg or name.startswith("dense_")


def _ids():
    return [os.path.basename(p) for p in CONFIG_FILES]


@pytest.fixture(params=CONFIG_FILES, ids=_ids())
def config(request):
    path = request.param
    cfg = _load(path)
    return os.path.basename(path), os.path.splitext(os.path.basename(path))[0], cfg


def test_derived_key_sets_are_sane():
    # The AST scan must find the keys we know the runners use; an empty set means a broken scan.
    assert {"run_name", "data_dir", "text_variant", "k", "top_n", "track", "eval_split", "analyzer"} <= BM25_KEYS
    assert set(DENSE_REQUIRED) <= DENSE_KEYS
    assert "analyzer" not in DENSE_KEYS  # run_dense.py does not read the BM25 analyzer


def test_config_loads_as_mapping(config):
    fname, _, cfg = config
    assert isinstance(cfg, dict), f"{fname}: YAML must parse to a mapping, got {type(cfg).__name__}"


def test_run_name_equals_filename_stem(config):
    fname, stem, cfg = config
    assert cfg.get("run_name") == stem, f"{fname}: run_name {cfg.get('run_name')!r} must equal the filename stem"


def test_common_required_keys(config):
    fname, _, cfg = config
    missing = [k for k in COMMON_REQUIRED if k not in cfg]
    assert not missing, f"{fname}: missing required keys {missing}"


def test_text_variant_and_k_top_n(config):
    fname, _, cfg = config
    assert cfg.get("text_variant") in {"orig", "eng", "both"}, f"{fname}: text_variant {cfg.get('text_variant')!r}"
    assert cfg.get("k") == 10, f"{fname}: k must be 10 (Success@10), got {cfg.get('k')!r}"
    assert isinstance(cfg.get("top_n"), int) and cfg["top_n"] >= cfg["k"], f"{fname}: top_n must be an int >= k"


def test_official_fields(config):
    fname, stem, cfg = config
    if "track" not in cfg:
        pytest.skip("sample config (no track)")
    assert cfg["track"] in TRACKS, f"{fname}: track {cfg['track']!r}"
    assert "eval_split" in cfg and cfg["eval_split"] in SPLITS, f"{fname}: eval_split {cfg.get('eval_split')!r}"
    split = cfg["eval_split"]
    data_leaf = os.path.basename(os.path.normpath(cfg["data_dir"]))
    assert data_leaf == SPLIT_DIR[split], f"{fname}: eval_split {split} needs data_dir ending in {SPLIT_DIR[split]}"
    run = cfg["run_name"]
    track_token = "mono" if cfg["track"] == "monolingual" else "cross"
    other_token = "cross" if track_token == "mono" else "mono"
    other_split = "test" if split == "dev" else "dev"
    assert f"_{split}_" in run, f"{fname}: run_name must contain _{split}_"
    assert f"_{track_token}_" in run, f"{fname}: run_name must contain _{track_token}_"
    assert f"_{other_token}_" not in run, f"{fname}: run_name mentions the other track ({other_token})"
    assert f"_{other_split}_" not in run, f"{fname}: run_name mentions the other split ({other_split})"


def test_sample_fields(config):
    fname, _, cfg = config
    if "track" in cfg:
        pytest.skip("official config")
    missing = [k for k in SAMPLE_REQUIRED if k not in cfg]
    assert not missing, f"{fname}: sample config missing {missing}"


def test_analyzer_keys_and_values(config):
    fname, _, cfg = config
    if "analyzer" not in cfg:
        pytest.skip("no analyzer block")
    an = cfg["analyzer"]
    assert isinstance(an, dict), f"{fname}: analyzer must be a mapping"
    unknown = set(an) - {"stem", "stopwords", "thai_tokenizer"}
    assert not unknown, f"{fname}: unknown analyzer keys {sorted(unknown)}"
    bad = {k: v for k, v in an.items() if not isinstance(v, bool)}
    assert not bad, f"{fname}: analyzer values must be booleans, got {bad}"


def test_analyzer_keys_match_the_implementation(config):
    fname, _, cfg = config
    if "analyzer" not in cfg:
        pytest.skip("no analyzer block")
    from src.retrieval.bm25 import Analyzer
    # the config may only use options Analyzer.from_config accepts
    for key in cfg["analyzer"]:
        if key == "thai_tokenizer":
            pytest.importorskip("pythainlp")  # the Thai option imports pythainlp at construction
        Analyzer.from_config({key: True})
    with pytest.raises(ValueError):
        Analyzer.from_config({"stem": True, "bogus": True})


def test_dense_or_bm25_keys(config):
    fname, stem, cfg = config
    dense = _is_dense(stem, cfg)
    if dense:
        assert "model" in cfg and stem.startswith("dense_"), f"{fname}: dense configs need model and a dense_ name"
        missing = [k for k in DENSE_REQUIRED if k not in cfg]
        assert not missing, f"{fname}: dense config missing {missing}"
        allowed = DENSE_KEYS
    else:
        assert not stem.startswith("dense_"), f"{fname}: dense_ name without a model key"
        allowed = BM25_KEYS
    unknown = sorted(set(cfg) - allowed)
    assert not unknown, (
        f"{fname}: keys not read by {'run_dense.py' if dense else 'run_bm25.py'}: {unknown}"
    )
