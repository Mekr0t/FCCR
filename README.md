# SemEval-2025 Task 7 — Multilingual & Crosslingual Fact-Checked Claim Retrieval

Replication of [SemEval-2025 Task 7](https://disai.eu/semeval-2025/): given a social media post, rank previously fact-checked claims so that a matching one lands in the top 10 (Success@10).
Overview paper: [Peng et al., 2025](https://arxiv.org/abs/2505.10740).

> **The dataset is NOT in this repo and must never be committed.** It is licensed for research use only and distributed on request via [Zenodo (10.5281/zenodo.14989176)](https://doi.org/10.5281/zenodo.14989176). `data/` is gitignored — keep it that way.

## Setup

```bash
py -3.13 -m venv .venv            # Windows (python3.13 -m venv .venv elsewhere)
.venv\Scripts\activate            # source .venv/bin/activate on Linux/macOS
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8 wheels (RTX 50-series / Blackwell needs cu128)
pip install -r requirements.txt   # torch line is already satisfied; the rest comes from PyPI
```

Dense retrieval needs a CUDA GPU (developed on an RTX 5070 laptop, 8 GB). Check the install with
`python -c "import torch; a=torch.randn(512,512,device='cuda'); print((a@a).sum())"` (an error mentioning
"no kernel image" means a wheel without sm_120 support). The first dense run downloads
`intfloat/multilingual-e5-large` (~2.2 GB) from the Hugging Face hub; embeddings are cached in `cache/` (gitignored).

### Data layout

Unpack the official release and the public sample so that you have:

```
data/raw/
  semeval2025_task7_data/   # official release from Zenodo (train_dev_sets/, test_set/, scripts/, baselines/)
  sample/                   # public sample: trial_posts.csv, trial_fact_checks.csv, trial_data_mapping.csv
```

The sample (needed by the tests) is public; fetch it with:

```bash
mkdir -p data/raw/sample
for f in trial_posts.csv trial_fact_checks.csv trial_data_mapping.csv load.py; do
  curl -sL -o data/raw/sample/$f https://raw.githubusercontent.com/kinit-sk/semeval_2025/main/sample_data/$f
done
```

Parsed CSVs are cached to `data/processed/` on first load (~15–25 s once, ~1 s afterwards).

## Usage

```bash
python -m pytest -q                                              # unit tests (needs data/raw/sample)
python scripts/run_bm25.py --config configs/bm25_mono_dev_eng.yaml
python scripts/run_dense.py --config configs/dense_mono_dev_both.yaml   # mE5-large, GPU
python scripts/score_official.py --track monolingual \
    --split-dir data/raw/semeval2025_task7_data/test_set \
    --pred results/bm25_mono_test_eng/predictions.json            # our scorer + official logic, must agree
```

Each run writes `results/<run_name>/metrics.json` and `predictions.json` (gitignored, regenerable) and upserts a row in [`results/results.md`](results/results.md).

## Evaluation protocol

- **Tune on dev** (`train_dev_sets`, `posts_dev`). The test set is used only for final numbers and baseline replication.
- Retrieval pools come from `tasks.json`: per language for monolingual, one shared pool for crosslingual.
- Scoring matches the official scripts: monolingual = mean of per-language S@10, crosslingual = micro S@10. We also report MRR@10.

## Layout

```
src/data.py              loading + parsing, tasks.json splits/pools, text construction
src/evaluate.py          S@10, MRR@10, per-language breakdown, official-format predictions
src/retrieval/bm25.py    sparse BM25 (matches rank_bm25 BM25Okapi)
src/retrieval/dense.py   fp16 bi-encoder embeddings (mE5), on-disk embedding cache, matmul search
scripts/run_bm25.py      config-driven BM25 runs
scripts/run_dense.py     config-driven dense (mE5-large) runs
scripts/score_official.py  score any prediction file, cross-checked against the official logic
configs/                 one YAML per experiment
results/                 metrics per run + results.md summary
tests/                   pytest suite
```

## Status

| System (test S@10) | Monolingual avg | Crosslingual |
|---|---|---|
| Organizers' BM25 | 0.8123 | 0.6028 |
| Organizers' multilingual-E5-large | 0.8589 | 0.6373 |
| Ours — BM25 baseline (English translations, `\w+` tokens) | 0.7714 | 0.5373 |
| Ours — BM25 + stemming/stopwords/Thai segmentation (mono: orig+eng, cross: eng) | 0.8682 | 0.6225 |
| Ours — mE5-large zero-shot (mono: orig+eng, cross: eng) | 0.8862 | 0.6813 |

Roadmap: dense retrieval (mE5 / BGE-M3) → rank fusion → reranking. See `CLAUDE.md` for the full plan and conventions.
