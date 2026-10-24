# SemEval-2025 Task 7 — Multilingual & Crosslingual Fact-Checked Claim Retrieval

NLP course project. Given a social media post, rank previously fact-checked claims so that a matching one lands in the top 10 (Success@10), within the same language (monolingual track) or across languages (crosslingual track).
Task: [SemEval-2025 Task 7](https://disai.eu/semeval-2025/) · overview paper: [Peng et al., 2025](https://aclanthology.org/2025.semeval-1.323/).

> **The dataset is NOT in this repo and must never be committed.** It is licensed for research use only and distributed on request via [Zenodo (10.5281/zenodo.14989176)](https://doi.org/10.5281/zenodo.14989176). `data/` is gitignored — keep it that way.

## Documents

| | |
|---|---|
| **Stage 1 report (PDF, Slovak)** | [`report/stage1/main.pdf`](report/stage1/main.pdf) — source: [`report/stage1/main.tex`](report/stage1/main.tex) |
| Use of LLMs / AI agents (appendix) | [`docs/llm_usage.md`](docs/llm_usage.md) |
| Literature notes + BibTeX | [`docs/literature.md`](docs/literature.md), [`docs/references.bib`](docs/references.bib) |
| Dataset statistics | [`results/data_stats.md`](results/data_stats.md) |
| All results (every run) | [`results/results.md`](results/results.md), report tables: [`results/report_tables.md`](results/report_tables.md) |

## Status (test S@10)

| System | Monolingual avg | Crosslingual |
|---|---|---|
| Organizers' BM25 | 0.8123 | 0.6028 |
| Organizers' multilingual-E5-large | 0.8589 | 0.6373 |
| Ours — BM25 baseline (English translations, `\w+` tokens) | 0.7714 | 0.5373 |
| Ours — BM25 + stemming/stopwords/Thai segmentation (mono: orig+eng, cross: eng) | 0.8682 | 0.6225 |
| **Ours — mE5-large zero-shot (mono: orig+eng, cross: eng)** | **0.8862** | **0.6813** |

Stage 1 (baselines) is done. Stage 2 roadmap: rank fusion (RRF) → fine-tuning mE5 with hard negatives → cross-encoder reranking → own contribution (truncation handling, temporal signal), plus error analysis. See [`CLAUDE.md`](CLAUDE.md) for the full plan and conventions.

## Team rules (short version)

Full rules: **[CONTRIBUTING.md](CONTRIBUTING.md)**. Read them before your first PR.

1. **Never commit or upload the dataset** or anything containing its text (predictions, exports, screenshots). Results in git are aggregates only.
2. **Tune on dev, run test once** per frozen configuration. Training uses `posts_train` only.
3. **Every experiment runs from a config** and logs to `results/`; every number in the report comes from there.
4. **Branch per issue, PR with one review**, `pytest` green. No direct pushes or force pushes to `main`.
5. **Log every use of LLMs/AI tools** in [`docs/llm_usage.md`](docs/llm_usage.md) the same day (tool, purpose, prompt, what was used and how it was verified). This is mandatory for grading.

## Setup

Install Python 3.13 first; on Windows, `py -3.13 --version` must find that runtime before creating the environment.

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
BM25 runs need only a CPU.

### Data layout

Request the dataset on Zenodo (link above), then unpack it together with the public sample so that you have:

```
data/raw/
  semeval2025_task7_data/   # official release from Zenodo (train_dev_sets/, test_set/, scripts/, baselines/)
  sample/                   # public sample: trial_posts.csv, trial_fact_checks.csv, trial_data_mapping.csv
```

The sample (needed by the tests) is public. In PowerShell:

```powershell
New-Item -ItemType Directory -Force data/raw/sample | Out-Null
foreach ($f in 'trial_posts.csv', 'trial_fact_checks.csv', 'trial_data_mapping.csv', 'load.py') {
  Invoke-WebRequest "https://raw.githubusercontent.com/kinit-sk/semeval_2025/main/sample_data/$f" -OutFile "data/raw/sample/$f"
}
```

In Bash (Linux/macOS or Git Bash):

```bash
mkdir -p data/raw/sample
for f in trial_posts.csv trial_fact_checks.csv trial_data_mapping.csv load.py; do
  curl -sL -o data/raw/sample/$f https://raw.githubusercontent.com/kinit-sk/semeval_2025/main/sample_data/$f
done
```

Parsed CSVs are cached to `data/processed/` on first load. This can take several minutes on a cold Windows setup; later runs reuse the parsed cache. The experiment times below depend on hardware and whether this cache already exists.

## Reproducing the stage-1 results

```bash
python -m pytest -q -rs

# the four test-set runs reported in the stage-1 report (Table "main results")
python scripts/run_bm25.py  --config configs/bm25_mono_test_both_stem_stop_thai.yaml   # CPU, ~20 s
python scripts/run_bm25.py  --config configs/bm25_cross_test_eng_stem_stop.yaml        # CPU, ~25 s
python scripts/run_dense.py --config configs/dense_mono_test_both.yaml                 # GPU, ~7 min (embeds 261k fact-checks once)
python scripts/run_dense.py --config configs/dense_cross_test_eng.yaml                 # GPU, ~5 min

# cross-check any prediction file against the official scoring logic
python scripts/score_official.py --track monolingual \
    --split-dir data/raw/semeval2025_task7_data/test_set \
    --pred results/dense_mono_test_both/predictions.json

# regenerate statistics and report tables
python scripts/data_stats.py
python scripts/make_report_tables.py
```

Expected test S@10 (MRR@10):

| Config | S@10 | MRR@10 | Tolerance |
|---|---|---|---|
| `bm25_mono_test_both_stem_stop_thai` | 0.8682 | 0.7130 | exact (deterministic) |
| `bm25_cross_test_eng_stem_stop` | 0.6225 | 0.3825 | exact (deterministic) |
| `dense_mono_test_both` | 0.8862 | 0.7250 | ±0.005 (GPU/driver numerics) |
| `dense_cross_test_eng` | 0.6813 | 0.4166 | ±0.005 (GPU/driver numerics) |

Dev ablations (all `configs/*_dev_*.yaml`) are listed with their results in [`results/results.md`](results/results.md).
Each run writes `results/<run_name>/metrics.json` and `predictions.json` (gitignored, regenerable) and upserts a row in `results/results.md`.
For an independent reproduction, copy each frozen YAML and change only `run_name` (also rename the YAML to match), so the original metrics and predictions are preserved.
The test count depends on the YAML files and locally saved predictions. Some parameterized config checks deliberately skip when they do not apply; the model test needs CUDA and a loadable E5 model, and saved-prediction checks need the official data and prediction files. `-rs` shows each skip reason.

## Evaluation protocol

- **Tune on dev** (`train_dev_sets`, `posts_dev`). Each test configuration is chosen on dev and run once on test.
- Retrieval pools come from `tasks.json`: per language for monolingual, one shared pool for crosslingual.
- Scoring matches the official scripts: monolingual = mean of per-language S@10, crosslingual = micro S@10. We also report MRR@10. Our scorer reproduces the organizers' published baseline numbers exactly.

## Building the report

The report compiles with [Tectonic](https://tectonic-typesetting.github.io/) (single binary, fetches packages on demand) or any TeX distribution / Overleaf (pdfLaTeX or XeLaTeX):

```bash
cd report/stage1
tectonic main.tex          # → main.pdf (runs BibTeX automatically; bibliography: docs/references.bib)
```

On Overleaf, upload `main.tex`, the `tables/` folder and `docs/references.bib`, and change `\bibliography{../../docs/references}` to `\bibliography{references}`.

## Layout

```
src/data.py                  loading + parsing, tasks.json splits/pools, text construction
src/evaluate.py              S@10, MRR@10, per-language breakdown, official-format predictions
src/retrieval/bm25.py        sparse BM25 (matches rank_bm25 BM25Okapi) + analyzer (stemming, stopwords, Thai)
src/retrieval/dense.py       fp16 bi-encoder embeddings (mE5), on-disk embedding cache, matmul search
scripts/run_bm25.py          config-driven BM25 runs
scripts/run_dense.py         config-driven dense (mE5-large) runs
scripts/score_official.py    score any prediction file, cross-checked against the official logic
scripts/data_stats.py        aggregate dataset statistics → results/data_stats.{json,md}
scripts/make_report_tables.py  report tables from results/*/metrics.json → results/report_tables.md
configs/                     one YAML per experiment
results/                     metrics per run, results.md summary, statistics, report tables
report/stage1/               stage-1 report (LaTeX source, tables, PDF)
docs/                        literature notes, BibTeX, LLM/AI usage appendix
tests/                       pytest suite
```
