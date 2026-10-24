# Stage-1 clean-clone reproduction — 2026-10-22–24

Issue: [#1](https://github.com/Mekr0t/FCCR/issues/1). Reproduced by Jakub Spišák with Codex assistance.
Source: `7c55503833ee5dab4ee1e601f2694fafffbce56d` (fresh clone of `main`, branch `s1-1-reproduce`).

All four frozen configurations satisfy the issue: both BM25 metrics match exactly; both dense metrics are within the absolute ±0.005 tolerance. The official-logic cross-check prints `AGREE` for all four prediction files.

| Frozen config | Expected S@10 | Got S@10 | Expected MRR@10 | Got MRR@10 | OK | Final invocation (s) |
|---|---:|---:|---:|---:|:---:|---:|
| `bm25_mono_test_both_stem_stop_thai` | 0.8682 | 0.8682 | 0.7130 | 0.7130 | yes | 224.3 |
| `bm25_cross_test_eng_stem_stop` | 0.6225 | 0.6225 | 0.3825 | 0.3825 | yes | 33.3 |
| `dense_mono_test_both` | 0.8862 | 0.8862 | 0.7250 | 0.7250 | yes | 138.6 |
| `dense_cross_test_eng` | 0.6813 | 0.6810 | 0.4166 | 0.4163 | yes | 473.7 |

Full-precision differences (new minus reference):

| Config | Δ S@10 | Δ MRR@10 |
|---|---:|---:|
| `bm25_mono_test_both_stem_stop_thai` | +0.000000000000 | +0.000000000000 |
| `bm25_cross_test_eng_stem_stop` | +0.000000000000 | +0.000000000000 |
| `dense_mono_test_both` | +0.000000000000 | +0.000046666667 |
| `dense_cross_test_eng` | -0.000250000000 | -0.000328968254 |

## Setup and provenance

- Windows 11, build 10.0.26300, x64; PowerShell 7.6.5; CPython 3.13.16.
- NVIDIA RTX 4060 Laptop, 8 GB (8,188 MiB); driver 566.24; PyTorch 2.11.0+cu128 / CUDA runtime 12.8. The CUDA matrix multiplication check passed.
- Fresh `.venv`, full pinned `requirements.txt`; no environment, parsed-data cache, model cache or embeddings copied from the earlier checkout. Python and uv were already available as isolated workspace tools; uv was used instead of pip. The full installed package versions are in the companion JSON.
- While the CUDA installation ran, the two BM25 runs used a separate fresh CPU environment with the same pinned BM25 dependencies and CPU PyTorch 2.11.0. The dense runs and final complete test suite used the new CUDA `.venv`.
- Setup took approximately 15 minutes from cloning to the passing E5 GPU sample test, including data preparation, baseline tests, parallel dependency installation and model download. The CUDA torch download/install alone took about 7 minutes; E5 download took 175.8 seconds. These are wall-clock observations, not a bandwidth benchmark.
- The human supplied the licensed Zenodo release locally. All 28 copied files (328,181,065 bytes) matched their source SHA-256 hashes. The public sample was downloaded afresh. No dataset text, predictions, caches or weights are committed.
- E5 model: `intfloat/multilingual-e5-large`, Hub revision `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3`. Only the required model/tokenizer files were downloaded, then dense runs used the local Hub cache in offline mode.
- Each reproduction YAML changes only `run_name`, with `_repro_20261022` appended. Model, text variant, batch size 32, sequence length 512, top-100 output, seed 42 and evaluation code are unchanged. The original reference metrics/configs are preserved. No tuning was done on test.
- Test CSV parsing started cold in this clone; later runs reused only the cache generated here. Both dense variants initially started without embeddings. The human stopped the monolingual dense run on October 22 before it produced predictions or metrics; on October 24 it resumed with its own 196,608 checkpointed fact-check embeddings. The crosslingual variant started cold, and neither used post embeddings from another run. The suffix `20261022` records the start date.
- The table reports the final process invocation time. Monolingual dense additionally spent at least 545 s embedding before the interruption; its final invocation alone is not a cold-run benchmark. The multi-day pause is excluded. No completed test experiment was rerun or used to select a variant.

## Verification

- Before changes, the fresh CPU setup passed **226 tests, 59 skipped**: 57 inapplicable parameterized config checks, one missing transformer/GPU test, and one test without saved official predictions.
- The real E5 GPU sample test passed separately (**1 passed**, 29.94 s), including normalized embeddings, repeatability and retrieval on the public sample.
- Final full suite in the CUDA environment: **259 passed, 65 skipped**, 16.64 s. All skips are inapplicable parameterized config checks; the GPU/model test and checks on all four saved predictions ran.
- `scripts/score_official.py` was run on all four reproduction prediction files; each exited 0 with `AGREE`. This compares S@10 with the independent official-logic copy in the repository; MRR has no official counterpart.
- Aggregate metrics and per-language results are stored in each reproduction run directory and indexed in `results/results.md`. The companion `reproduction_20261022.json` stores full precision, provenance and versions.

## README findings and environment-specific notes

- The public-sample download loop was Bash-only despite the Windows setup. Added its PowerShell equivalent.
- The stated 43 tests was obsolete. Removed the fixed count and documented conditional skips and `-rs`.
- The first cold Windows parse made the initial BM25 run take 224.3 s rather than the advertised roughly 20 s; the crosslingual run with the parsed cache took 33.3 s. The README now distinguishes cold parsing from cached timing.
- Added instructions to preserve reference results by copying frozen YAMLs and changing only the run name.
- The `py -3.13` launcher did not provide a usable 3.13 runtime in this environment; the pre-existing isolated CPython 3.13.16 was selected explicitly with uv. PyThaiNLP data, temporary files and the Hugging Face cache were redirected into writable workspace directories; PyThaiNLP was used offline with its bundled tokenizer data. These are local sandbox accommodations.

The launcher error from `py -3.13 --version` was:

```text
No suitable Python runtime found
Pass --list (-0) to see all detected environments on your machine
or set environment variable PYLAUNCHER_ALLOW_INSTALL to use winget
or open the Microsoft Store to the requested version.
```

- During token-length measurement, Transformers warned about a sequence longer than 512 tokens. The existing encoder measures lengths before applying its configured 512-token truncation; the warning did not abort either experiment.

This is an AI-assisted teammate reproduction of issue #1; it does not establish the separate bonus requirement for a reproducer outside the team.
