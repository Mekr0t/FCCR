# Appendix — Use of LLMs and AI agents

Required by the project assignment: every use of an LLM or AI agent is listed here with the tool/model, purpose, prompts, and relevant outputs.

**Tool:** Claude Code (Anthropic CLI agent). Main model: Claude Opus 5.5 (`claude-opus-5-5`), which planned the work and delegated subtasks to sub-agents:
- *worker* sub-agents (Claude Haiku 5.5 / Claude Sonnet 5.5) that wrote and ran code;
- *judge* sub-agents (Claude Sonnet) that independently checked the worker's output against explicit acceptance criteria.

Full raw transcripts (every prompt, tool call, and output) are kept by Claude Code as JSONL files in `~/.claude/projects/D--skola-nlp-SemEval-2025/` and can be attached if required.

All numbers reported in the project come from running the code in this repository, not from model output. Team members reviewed the generated code.

---

## Session 1 — 2026-10-12 / 2026-10-13: project setup, data pipeline, BM25 baseline

### 1. Task research and project bootstrap
- **User prompt:** "https://disai.eu/semeval-2025/ SemEval 2025 Shared Task 7: Multilingual and Crosslingual Fact-Checked Claim Retrieval — i need to replicate this, can you help me get started"
- **What the agent did:**
  - Read the task page, the official GitHub repo (`kinit-sk/semeval_2025`), the overview paper (arXiv 2505.10740), and the Zenodo record.
  - Summarised the tracks, metric (Success@10), the organizers' baselines, and the top systems.
  - Found where the dataset is distributed and that access is granted on request.
- **Delegated (worker, Haiku):** set up the project skeleton:
  - venv and pinned dependencies;
  - data loader for the stringified-tuple CSVs (reusing the official `parse_col` logic);
  - evaluator (S@10, MRR, per-language, macro average);
  - BM25 baseline script;
  - pytest suite;
  - smoke test on the public sample data.
- **Verified by:** judge sub-agent (PASS 5/5): tests pass, metrics hand-checked on a constructed example, no gold-label leakage.
- **Output:** sample-data BM25 S@10 = 0.957 (English text) / 0.702 (original text). This is only a smoke test, since the pool has just 50 fact-checks.

### 2. Dataset access problem
- **User prompt:** "there may be a little problem, i cant create account to send request for dataset (error while creating account)"
- **Output:** advice to log in to Zenodo with GitHub/ORCID, or to email the organizers or the course instructor. No code involved.

### 3. Integrating the official dataset
- **User prompt:** "got dataset its in root folder"
- **What the agent did:**
  - Read the release README and the official loading/scoring scripts.
  - Found 4 mismatches with our code:
    - the scorer requires integer fact-check ids;
    - it requires ≥10 predictions per post;
    - the pairs files have no `pair_lang` column;
    - `rank_bm25` is too slow for pools of 85k–270k fact-checks.
  - Moved the data to `data/raw/` (gitignored) and recorded the verified facts in `CLAUDE.md`.
- **Delegated (worker, Sonnet):**
  - loader for `tasks.json` splits and pools;
  - sparse-matrix BM25, unit-tested to give identical rankings to `rank_bm25`;
  - official prediction format and MRR@10;
  - `scripts/score_official.py`, which cross-checks our scorer against a copy of the official scoring logic;
  - BM25 runs on dev and test.
- **Verified by:** judge sub-agent (PASS 5/5). Our scorer reproduces the paper's numbers for all 4 organizer baselines exactly (e.g. BM25 0.8123 / 0.60275). Predictions stay inside the pool, and there is no leakage.
- **Output (S@10):**

  | Run | Monolingual | Crosslingual |
  |---|---|---|
  | Dev, English text | 0.685 | 0.413 |
  | Dev, original text | 0.700 | — |
  | Test | 0.771 | 0.537 |

  Our BM25 is lower than the organizers' BM25 (0.812 / 0.603 on test). The judge found no bug; the likely causes are tokenization (no stemming, Thai has no word spaces) and text construction.
- **Note:** during diagnosis the judge ran a small text-variant comparison on 3 test languages. It was not used for any design decision; all tuning is done on dev.

### 4. Repository preparation for GitHub
- **User prompt:** "can you prepare repo for pushing on github like gitignore and stuff, add dataset to gitignore (its forbiden), my teamates already have access"
- **What the agent did:**
  - Wrote `.gitignore`: the dataset and all derived files are excluded, with extra safety-net patterns.
  - Added `.gitattributes` and `README.md`.
  - Ran `git add` as a dry run and confirmed that no data file would be committed.

### 5. Scope decisions
- **User prompts:** "can i start with Monolingual and if there will be no complications move to Crosslingual or its just better to start Crosslingual right away"; "own system, uploaded assigment in root folder"
- **Output:**
  - Recommended monolingual as the main development loop, with crosslingual run alongside, since both share the same pipeline and cached embeddings.
  - Summarised the assignment requirements and created this appendix.

### 6. Deadlines and plan
- **User prompt:** "1. stage Oct 25th, 2. stage Dec 6th"
- **Output:**
  - Recorded both deadlines in `CLAUDE.md`.
  - Proposed a stage-1 timeline, a list of literature sources, and a split of roles in the team.

### 7. Zero-shot dense retrieval baseline (multilingual-E5-large)
- **Delegated (worker, Sonnet):**
  - install PyTorch built for CUDA 12.8;
  - `src/retrieval/dense.py`: fp16 encoder with e5 `query:`/`passage:` prefixes and 512 max tokens, an on-disk embedding cache, matmul search;
  - `scripts/run_dense.py`, configs, and tests;
  - dev runs for the text variants: original, English, original + English;
  - choose the best variant per track on dev, then run test once with that variant.
- **Verified by:** judge sub-agent (PASS 5/5). It checked:
  - no leakage, and predictions stay inside the `tasks.json` pools;
  - cached embedding rows match freshly computed ones (cosine ≥ 0.9995);
  - the variant was chosen on dev only, since all dev runs come before the test runs;
  - results reproduce, and the official scorer agrees.
- **Output (S@10):**

  | Run | Monolingual | Crosslingual |
  |---|---|---|
  | Dev, original + English | 0.862 | — |
  | Dev, English | — | 0.591 |
  | Test | 0.886 | 0.681 |
  | Organizers' mE5, test | 0.859 | 0.637 |

  - The monolingual gain over the organizers' mE5 comes mainly from ms/th/fr/pl/ar, where the model sees the original and English text together. English and Portuguese are flat.
  - With original + English text, 11% of posts are truncated at 512 tokens.

### 8. Stronger BM25 analyzer and crosslingual dense variant
- **User prompt:** "yes run them and then commit"
- **Delegated (worker, Sonnet):**
  - (A) Crosslingual dense retrieval with original + English text, on dev only.
  - (B) Configurable BM25 analyzer:
    - Snowball stemming (PyStemmer);
    - stopword removal (stopwordsiso);
    - Thai word segmentation (pythainlp newmm).
  - Dev ablation of (B), then one test run per track with the config that scored best on dev.
- **Verified by:** judge sub-agent (PASS 5/5). It checked:
  - backward compatibility (old results reproduce exactly);
  - the language-field format fix (only the test fact-checks use the flat format, so dev runs were unaffected);
  - one test run per track, with no extra test results;
  - test predictions are byte-identical on re-run, and the official scorer agrees;
  - no leakage;
  - the analyzer works offline.
- **Output:**
  - Crosslingual dense on dev: original + English text scores 0.533, below English-only at 0.591, so it was not run on test.
  - BM25 dev, monolingual (S@10):

    | Config | S@10 |
    |---|---|
    | English text (old) | 0.685 |
    | + stemming | 0.723 |
    | + stemming + stopwords | 0.773 |
    | Original + English, stemming + stopwords + Thai segmentation | **0.793** |

  - BM25 dev, crosslingual (S@10): 0.413 → 0.520 (English, stemming + stopwords).
  - BM25 test (S@10): monolingual 0.868, crosslingual 0.623. The organizers' BM25 scores 0.812 / 0.603.
  - Finding: the default `\w+` tokenizer splits Thai text into meaningless fragments, because Thai vowel and tone marks are not word characters.

### 9. Commit of stage-1 baselines
- The agent ran `git add` and checked the staged file list for data files before committing.

## Session 2 — 2026-10-16

### 10. Literature collection for the stage-1 review
- **User prompt:** "yeah gather them" (reply to an offer to collect sources for the literature review)
- **What the agent did:**
  - Read the task overview paper (arXiv 2505.10740).
  - Scanned the SemEval-2025 proceedings index for every Task 7 paper.
  - Opened each key paper's ACL Anthology page to check its title, authors, pages and ID. The proceedings index has links that are off by one, so each ID was checked individually.
  - Used web search to locate the foundational papers (MultiClaim, Shaar 2020/2022, Kazemi 2021, Vo & Lee 2020, GTR, mE5, BGE-M3, PyThaiNLP).
- **Output:**
  - `docs/literature.md`: annotated tables (task/dataset, prior work, system papers with scores and takeaways, methods), plus takeaways for the report.
  - `docs/references.bib`: BibTeX entries.
- **Caveats:** a few BibTeX entries were assembled from the verified fields rather than copied from the official export (e.g. author lists shortened with "others", one page range missing). Re-export them from the Anthology before final submission. Team members still need to read the papers; the one-line summaries come from abstracts and the overview paper.

### 11. Dataset statistics and report tables (stage-1)
- **Tool / model:** Claude Code subagent (Claude Haiku 5.5, claude-haiku-5-5), delegated general-purpose task from the main session.
- **Prompt (summary):** "Write `scripts/data_stats.py` (aggregate-only dataset statistics for both official splits, written to `results/data_stats.{json,md}`), `scripts/make_report_tables.py` (main test table, per-language table, dev ablation table, built from `results/*/metrics.json` and the organizers' baseline predictions, written to `results/report_tables.md`), and a pytest for the bold-best logic. Do not commit; output must contain no post or fact-check text; reuse `src/data.py` without changing it."
- **What the agent did:** read `src/data.py`, `src/evaluate.py` and the existing tests; wrote the three files; ran both scripts; ran pytest (43 passed: 39 existing + 4 new).
- **Output used:** `results/data_stats.md`, `results/report_tables.md` (generated; no model-written prose in them beyond fixed headings and footnotes).
- **Human review:** the generated numbers were checked only by the scoring code itself. The organizer rows were checked against the published S@10 values (max difference 0.00004). Read the statistics before quoting them in the report.

### 12. Draft of the stage-1 report (Slovak, LaTeX)
- **User prompts:** "if everuthing clear for now can we finish things in stage 1"; report language and format chosen in a dialog: "Slovak, LaTeX".
- **What the agent did (main model, Claude Opus 5.5, no sub-agent):** wrote `report/stage1/main.tex` and `report/stage1/tables/*.tex`:
  - sections: introduction, related work, data, methodology, baselines, results with interpretation, stage-2 plan, reproducibility;
  - citations from `docs/references.bib`;
  - all numbers in the tables are copied from the generated, judge-verified files `results/report_tables.md` and `results/data_stats.md`.
- **Caveats:** explanations the experiments have not confirmed yet are marked as hypotheses in the text. Places needing team input are marked `\todo{}`: author names, repository URL, the Slovak translation of this appendix. The team must review and edit the text.
- **Verified by:** judge sub-agent (PASS 5/5) on the statistics and tables (entry 11). It independently confirmed:
  - 0% OCR-only posts on test is a real property of the data;
  - the Hindi share of the crosslingual test set (40.9% of posts);
  - all table values against the metrics files.

### 13. README update
- **User prompt:** "readme seems little bit update add like link to main.pdf or something also commit and push"
- **What the agent did:** rewrote `README.md`:
  - a documents table linking the stage-1 PDF, the LLM-usage appendix, the literature notes, the statistics and the results;
  - the current status table;
  - a "Reproducing the stage-1 results" section with the exact commands and the expected S@10/MRR@10 and tolerances for the 4 reported test runs;
  - report build instructions (Tectonic / Overleaf);
  - an updated repository layout.
- **Output:** the expected numbers in the README come from the judge-verified `metrics.json` files.

### 14. Team rules (CONTRIBUTING.md)
- **User prompt:** "create contributing.md where all the rules will be then also add short version of team rules in readme with link to that md file then commit and push"
- **What the agent did:** wrote `CONTRIBUTING.md` with sections on:
  - data/licence handling;
  - experiment and evaluation protocol;
  - code conventions;
  - git workflow;
  - issues;
  - mandatory documentation of LLM/AI use;
  - report rules;
  - deadlines.

  Added a 5-point summary with a link to `README.md`. The rules are drawn from the project's existing conventions (`CLAUDE.md`) and the assignment requirements.

### 15. CI workflow and rules for AI agents
- **User prompts:** "also do you think its worth to create CI workflof? if so could you create one"; "can you edit claude.md so when some teamate will type like 'clone this repo and help with issue #' agent will go by the rules, dont do anything that shouldnt do or rogue …"
- **What the agent did:**
  - Wrote `.github/workflows/ci.yml` with three jobs:
    - (1) a data guard that fails if dataset-like files or files over 5 MB are tracked;
    - (2) `pytest` on Python 3.13 with CPU PyTorch and the public sample data;
    - (3) a Tectonic build of the report, which fails on undefined citations; it runs only when `report/` or the BibTeX file changes.
  - Added a "Rules for AI agents" section and a "Current state" section to `CLAUDE.md`, covering:
    - issue workflow and scope;
    - forbidden actions (data, git, test split, evaluation code);
    - when to ask the human;
    - honesty;
    - mandatory logging.
  - `AGENTS.md` (written by the user) points Codex to `CLAUDE.md`.
- **Verified:** the workflow YAML parses. The data guard was run on the repo (no hits) and on synthetic paths (correct hits). A fresh clone with the CI install set (no transformers stack) and the GPU hidden gave 42 passed, 1 skipped (GPU test). The first simulation without PyTorch caught a failing test, so CPU PyTorch was added. The report-job citation check was run on the local build log.

## Session 3 — 2026-10-16

### 16. Runtime pool check and test suite for configs, pinned sample results, scorer parity and local data
- **Started by:** Roman (GitHub user Mekr0t), through a Claude Code session (main agent: Claude Opus 5.5, `claude-opus-5-5`), which delegated this subtask to a sub-agent.
- **Tool and model:** Claude Code sub-agent on Claude Haiku 5.5 (`claude-haiku-5-5`).
- **User prompt (faithful summary of the delegated task):** read `CLAUDE.md` (rules for AI agents) and `CONTRIBUTING.md`; add (1) a pinned BM25 sample test (S@10 0.9574 / 0.7794 for eng, 0.7021 / 0.5497 for orig) that does not write to `results/`; (2) a test over every `configs/*.yaml` (required keys, run_name equals filename stem, track/split/run_name consistency, dense vs BM25 keys, unknown keys derived from the runner code); (3) a toy parity test of `src/evaluate.py` against the copied official scorer in `scripts/score_official.py`; (4) a runtime helper that checks predictions are inside each post's pool, have at least 10 unique ids, and cover exactly the evaluated posts, called by both official runners before saving; (5) `needs_data` tests on saved official predictions that skip without the dataset. Do not commit; CI must pass without the dataset, CUDA or transformers.
- **What the agent did:**
  - added `src/checks.py` with `assert_predictions_in_pool(preds, units)`; called it in `scripts/run_bm25.py` and `scripts/run_dense.py` (official path only, right before `save_predictions`); default behaviour otherwise unchanged;
  - added `tests/test_checks.py` (toy data), `tests/test_pinned_sample.py` (calls the runner's `run_sample` in memory, nothing written), `tests/test_configs.py` (allowed keys read from the runners' source via AST scan), `tests/test_scorer_parity.py` (imports the official functions from `scripts/score_official.py`, round-trips through `save_predictions`), `tests/test_local_data.py` (marker `needs_data`, registered in `pytest.ini`; environment variable `SEMEVAL_OFFICIAL_DIR` only to simulate a missing dataset);
  - one known config issue was not changed: `configs/bm25_sample.yaml` has `run_name: bm25_sample_eng`, which differs from its filename stem. It is recorded as a strict xfail in `tests/test_configs.py`;
  - re-ran `configs/bm25_mono_dev_eng.yaml`: S@10 0.6847 and MRR@10 0.4942 as before; `results/results.md` byte-identical; the regenerated `metrics.json` differed only in timing fields and line endings, so it was restored with `git checkout` to keep the committed file unchanged.
- **Verified:** full suite with data present: 248 passed, 57 skipped, 1 xfailed (the known config mismatch). With `CUDA_VISIBLE_DEVICES=-1`: 247 passed, 58 skipped (GPU test), 1 xfailed. With `SEMEVAL_OFFICIAL_DIR` pointing at a missing path: all 22 `needs_data` tests skip. `git status` after the suite shows no changes under `results/` or `configs/`. No judge agent was run on this session. No run used the test split.

### 17. Follow-up fixes to entry 16 (main session, Claude Opus 5.5)
- **Context:** the judge sub-agent passed entry 16's work (PASS 5/5). It ran a CI simulation with the transformers stack blocked, no CUDA and no dataset: 225 passed, 80 skipped, 0 failed. It also checked the dense and BM25 runtime check end to end on cached dev runs.
- **Changes:**
  - renamed `configs/bm25_sample.yaml` → `configs/bm25_sample_eng.yaml`, so the filename matches its `run_name`, and removed the strict xfail from `tests/test_configs.py`;
  - moved the pool check in `run_bm25.py`/`run_dense.py` before `metrics.json` is written, so a failing run leaves no stale metrics.
- **Verified:** `pytest` gives 249 passed, 57 skipped (skips are config checks that don't apply to that config type). The sample run reproduces S@10 0.9574.

## Session 5 — 2026-10-22–24 (issue #1)

### 19. Clean-clone reproduction of the four frozen stage-1 configurations
- **Started by:** Jakub Spišák (GitHub `jakub-spisak`).
- **Tool / model:** OpenAI Codex desktop, GPT-6 family (the exact runtime model identifier was not exposed in this session).
- **User prompts (faithful summary):** inspect issue #1; use the supplied local path to the licensed SemEval-2025 Task 7 release and complete the issue. After being asked about the GPU dependencies and >1 GB model download, the user explicitly approved: "Áno, over všetky štyri konfigurácie".
- **What the agent did:**
  - cloned `main` at `7c55503833ee5dab4ee1e601f2694fafffbce56d` into a separate directory and created `s1-1-reproduce`; kept the earlier issue-2 checkout untouched;
  - copied only the human-supplied raw release, verified all 28 source/copy file hashes, and downloaded the public sample afresh; created new environments and new model/embedding caches;
  - installed the pinned CUDA requirements and downloaded `intfloat/multilingual-e5-large` at revision `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3` with the human's approval; verified CUDA on the RTX 4060 Laptop;
  - ran the CPU baseline suite before changes (226 passed, 59 skipped), then the real E5 GPU sample test (1 passed); completed the four frozen test configurations using new YAMLs whose only changed setting is `run_name`; after the human stopped work on October 22, resumed the unfinished monolingual dense run on October 24 from its own 196,608 saved embeddings, with no previous predictions or metrics and no repeated completed test run;
  - ran BM25 in a separate freshly created CPU environment while the GPU installation completed; ran dense and final tests in the fresh CUDA `.venv`; recorded uv/interpreter/cache/temp deviations from the README;
  - stored aggregate metrics and result rows, generated full-precision comparison/provenance notes, and prepared a PR description and the issue's requested result-table comment;
  - clarified the Windows sample download, Python prerequisite, cold-cache timing, variable test counts/skip reasons and preservation of reference results in the README.
- **Output used:** four reproduction YAMLs, four aggregate `metrics.json` files, new rows in `results/results.md`, `results/reproduction_20261022.{md,json}`, README corrections and this appended log entry. No earlier log entries, reference results, loaders, retrieval code or evaluators were edited.
- **Verified:** both BM25 headline metrics match the committed references exactly; both dense metrics meet ±0.005. All four prediction files passed the independent official-logic S@10 check (`AGREE`). Final CUDA suite: **259 passed, 65 skipped**; remaining skips are inapplicable config checks, while the real GPU/model and all four saved-prediction checks ran. Original configuration values and metrics were compared with the source commit. Data/artifact and file-size guards passed before publication.
- **Limits / human review:** AI-assisted teammate reproduction, not an independent person outside the team. Human PR review and merging remain the team's responsibility. Dataset text, predictions, caches and model weights stay local and ignored.
- **Publication:** the human requested on October 24 that Codex prepare commands and leave pushing to them. No remote branch, PR or issue comment was created by this session. The local commit's author and committer were checked against the connected GitHub profile (`Jakub Spišák`, `jakub.spisak@student.tuke.sk`, login `jakub-spisak`); push and PR/comment drafts are supplied for the human.

## Session 6 — 2026-10-25 (issue #3)

### 20. Primary-source audit of the stage-1 literature and bibliography
- **Started by:** Jakub Spišák (GitHub `jakub-spisak`).
- **Tool / model:** OpenAI Codex desktop, GPT-6 family (exact runtime model identifier not exposed); PDF skill, web/ACL Anthology tools, Python/pypdf, Poppler and Tectonic 0.17.0 for verification.
- **User prompt (faithful summary):** pull the newest repository changes; work through stage-1 issues #3 → #4 → #2 with a separate PR after each; keep subsequent local work based on the preceding changes; ask for required permissions and use the human's account for commits. The human will perform pushes and create PRs.
- **What the agent did:**
  - obtained network permission, pulled `main` with fast-forward only to `2c4323ddf1420693447436d38a62daaf3b73f1a3` (merged issue-1 PR #17), verified the baseline suite, and created `s1-3-literature-check`; preserved the separate earlier issue-2 checkout;
  - checked relevant methods/result sections of Peng (overview), Pikuliak (MultiClaim), TIFIN India, RACAI, UniBuc-AE and QUST_NLP, with supporting checks of UWBa, Duluth, fact check AI and Shaar 2022; fetched their public PDFs and official ACL BibTeX exports into local working files;
  - corrected the literature notes and related-work statements: final vs experimental RACAI adaptations, UWBa's per-language final model choice, the scope/location of QUST_NLP's E5 text-variant ablations, TIFIN's translated-post pipeline, UniBuc-AE score fusion, and Shaar's political-debate context;
  - used the overview's published precision for team results, filled missing crosslingual scores, added the optional fact check AI paragraph, and softened unverified hardware/gain generalizations;
  - re-exported ten core bibliography entries while retaining citation keys; regenerated the report PDF; added source locations and human-reading questions/checklist in `docs/literature.md` and a separate local review guide;
  - appended this record and prepared local PR/comment/push texts, without publishing them.
- **Output used:** `docs/literature.md`, `docs/references.bib`, the relevant literature paragraphs in `report/stage1/main.tex`, regenerated `main.pdf`, and this log entry. Retrieval/evaluation code, experiment configs, stored metrics and earlier log entries were unchanged.
- **Verified:** baseline 259 passed / 65 inapplicable config skips; final suite 259 passed / 65 skipped (15.33 s), including the real GPU/model and local prediction checks. Tectonic compiled from cached packages; all 24 citation keys have bibliography entries, with no undefined references, missing-character warnings, replacement glyphs or `??`. All 11 rendered pages and enlarged bibliography pages were inspected. The existing small dense-paragraph overfull-box warning and the AI-appendix TODO belong to the remaining report/appendix work; no new experiment was run and no package or model was installed/downloaded.
- **Human confirmation / limits:** when asked about the issue's required human reading and source check, the human replied: "Zatiaľ nepotvrdzujem, prejdem podklady." Human reading remains unconfirmed; the prepared PR uses `Refs #3`, not automatic closure. This AI source audit is not represented as a human reading. Issue #4 and final issue #2 work remain separate.
