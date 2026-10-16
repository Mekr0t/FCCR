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
