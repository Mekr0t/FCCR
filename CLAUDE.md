# CLAUDE.md — SemEval-2025 Task 7: Multilingual & Crosslingual Fact-Checked Claim Retrieval

## What this project is
Given a social media post (SMP), retrieve and rank the most relevant previously fact-checked claims (FC) from a large pool. This is **retrieval/ranking**, not true/false classification.

- **Monolingual track:** post and its fact-check share a language (e.g. pol→pol). Retrieval pool = fact-checks in that language.
- **Crosslingual track:** post and fact-check are in different languages (usually X→eng). Retrieval pool = all fact-checks.
- Data: MultiClaim-derived (≈200k fact-checks, ≈28k posts, 27 languages). Sample data + `load.py`: https://github.com/kinit-sk/semeval_2025
- Task page: https://disai.eu/semeval-2025/ · Codabench: https://www.codabench.org/competitions/3737/
- The official competition ended in 2025. We evaluate **locally** on dev/held-out splits unless told otherwise.

## Assignment specifics (fill in)
- Track(s): **monolingual first** (primary dev loop); run crosslingual alongside each step (same pipeline + cached embeddings), move to it fully once mono is solid.
- Goal: **own system** (not replicating a specific team). Course: NLP project (`Projektove_zadanie.pdf`), team of 3–4, 20 pts in two stages:
  - **Stage 1 (10 pts):** problem + motivation, literature/data/existing-solutions review, metrics + methodology, **reproducible baseline**, report quality with citations.
  - **Stage 2 (10 pts):** own contribution (new solution or substantial justified extension of baseline), experiments, correct evaluation, comparison with baseline + interpretation, limitations, reproducibility.
  - Bonus (≤15) chosen: (1) systematic error analysis with a predefined taxonomy + two independent annotators + agreement + a fix confirmed on held-out data; (2) independent reproducibility by someone outside the team; (3) deployment: simple public demo (query → matching fact-checks with score + link to the public fact-check URL, never dataset text; embeddings kept private), tested by 2–3 outside users with a structured form, one documented feedback→fix iteration, online ~1 week during grading then taken down.
- Compute: fine-tuning runs on Google Colab (T4, 16 GB, fp16 only — no bf16); laptop for inference/embedding.
- Deliverable: code + written report (per stage). **Deadlines: Stage 1 — 2026-10-25; Stage 2 — 2026-12-06.**
- Stage 1 scope (freeze by ~10-22, leave the rest for writing): BM25 + zero-shot mE5 baselines on dev + test, literature review, methodology. Stage 2 scope: fusion → reranking → fine-tuning, ablations, error analysis.
- Data source + split: official Zenodo release; develop/tune on dev (`posts_dev`), report final numbers on test.
- **Mandatory: every use of LLMs/AI agents goes in a separate appendix** (tool/model, purpose, prompts, relevant outputs; undocumented use costs points). Keep `docs/llm_usage.md` up to date: append an entry for every AI-assisted step in the same session, before finishing.

## Official dataset (Zenodo 10.5281/zenodo.14989176) — verified facts
Location: `data/raw/semeval2025_task7_data/` with `train_dev_sets/`, `test_set/`, `scripts/` (official loaders + scorers), `baselines/` (organizers' test predictions, top-100 per post).
- Splits and pools come from `tasks.json`: `monolingual[lang] = {fact_checks, posts_train, posts_dev}` (test: `posts_test`), `crosslingual = {fact_checks, posts_train, posts_dev}`. The `fact_checks` list **is** the retrieval pool for that task. Mono train/dev langs: ara deu eng fra msa por spa tha; mono test adds pol tur (10 langs). Crosslingual pool: 153,743 (dev) / 272,447 (test).
- Pairs files have only `post_id, fact_check_id` — **no `pair_lang`**. Post language for scoring = the `tasks.json` monolingual key.
- Gold: `train_dev_sets/{mono,cross}lingual_reference.json` = dev gold; `test_set/...reference.json` = test gold. Format `{"<post_id str>": [fc_id int, ...]}`.
- **Prediction format the official scorer expects:** JSON `{"<post_id as str>": [fc_id as int, ...]}`, exactly the reference's post set, **≥10 ids per post**. String fc ids silently score 0.
- Official scoring: monolingual = per-language S@10 then mean over languages; crosslingual = single micro S@10 over all posts.
- Organizers' baselines (test S@10, mono avg / cross): BM25 0.8123 / 0.60275 · mE5-large 0.8589 / 0.63725 · GTR-T5-large 0.7299 / 0.70525 · mpnet-multi-v2 0.5683 / 0.41075.
- Test set is for final reporting and baseline replication only; tune on dev.

## Data format — read carefully
Text fields are stored as **stringified Python tuples**, not plain strings. Always parse with the official `load.py` (or `ast.literal_eval`), never with naive string ops.

- Translated text tuple: `(original_text, english_translation, [(lang_code, confidence), ...])`
- `posts.csv`: `post_id`, `instances` (timestamps + platform), `ocr` (list of translated-text tuples from attached images), `verdicts` (Meta labels), `text` (translated-text tuple or empty)
- `fact_checks.csv`: `fact_check_id`, `claim` (translated-text tuple), `instances` (timestamps + URLs), `title` (translated-text tuple, may be empty)
- Sample data only: `trial_data_mapping.csv` with `fact_check_id`, `post_id`, `pair_lang` (e.g. `spa-eng`). The official release uses `pairs*.csv` without `pair_lang` (see above).
- A post may have **multiple** correct fact-checks. A post may have empty `text` and only `ocr`, or vice versa — handle both.
- Language codes are ISO 639-3 (`eng`, `deu`, `pol`, `tha`, `msa`, `hbs`, ...).

## Evaluation
- Primary metric: **Success@10** — fraction of posts with ≥1 gold FC in the top 10. Also report **MRR**.
- Report per-language scores plus the average (the leaderboard averages across languages, not across posts).
- Reference points (monolingual, S@10 avg): 1st place 0.960, ~10th 0.923, ~20th 0.867. English is consistently the hardest language.
- Keep one evaluation function in `src/evaluate.py` and use it everywhere. Never tune on the test split.
- Prediction format: `{post_id: [fact_check_id, ...]}` (ranked, top 10) saved as JSON. Verify against the Codabench guidelines before any official submission.

## Planned pipeline (build incrementally, measure after every step)
1. **Text construction:** post = text + OCR; FC = claim + title. Keep both original and English versions available.
2. **Baseline retrieval:** BM25 on English translations. Record S@10/MRR. This is the floor every change must beat.
3. **Dense retrieval:** multilingual embedder (start with `BAAI/bge-m3` or `intfloat/multilingual-e5-large`; e5 needs `query: ` / `passage: ` prefixes). Try original vs. English vs. both.
4. **Fusion:** reciprocal rank fusion (k=60) over BM25 + dense runs → top 50–100 candidates.
5. **Reranking:** cross-encoder (e.g. `BAAI/bge-reranker-v2-m3`) or a small LLM on the candidates.
6. **Optional:** contrastive fine-tuning of the embedder on train pairs with hard negatives mined from step 3.

Do not jump to steps 5–6 before steps 2–4 are working and logged.

## Hardware constraints
- Local GPU: RTX 5070 laptop, **8 GB VRAM**, Blackwell (sm_120). Requires a PyTorch build with CUDA 12.8+ (`cu128` wheels); older wheels fail with "no kernel image".
- Embed the fact-check pool **once**, in batches with fp16, and cache embeddings to disk (`cache/` as `.npy` + id list). Never re-embed the 200k pool per experiment.
- Use FAISS (or plain numpy matmul on normalized vectors) for search.
- Rerankers/LLMs: keep batch sizes small; prefer ≤4B models or quantized ones. Fine-tuning large embedders may need gradient checkpointing or a remote GPU — flag this before starting such a job.

## Repo layout
```
data/raw/          # original CSVs — never modified, gitignored
data/processed/    # parsed parquet/jsonl
cache/             # embeddings, BM25 indexes — gitignored
src/
  data.py          # loading + parsing (wraps official load.py logic)
  retrieval/       # bm25.py, dense.py, fusion.py
  rerank/
  evaluate.py      # S@K, MRR, per-language breakdown
scripts/           # runnable experiment entry points
configs/           # one YAML per experiment
results/           # metrics JSON per run + results.md summary table
```

## Working conventions
- Python 3.13 (`.venv`), dependencies in `requirements.txt`. Pin versions once something works.
- Every experiment is run from a config and writes `results/<run_name>/metrics.json` (S@10, MRR, per-language) plus the config used. Append one row to `results/results.md` per run.
- Fix random seeds. Log model name, text variant (orig/eng/both), and pool restriction for every run.
- Smoke-test any new component on the sample data before running on the full set.
- Don't commit data, caches, or model weights. The dataset has license restrictions — never upload it anywhere.
- When a change hurts metrics, say so plainly and keep the previous best as the reference; don't quietly drop the comparison.

## When helping me
- Prefer small, verifiable steps: write the code, run it on sample data, show the numbers.
- Explain retrieval/IR design choices briefly when they're non-obvious; skip basics.
- If something in the data contradicts these notes, trust the data and update this file.
