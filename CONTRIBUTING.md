# Contributing — team rules

Rules for everyone working on this repo. They exist for three reasons: the **dataset licence**, the **grading criteria** (methodology, reproducibility, documented AI use), and avoiding painful merge conflicts.
If a rule blocks you, raise it in an issue, don't just skip it.

---

## 1. Data and licence — non-negotiable

- **Never commit the dataset or anything derived from it that contains text**: CSVs, parsed pickles/parquet, prediction files, error-analysis exports, screenshots of posts. `data/`, `cache/` and `results/*/predictions.json` are gitignored, so don't force-add them (`git add -f`).
- **Never upload the data anywhere public**: no public Colab notebooks with outputs, no public HF datasets/Spaces, no GitHub gists, no chat tools that train on inputs.
- Committed results contain **aggregates only**: counts, percentages, metrics, ids. When the report needs an example, paraphrase it; don't quote the data.
- On Colab, keep the data in your **private** Google Drive. Clear notebook outputs that show data text before committing a notebook.
- Everyone needs their own access via [Zenodo](https://doi.org/10.5281/zenodo.14989176). Don't share the files with people outside the team, including the person reproducing our results.

## 2. Experiments and evaluation

- **Tune on dev only** (`train_dev_sets`, `posts_dev`). Every decision (model, variant, hyperparameter, threshold) is made on dev.
- **Test runs only for a frozen configuration**, once per track. Don't look at test results to choose between variants. If you ran something on test by mistake, say so in the issue.
- **Every experiment runs from a config** in `configs/` and writes `results/<run_name>/metrics.json` plus a row in `results/results.md`. No numbers from notebooks or ad-hoc scripts go into the report.
- **Naming:** `<method>_<track>_<split>_<variant>[_<features>]`, e.g. `bm25_mono_dev_both_stem_stop_thai`, `dense_cross_dev_eng`.
- **Compare against the current best on dev.** If a change hurts, report it anyway; negative results go into the ablation table, not the bin.
- **Scoring:** always through `src/evaluate.py`. For prediction files, also cross-check with `scripts/score_official.py`. Monolingual = mean over languages, crosslingual = micro.
- **Training data:** only `posts_train`. Dev and test posts must never appear in training data, hard-negative mining or threshold fitting.
- **Fix seeds** and log the model name, text variant (`orig`/`eng`/`both`) and pool in every run (the runners do this; keep it that way in new ones).
- **Expensive artifacts:** embeddings go to `cache/<model>/...` (one directory per model and variant; never overwrite another model's cache). Model weights go to Drive or the Hugging Face hub (private), never to git.

## 3. Code

- Python 3.13, `.venv`, dependencies pinned in `requirements.txt` (`==`). New package means a pinned line in the same PR.
- Reuse `src/data.py` (loading, splits, text construction) and `src/evaluate.py`. **Don't write a second loader or metric.**
- New components get a **unit test** and a **smoke test on the sample data** before running on the full set.
- `python -m pytest -q` must pass before you open a PR.
- Match the existing style: small functions, a docstring on public functions, comments only where the *why* isn't obvious.

## 4. Git workflow

- **Don't push directly to `main`** (except trivial doc typos). Work on a branch named after the issue: `s2-1-fusion`, `s1-2-report-review`.
- **One issue → one branch → one PR.** Link it with `Closes #<n>` in the PR description.
- **Every PR needs one review** from a teammate who didn't write it. The reviewer runs the tests and at least one changed config.
- **Commit messages:** imperative, specific (`Add RRF fusion with configurable k and weights`, not `fix`, `update`, `wip`).
- **Never** `git push --force` to `main`, and never rewrite pushed history. The only exception is moving a submission tag (`stage1`, `stage2`), and only the report owner does that.
- Pull/rebase before starting work; resolve conflicts locally, don't push broken merges.
- Before pushing, check `git status` for anything under `data/`, `cache/` or any `*.csv` / `*.pkl` / `predictions.json`. If something slipped in, **tell the team immediately**: removing data from history requires rewriting it.

## 5. Issues

- Every task is an issue with **Steps** (checklist) and **Done when** (a checkable criterion), a label (`stage-1`, `stage-2`, `bonus`, `experiment`, `report`, `review`) and a milestone.
- Assign yourself before you start, so two people don't do the same work.
- Post results as a comment (a dev table, not "it works"), then close via the PR.
- Blocked for more than 2 days? Comment in the issue and tell the team.

## 6. Use of LLMs and AI tools — mandatory documentation

The assignment requires **every** use of LLMs or AI agents to be documented in an appendix. Undocumented or misleading use costs points.

- Log every use in [`docs/llm_usage.md`](docs/llm_usage.md), **the same day**, as a numbered entry with:
  - the **tool and model** (e.g. ChatGPT GPT-x, Claude Code with model Y, GitHub Copilot, DeepL);
  - the **purpose** (code, debugging, text drafting, translation, literature search);
  - the **prompt** (verbatim, or a faithful summary for long agent sessions; keep the full transcript);
  - **what was used** from the output, and how it was **verified** (tests, judge or review, manual check).
- This covers code, report text, translations, figures, issue texts and literature summaries.
- **You are responsible for anything AI-generated that you commit.** Read it, run it, check the numbers and citations against the sources.

## 7. Report

- Source in `report/stageN/` (LaTeX, Slovak). Bibliography in [`docs/references.bib`](docs/references.bib): add entries by exporting from the ACL Anthology / publisher, keep the citation keys stable.
- **Every number in the report must come from a file in `results/`.** Tables are generated by `scripts/make_report_tables.py` or copied from its output, never typed from memory.
- Unverified explanations are written as hypotheses ("pravdepodobne", "zatiaľ neoverené") until an experiment confirms them.
- Build with `tectonic main.tex` (see README) and commit the PDF together with its source changes.
- Everyone reads the whole report before submission: the assignment requires every member to understand the whole project.

## 8. Deadlines

| Milestone | Deadline | Freeze |
|---|---|---|
| Stage 1: proposal + baseline | 2026-10-25 | code frozen ~10-22, then writing only |
| Stage 2: own solution + evaluation | 2026-12-06 | final test run ~12-01, then writing only |

Submitted versions are tagged (`stage1`, `stage2`) so graders see exactly what was handed in.
