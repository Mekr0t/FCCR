# Stage-1 report review — issue #2

Technical review with Codex assistance, prepared for Jakub Spišák on 2026-10-25. This branch builds on the local issue-3 and issue-4 changes; it does not replace teammate review or claim that their outstanding human confirmations were supplied.

## Scope and numerical checks

The six report tables were compared at their displayed precision with `results/report_tables.md`, stored `metrics.json` files and `results/data_stats.json`/`data_stats.md`. The organizer scores were recalculated from their existing baseline predictions with the repository's scoring code. All 29 rows in `results/results.md` were compared with their own metric files. No retrieval experiment was rerun and no stored result was changed.

The local audit made 668 field comparisons: 595 numeric values and 73 intentionally missing fields. It found zero mismatches. The groups were 232 fields in the generated result Markdown, 172 report-table score fields, 177 report-table statistics/count fields and 87 run-index fields. Missing-field comparisons are included in those counts; they are not additional numeric measurements. The detailed audit stays in local working files.

Narrative numbers were also checked against those sources. Literature scores/ranks come from the primary papers, model settings from the saved configs/code, and the original GPU description from README. They are distinct from our measured evaluation results.

| Finding | Resolution and evidence |
|---|---|
| BM25's generic 5–25 s claim omitted the slower base crosslingual run | The original tuned test runs took 13.7302 s and 24.4326 s, while `bm25_cross_test_eng` took 52.3752 s (`timing.total_s`). The text now says approximately 14–25 s and 52 s, explicitly for the original stored runs and subject to hardware/cache state. |
| Published organizer scores were described as exactly reproduced | `results/report_tables.md` reports maximum absolute S@10 difference 0.00004. The report now states less than 1e-4 and describes the independent official-logic comparison separately. MRR has no official scorer counterpart. |
| Truncation percentage lacked a split/denominator | `dense_mono_test_both`: 478/4276 = 11.2%; `dense_mono_dev_both`: 219/1891 = 11.6%, from `embedding.posts.tokens.n_truncated`. |
| Approximately 11–21% multi-gold posts could be read as a per-language range | The range refers to aggregate track/split percentages (11.5, 14.3, 15.5, 20.5%). Some individual languages are outside it; the aggregate scope is now explicit. |
| OCR and length descriptions were imprecise | The text now states 54.4%/46.7% OCR presence in dev/test, 9.8% OCR-only in dev, FC medians 26/27 words, and mono-post p90 198/187 words. These are whitespace lengths of the English representation, not model token counts. |
| Original and reproduced runtimes/results could be confused | The report identifies its tables as original reference runs and links `results/reproduction_20261022.md`. The separate RTX 4060 reproduction met the fixed ±0.005 tolerance; it does not overwrite the reference numbers or claim an uninterrupted cold timing for the resumed mono dense run. |
| Portuguese/English ranking and E5 vs GTR comparison | Preserved the issue-3 corrections: Portuguese 0.802, then English 0.814; German 0.872 unchanged; English 0.818→0.814 and Portuguese 0.808→0.802; our E5 beats organizer E5 in both tracks but is below crosslingual GTR 0.705. |
| Unsupported approximate training-pair counts and universal improvement claims | Removed the untraced approximate counts; hard negatives are explicitly for training posts. Complementarity, gains from fine-tuning and causal explanations are proposed checks/hypotheses. |

## Claims and primary sources

The targeted system-paper checks use the locally retained ACL PDFs from the issue-3 source audit; the locations below identify the relevant evidence. Existing source-backed corrections and all bibliography keys were preserved.

| Source | Evidence checked and resulting wording |
|---|---|
| [Peng et al. overview](https://aclanthology.org/2025.semeval-1.323/) | Sections 2–3 and Tables 6, 8–9: task protocol, language averaging, team ranks and published precision. |
| [MultiClaim](https://aclanthology.org/2023.emnlp-main.1027/) | Section 4.1/Table 2: distinguish GTR on English translations from MPNet-Base-Multilingual on originals. The original dataset is not the SemEval test split. |
| [TIFIN](https://aclanthology.org/2025.semeval-1.299/) | Sections 3.2–3.6/Table 5: Aya translations of posts, Stella fine-tuning, top-50 Qwen reranking and RRF; 0.9217 is the translated/fine-tuned retriever, not the full system. |
| [RACAI](https://aclanthology.org/2025.semeval-1.77/) | Section 4/Tables 2–3: the final LoRA setup is separated from prompt tuning, QLoRA and Matryoshka experiments. |
| [QUST_NLP](https://aclanthology.org/2025.semeval-1.114/) | Section 4.1/Tables 1–2: the direction of the E5 original/translation ablations supports our comparison; it is not stated as a rule for every model/language or as proof of our causal explanation. |
| [UniBuc-AE](https://aclanthology.org/2025.semeval-1.89/) | Sections 3.4–3.5/Tables 1–4: hard negatives and four configurations fused through weighted similarity scores, weights selected on validation. |
| [UWBa](https://aclanthology.org/2025.semeval-1.31/) | Section 3.1/Table 1: final mono uses per-language NV-Embed alone or paired with GPT/Mistral; cross uses NV-Embed, not a five-model ensemble. |
| [Duluth](https://aclanthology.org/2025.semeval-1.98/) | Section 6/Table 3: final lexical word-token TF-IDF with 15,000 features, mono S@10 0.6883. |
| [fact check AI](https://aclanthology.org/2025.semeval-1.306/) | Sections 3–5: fine-tuned encoders, original/translated mono inputs, English cross inputs and five-fold combination. |
| [Shaar 2020](https://aclanthology.org/2020.acl-main.332/) | Sections 6.1–6.3: precise description of rankSVM over BM25/Sentence-BERT features; this is not a BERT cross-encoder trained as the final ranker. |
| [Shaar 2022](https://aclanthology.org/2022.findings-naacl.122/) | Section 5: context in political debates and MAP gains; not a direct OCR experiment. |

Definitions of [SBERT](https://aclanthology.org/D19-1410/), [multilingual distillation](https://aclanthology.org/2020.emnlp-main.365/), [E5](https://arxiv.org/abs/2212.03533), [multilingual E5](https://arxiv.org/html/2402.05672v1), [BGE-M3](https://aclanthology.org/2024.findings-acl.137/), [BERT reranking](https://arxiv.org/abs/1901.04085), [RRF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) and [PyThaiNLP](https://aclanthology.org/2023.nlposs-1.4/) were checked using primary publications/abstracts. The [author's E5 model card](https://huggingface.co/intfloat/multilingual-e5-large) confirms the query/passage prefixes, normalized embeddings and 512-token example. BM25 parameters 1.5/0.75 match the implementation defaults; no scoring implementation was changed.

## Assignment criteria and presentation

The human-supplied three-page `Projektove_zadanie.pdf` was read locally, including visual inspection of page 1. Its six stage-1 criteria map to the report as follows. The assignment PDF is not added to Git.

| Criterion | Report location and review result |
|---|---|
| Problem, motivation, objectives | Section 1: retrieval/ranking is distinguished from truth classification; objectives separate stages 1 and 2. |
| Literature, data and prior solutions | Sections 2–3, with primary citations and the paper checks above. Human reading required by issue #3 remains separately unconfirmed. |
| Metrics and methodology | Section 4: S@10, MRR@10, mono language macro vs cross micro, dev-only selection and fixed test/reproduction protocol. |
| Implemented, reproducible baseline | Sections 5 and 8, configs/metrics, README and the completed issue-1 reproduction record. |
| Report quality and citations | Slovak proofreading and the author's name corrected; consistent OCR/model descriptions; all six tables referenced and captions scoped; all 24 citation keys resolve. There are no figures. |
| LLM/AI documentation | Appendix A from issue #4, extended with this review record. Tools, purposes, preserved prompts/summaries and relevant outputs are present; remaining team/archival confirmations are explicitly tracked in `docs/ai_usage_confirmation.md`. |

The unused TODO macro was removed. The old translated record 18 now explicitly describes the placeholder as historical, without suggesting the current appendix is absent. This does not alter the original historical log entry. Hypotheses remain marked as unverified. The two prior typography warnings in the report body were eliminated.

## Remaining human actions

- Confirm the paper reading/source review required by issue #3.
- Complete the AI-use completeness and private-transcript archive confirmations tracked for issue #4; append actual information and regenerate the PDF if those confirmations change its content.
- Read the whole final report as required by the assignment and obtain teammate PR review. This technical audit does not stand in for that sign-off.
- Publish the prepared issue comment/PR and merge in order #3 → #4 → #2. The user performs all GitHub publication.

Assignment source SHA-256 (local evidence): `5051f5843152bb9d229a86d43d5b2443b6b11bb1facde52a7b567dd475d7aed9`.

## Final validation

- Baseline: 259 passed / 65 skipped, 17.87 s. Final suite: 259 passed / 65 skipped, 17.13 s; all skips are inapplicable configuration checks. Cached GPU/model and four saved-prediction checks ran.
- Tectonic 0.17.0 compiled from cache with no TeX warnings. All 25 PDF pages were visually inspected; 24 citation keys and all six table references resolve, with no TODO or missing glyphs.
- Historical primary AI records, evaluation/retrieval/data-loading code, configs, metrics and bibliography are unchanged. Only report/review/AI documentation and the regenerated PDF changed.
