# Literature — SemEval-2025 Task 7 (fact-checked claim retrieval)

Working notes for the stage-1 literature review. BibTeX for everything is in [`references.bib`](references.bib).

**How the entries were checked:**
- Issue #3 source audit: 2026-10-25, with Codex assistance. Relevant methods/results sections of the overview, MultiClaim and eight supporting papers were checked; this does not record a human reading confirmation. Selected BibTeX entries were re-exported from ACL Anthology, with citation keys preserved.
- Team scores and ranks below use Peng et al., Tables 8–9 (pp. 2503, 2505), at the published precision. System papers sometimes round differently; model selection and ablations come from each system paper. Original MultiClaim scores use a different evaluation set and are not SemEval leaderboard scores.

---

## 1. Task and dataset (must cite)

| Ref | What it is | Why it matters for us |
|---|---|---|
| **Peng et al. 2025**, SemEval-2025 Task 7 overview. [2025.semeval-1.323](https://aclanthology.org/2025.semeval-1.323/), pp. 2498–2511 | Task definition, data splits, the 4 organizer baselines, results of all teams, overview of the approaches | Defines the setup we reproduce; the source of all leaderboard and baseline numbers |
| **Pikuliak et al. 2023**, *Multilingual Previously Fact-Checked Claim Retrieval*. EMNLP 2023, [2023.emnlp-main.1027](https://aclanthology.org/2023.emnlp-main.1027/), pp. 16477–16500 | The MultiClaim dataset (≈28k posts in 27 languages, ≈206k fact-checks in 39 languages, ≈31k links) and an evaluation of unsupervised and supervised retrievers | The task data is derived from MultiClaim. Its main findings: GTR-T5-Large was a leading unsupervised model on the English-translated version (Table 2); MPNet-Base-Multilingual was the best on original-language text. Section 5 also tests supervised training |

Dataset: Zenodo [10.5281/zenodo.14989176](https://doi.org/10.5281/zenodo.14989176) (research use, access on request).

## 2. Prior work on retrieving previously fact-checked claims

| Ref | Contribution |
|---|---|
| **Shaar et al. 2020**, *That is a Known Lie: Detecting Previously Fact-Checked Claims*. ACL 2020, [2020.acl-main.332](https://aclanthology.org/2020.acl-main.332/), pp. 3607–3618 | Formulated the task and released a dataset. Uses BM25 plus BERT-based reranking with learning-to-rank, which is the classic retrieve-then-rerank setup |
| **Vo & Lee 2020**, *Where Are the Facts? Searching for Fact-checked Information to Alleviate the Spread of Fake News*. EMNLP 2020, [2020.emnlp-main.621](https://aclanthology.org/2020.emnlp-main.621/), pp. 7717–7731 | Search for fact-check articles from tweets, using text and images (multimodal). Relevant because our posts include image OCR |
| **Kazemi et al. 2021**, *Claim Matching Beyond English to Scale Global Fact-Checking*. ACL-IJCNLP 2021, [2021.acl-long.347](https://aclanthology.org/2021.acl-long.347/), pp. 4504–4517 | Multilingual claim matching on WhatsApp tipline data (hi, bn, ml, ta). Uses a distilled multilingual embedder and beats LASER and LaBSE |
| **Shaar et al. 2022**, *The Role of Context in Detecting Previously Fact-Checked Claims*. Findings of NAACL 2022, [2022.findings-naacl.122](https://aclanthology.org/2022.findings-naacl.122/), pp. 1619–1631 | Context around the input claim in political debate transcripts can improve retrieval by over 10 MAP points. This is not a direct experiment on social-media OCR; applying the idea to our posts is a separate hypothesis |

## 3. SemEval-2025 Task 7 system papers (existing solutions)

S@10 on test: the monolingual number is the average over languages, the crosslingual number is micro over posts. The two top teams, PINGAN AI (0.960 / 0.859) and PALI (0.947 / 0.827), published no system paper.

| Team (rank mono / cross) | S@10 mono / cross | Approach | Takeaway for us |
|---|---|---|---|
| **TIFIN India** (3 / 4) — Devadiga et al. 2025, [2025.semeval-1.299](https://aclanthology.org/2025.semeval-1.299/) | 0.9383 / 0.81025 | Aya-expanse-8B translations of posts; Stella-400M with hard negatives; Qwen2.5-72B reranks the top 50, followed by RRF with the retriever (§§3.2–3.6) | Table 5 separates translated/fine-tuned retrieval (0.9217) from the final pipeline. RTX 4090 timings in Table 3 do not establish feasibility on our 8 GB GPU |
| **RACAI** (4 / 3) — Chivereanu & Tufis 2025, [2025.semeval-1.77](https://aclanthology.org/2025.semeval-1.77/) | 0.9377 / 0.82450 | Original-language BGE-Multilingual-Gemma2-9B with contrastive LoRA fine-tuning, without reranking. Prompt tuning, QLoRA and Matryoshka variants are separate experiments (§4, Tables 2–3) | Keep the final LoRA submission distinct from the other adaptations. Its paper uses rounded test scores; the numbers here come from the overview |
| **QUST_NLP** (5 / 7) — Liu et al. 2025, [2025.semeval-1.114](https://aclanthology.org/2025.semeval-1.114/) | 0.9365 / 0.79250 | Select retrievers by language, rerank candidates, then combine reranker top-10 outputs with dev-based weights (§§2, 4) | E5 ablations support original+translation for mono and English-only for cross (§4.1, Tables 1–2). This is not a universal result for every model or language |
| **UniBuc-AE** (6 / 8) — Enache 2025, [2025.semeval-1.89](https://aclanthology.org/2025.semeval-1.89/) | 0.9336 / 0.79000 | English e5-large-v2 and original-language multilingual-e5-large-instruct, with contrastive training and hard-negative fine-tuning. Combine four configurations by weighted similarity scores (§3.5) | Hard negatives are sampled at ranks 8–10 to avoid overly similar false negatives (§3.4); weights are selected on validation data |
| **UWBa** (7 / 9) — Lenc et al. 2025, [2025.semeval-1.31](https://aclanthology.org/2025.semeval-1.31/) | 0.9270 / 0.78250 | Zero-shot English retrieval. Final mono uses NV-Embed-v2 alone or paired with GPT/Mistral depending on language; cross uses NV-Embed alone (§3.1, Table 1) | Five models were explored, but GTE/mGTE were dropped. The final system is not an ensemble of all five; model choice still uses dev data |
| **fact check AI** (10 / 5) — Rastogi 2025, [2025.semeval-1.306](https://aclanthology.org/2025.semeval-1.306/) | 0.9232 / 0.79750 | Fine-tuned bi-encoders with original/English text for mono, English translations for cross, and five-fold ensembling (§§3–4) | A further trained bi-encoder comparison. Its results do not isolate a gain caused by text cleaning alone |
| **Duluth** (23 / –) — Syed & Pedersen 2025, [2025.semeval-1.98](https://aclanthology.org/2025.semeval-1.98/) | 0.6883 / – | Final monolingual TF-IDF with word tokenization and a 15,000-feature vocabulary (§§5–6, Table 3) | A lexical reference on the same task; the paper rounds the test average to 0.69 |

Other Task 7 papers, worth skimming for the review, in the 2025.semeval-1 volume:

| Team | Anthology ID | Approach |
|---|---|---|
| DKE-Research | .22 | Language-specific adaptation |
| YNU-HPCC | .35 | |
| NCL-AR | .44 | Sieve filtering |
| ClaimCatchers | .63 | Sentence transformers |
| UMUTeam | .103 | XLM-R with self-alignment |
| ipezoTU | .154 | Hybrid ensemble |
| Word2winners | .183 | |
| CAISA | .184 | |
| CAIDAS | .215 | LLM-generated data augmentation |
| Howard University-AI4PC | .234 | |
| JU_NLP | .271 | |
| FactDebug | .284 | Hybrid sparse and dense retrieval |
| Shouth NLP | .295 | Contrastive learning |
| MultiMind | .303 | Multi-source alignment |

The proceedings index page lists some of these with links that are off by one, so check each ID on its own page before citing.

## 4. Methods we use (cite where used)

| Ref | Used for |
|---|---|
| **Robertson & Zaragoza 2009**, *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in IR 3(4):333–389 | The BM25 baseline (Okapi, k1 = 1.5, b = 0.75) |
| **Reimers & Gurevych 2019**, *Sentence-BERT*. EMNLP-IJCNLP 2019, [D19-1410](https://aclanthology.org/D19-1410/) | The bi-encoder paradigm and the sentence-transformers library |
| **Reimers & Gurevych 2020**, *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation*. EMNLP 2020, [2020.emnlp-main.365](https://aclanthology.org/2020.emnlp-main.365/) | paraphrase-multilingual-mpnet (an organizer baseline) |
| **Ni et al. 2022**, *Large Dual Encoders Are Generalizable Retrievers* (GTR). EMNLP 2022, [2022.emnlp-main.669](https://aclanthology.org/2022.emnlp-main.669/), pp. 9844–9855 | GTR-T5-Large (an organizer baseline; the strongest crosslingual baseline) |
| **Wang et al. 2022**, *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (E5). arXiv:2212.03533 | The E5 training recipe and the `query:`/`passage:` prefixes |
| **Wang et al. 2024**, *Multilingual E5 Text Embeddings: A Technical Report*. arXiv:2402.05672 | **Our dense baseline** (multilingual-e5-large) |
| **Chen et al. 2024**, *M3-Embedding* (BGE-M3). Findings of ACL 2024, pp. 2318–2335 | A candidate alternative embedder (stage 2) |
| **Cormack et al. 2009**, *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*. SIGIR 2009, pp. 758–759 | Fusing BM25 and dense rankings (stage 2) |
| **Nogueira & Cho 2019**, *Passage Re-ranking with BERT*. arXiv:1901.04085 | The cross-encoder reranking paradigm (stage 2) |
| **Phatthiyaphaibun et al. 2023**, *PyThaiNLP*. NLP-OSS 2023, [2023.nlposs-1.4](https://aclanthology.org/2023.nlposs-1.4/), pp. 25–36 | Thai word segmentation for BM25 |

## 5. Takeaways for the report

1. **The field converges on a pipeline:** most strong systems do dense retrieval, then reranking or ensembling. This motivates our roadmap: BM25 and dense, rank fusion (RRF), reranking and fine-tuning; improvements must be measured on dev.
2. **Original vs. English text depends on the track.** QUST_NLP reports this pattern in its E5 ablations (§4.1, Tables 1–2), not uniformly across all languages or all retrievers. We independently see the same thing on dev with mE5: monolingual 0.862 (both) vs 0.827 (English only); crosslingual 0.533 (both) vs 0.591 (English only). A good point for the interpretation section.
3. **Translation is a strong lever.** TIFIN and UWBa use English translations. Our supplied translations avoid another translation run, but TIFIN uses a different translation model, so its scores are not an isolated comparison of our preprocessing.
4. **Fine-tuning is a stage-2 hypothesis to test.** TIFIN and UniBuc-AE provide useful hard-negative ablations; RACAI uses contrastive LoRA training. Their results do not prove that our proposed training fits in 8 GB or will deliver the same gains; use the planned remote GPU and verify resource needs separately.
5. **Lexical baselines are underrated when tuned.** Duluth's TF-IDF reaches 0.69; our BM25 with stemming, stopwords and Thai segmentation reaches 0.868, above the organizers' BM25 (0.812). Tokenization matters: plain `\w+` breaks Thai text into fragments.
6. **Reference points (test S@10, monolingual / crosslingual):**

   | System | Monolingual | Crosslingual |
   |---|---|---|
   | Organizers' BM25 | 0.812 | 0.603 |
   | Organizers' mE5 | 0.859 | 0.637 |
   | Organizers' GTR | 0.730 | 0.705 |
   | **Our BM25** | **0.868** | **0.623** |
   | **Our mE5** | **0.886** | **0.681** |
   | Rank 10 | 0.923 | – |
   | Rank 1 | 0.960 | 0.859 |

## 6. Issue #3: source locations and human review

The checks below identify what to inspect in the primary papers. Page numbers are printed proceedings pages. This audit checks source support; the required human reading/review remains unconfirmed.

| Required paper | Primary location | Human review question |
|---|---|---|
| Peng et al. | §2–3; Tables 6, 8–9, pp. 2499–2505 | Why is mono averaged over languages, while cross uses one pool? Distinguish shared-task splits from original MultiClaim. |
| Pikuliak et al. | §2–5; Tables 2, 4, pp. 16478–16484 | Why are original-language and English-translated experiments different? Why are its scores not directly our SemEval test scores? |
| TIFIN India | §§3.2–3.6; Table 5, pp. 2298–2302 | Which stages separate the fine-tuned retriever from the final system? |
| RACAI | §4; Tables 2–3, pp. 552–555 | Which adaptation produced the final submission, and which techniques were separate experiments? |
| UniBuc-AE | §§3.4–3.5; Tables 1–4, pp. 636–638 | Why skip the highest-ranked hard negatives, and where are fusion weights selected? |
| QUST_NLP | §4.1; Tables 1–2, pp. 836–837 | Which E5 comparisons support the original/translation observation, and why is it not a universal rule? |

Additional related-work checks: UWBa §3.1/Table 1 (p. 211), Duluth §6/Table 3 (pp. 715–716), fact check AI §§3–5 (pp. 2353–2356), and Shaar et al. §5 (pp. 1623–1626).

Human confirmation for issue #3 (to be completed after reading):
- [ ] Overview and MultiClaim read; at least three of the four required system papers read.
- [ ] Related-work statements checked against those papers; remaining disagreements recorded.
- [ ] Reviewer name and review date supplied in the PR or issue comment.

No human confirmation is inferred from this AI-assisted audit. The full-text PDFs and official BibTeX exports were fetched from ACL Anthology on 2026-10-25; PDFs remain in local working files, not in the repository.
