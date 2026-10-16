# Literature — SemEval-2025 Task 7 (fact-checked claim retrieval)

Working notes for the stage-1 literature review. BibTeX for everything is in [`references.bib`](references.bib).

**How the entries were checked:**
- ACL Anthology entries: venue, pages and ID were checked against the Anthology page (2026-10-16). Re-export the BibTeX from the URL if you need the abstract field.
- Team scores: taken from the overview paper (Peng et al., 2025) unless the team's own paper is cited.

---

## 1. Task and dataset (must cite)

| Ref | What it is | Why it matters for us |
|---|---|---|
| **Peng et al. 2025**, SemEval-2025 Task 7 overview. [2025.semeval-1.323](https://aclanthology.org/2025.semeval-1.323/), pp. 2498–2511 | Task definition, data splits, the 4 organizer baselines, results of all teams, overview of the approaches | Defines the setup we reproduce; the source of all leaderboard and baseline numbers |
| **Pikuliak et al. 2023**, *Multilingual Previously Fact-Checked Claim Retrieval*. EMNLP 2023, [2023.emnlp-main.1027](https://aclanthology.org/2023.emnlp-main.1027/), pp. 16477–16500 | The MultiClaim dataset (≈28k posts in 27 languages, ≈206k fact-checks in 39 languages, ≈31k links) and an evaluation of unsupervised and supervised retrievers | The task data is derived from MultiClaim. Its main findings: GTR-T5-Large was the best model overall, and paraphrase-multilingual-MPNet was the best multilingual one |

Dataset: Zenodo [10.5281/zenodo.14989176](https://doi.org/10.5281/zenodo.14989176) (research use, access on request).

## 2. Prior work on retrieving previously fact-checked claims

| Ref | Contribution |
|---|---|
| **Shaar et al. 2020**, *That is a Known Lie: Detecting Previously Fact-Checked Claims*. ACL 2020, [2020.acl-main.332](https://aclanthology.org/2020.acl-main.332/), pp. 3607–3618 | Formulated the task and released a dataset. Uses BM25 plus BERT-based reranking with learning-to-rank, which is the classic retrieve-then-rerank setup |
| **Vo & Lee 2020**, *Where Are the Facts? Searching for Fact-checked Information to Alleviate the Spread of Fake News*. EMNLP 2020, [2020.emnlp-main.621](https://aclanthology.org/2020.emnlp-main.621/), pp. 7717–7731 | Search for fact-check articles from tweets, using text and images (multimodal). Relevant because our posts include image OCR |
| **Kazemi et al. 2021**, *Claim Matching Beyond English to Scale Global Fact-Checking*. ACL-IJCNLP 2021, [2021.acl-long.347](https://aclanthology.org/2021.acl-long.347/), pp. 4504–4517 | Multilingual claim matching on WhatsApp tipline data (hi, bn, ml, ta). Uses a distilled multilingual embedder and beats LASER and LaBSE |
| **Shaar et al. 2022**, *The Role of Context in Detecting Previously Fact-Checked Claims*. Findings of NAACL 2022, [2022.findings-naacl.122](https://aclanthology.org/2022.findings-naacl.122/), pp. 1619–1631 | Context on the source (post) side matters most, giving 10+ points. This motivates combining post text with OCR |

## 3. SemEval-2025 Task 7 system papers (existing solutions)

S@10 on test: the monolingual number is the average over languages, the crosslingual number is micro over posts. The two top teams, PINGAN AI (0.960 / 0.859) and PALI (0.947 / 0.827), published no system paper.

| Team (rank mono / cross) | S@10 mono / cross | Approach | Takeaway for us |
|---|---|---|---|
| **TIFIN India** (3 / 4) — Devadiga et al. 2025, [2025.semeval-1.299](https://aclanthology.org/2025.semeval-1.299/) | 0.938 / 0.810 | Translate everything to English with an LLM (Aya-expanse-8B). Fine-tune Stella-400M on the translations with hard negatives, then rerank the top 50 with Qwen2.5-72B. Combine with RRF | Translation plus a small embedder already reaches 0.92 before reranking. Runs on one RTX 4090 (translation took 61 min, fine-tuning 35 min). Only the 72B reranker would be out of reach for us |
| **RACAI** (4 / 3) — Chivereanu & Tufiș 2025, [2025.semeval-1.77](https://aclanthology.org/2025.semeval-1.77/) | 0.938 / 0.825 | BGE-Multilingual-Gemma2-9B on original text, with LoRA/QLoRA contrastive fine-tuning, prompt tuning and Matryoshka embeddings; no reranker | Fine-tuning helps the crosslingual track most. A 9B model needs QLoRA plus a remote GPU in our setup |
| **QUST_NLP** (5 / 7) — Liu et al. 2025, [2025.semeval-1.114](https://aclanthology.org/2025.semeval-1.114/) | 0.937 / – | Pick the best retriever per language, run 6 rerankers each giving a top 10, then weighted voting | **Combining original and English text helped monolingual but hurt crosslingual, which is exactly what we observe** (§5) |
| **UniBuc-AE** (6 / 8) — Enache 2025, [2025.semeval-1.89](https://aclanthology.org/2025.semeval-1.89/) | 0.934 / 0.79 | E5-family models with contrastive pre-training plus hard-negative fine-tuning, combining 4 models by weighted vote | The closest template to our planned stage-2 fine-tuning (mE5 plus hard negatives), at a feasible scale |
| **UWBa** (7 / 9) — Lenc et al. 2025, [2025.semeval-1.31](https://aclanthology.org/2025.semeval-1.31/) | 0.927 / – | Zero-shot: English translations plus 5 off-the-shelf embedders (NV-Embed-v2 best), with their top 5 lists merged | A strong zero-shot ensemble with no training. A useful comparison point for our zero-shot mE5 at 0.886 |
| **fact check AI** (10 / 5) — Rastogi 2025, [2025.semeval-1.306](https://aclanthology.org/2025.semeval-1.306/) | 0.923 / 0.798 | English text; fine-tunes mE5-large-instruct, Stella-400M and mxbai with K-fold training, then votes; text cleaning | Text cleaning (URLs, emoji, very short texts) is a cheap preprocessing idea |
| **Duluth** — Syed & Pedersen 2025, [2025.semeval-1.98](https://aclanthology.org/2025.semeval-1.98/) | 0.69 (mono, test) | TF-IDF only, with a tuned vocabulary size (15k features) | A lexical lower bound. Our tuned BM25 reaches 0.868 |

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
| **Nogueira & Cho 2019**, *Passage Re-ranking with BERT*. arXiv:1901.04085 | The cross-encoder reranking paradigm (stage 2) |\
| **Phatthiyaphaibun et al. 2023**, *PyThaiNLP*. NLP-OSS 2023, [2023.nlposs-1.4](https://aclanthology.org/2023.nlposs-1.4/), pp. 25–36 | Thai word segmentation for BM25 |

## 5. Takeaways for the report

1. **The field converges on a pipeline:** most strong systems do dense retrieval, then reranking or ensembling. This matches our roadmap: BM25 and dense, then rank fusion (RRF), then reranking, then fine-tuning.
2. **Original vs. English text depends on the track.** QUST_NLP reports that combining original and English text helps monolingual and hurts crosslingual. We independently see the same thing on dev with mE5: monolingual 0.862 (both) vs 0.827 (English only); crosslingual 0.533 (both) vs 0.591 (English only). A good point for the interpretation section.
3. **Translation is a strong lever.** TIFIN and UWBa both rely on English translations. The dataset already ships Google Translate output, so we get this for free.
4. **Fine-tuning with hard negatives** gives the largest gains in TIFIN, UniBuc-AE and RACAI, more so for crosslingual. This is the most promising stage-2 contribution within 8 GB of VRAM (mE5-large or a smaller model, with gradient checkpointing).
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
