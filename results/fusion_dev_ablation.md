# S2-1 RRF fusion, dev ablation

Generated from `results/rrf_*_dev_*/metrics.json` (48 runs, dev only, nothing on test). Inputs are the top-100 `predictions.json` of existing runs; weights: dense 1.0, every BM25 input w (`wb`). Monolingual S@10/MRR are macro means over languages, crosslingual are micro over posts. S@20/50/100 are candidate recall for the later reranker (same aggregation).

Input sets. Mono A = bm25_mono_dev_both_stem_stop_thai + dense_mono_dev_both; mono B = A + bm25_mono_dev_eng_stem_stop. Cross A = bm25_cross_dev_eng_stem_stop + dense_cross_dev_eng; cross B = A + bm25_cross_dev_both_stem_stop_thai.

## Best configuration per track (chosen on dev S@10, tie-break MRR)

| track | best run | S@10 | MRR@10 | S@20 | S@50 | S@100 | reference | ref S@10 | ref MRR | ref S@100 | delta S@10 | delta MRR | delta S@100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mono | rrf_mono_dev_B_k10_wb0.3 | 0.8833 | 0.6687 | 0.9094 | 0.9345 | 0.9507 | dense_mono_dev_both | 0.8617 | 0.6564 | 0.9412 | +0.0216 | +0.0123 | +0.0095 |
| cross | rrf_cross_dev_B_k30_wb0.5 | 0.6504 | 0.4050 | 0.6957 | 0.7591 | 0.7899 | dense_cross_dev_eng | 0.5906 | 0.3441 | 0.7554 | +0.0598 | +0.0609 | +0.0345 |

Cross-check: `scripts/score_official.py` on both best prediction files (dev split dir) gives S@10 0.883302 (mono, all 8 per-language scores identical) and 0.650362 (cross), AGREE with `src/evaluate.py`.

Per-language S@10, best mono config vs dense_mono_dev_both:

| lang | n posts | dense_mono_dev_both | RRF best | delta |
|---|---|---|---|---|
| ara | 78 | 0.8590 | 0.8974 | +0.0385 |
| deu | 83 | 0.8193 | 0.8554 | +0.0361 |
| eng | 478 | 0.7950 | 0.8159 | +0.0209 |
| fra | 188 | 0.8670 | 0.8883 | +0.0213 |
| msa | 105 | 0.8952 | 0.8952 | +0.0000 |
| por | 302 | 0.8377 | 0.8609 | +0.0232 |
| spa | 615 | 0.8683 | 0.9008 | +0.0325 |
| tha | 42 | 0.9524 | 0.9524 | +0.0000 |

## Interpretation

Fusion beats the best single dense run in 20 of 24 monolingual and all 24 crosslingual configurations, so the gain does not hinge on one lucky setting; the best configurations add +2.2 points (mono) and +6.0 points (cross) S@10, and also raise candidate recall (S@100 +1.0 and +3.5 points), which is the number a later reranker depends on. A BM25 weight below 1 works better than equal weights in both tracks (mono mean S@10 over the grid 0.877 / 0.876 / 0.863 for wb 0.3 / 0.5 / 1.0; cross 0.632 / 0.640 / 0.629), plausibly because BM25 is the weaker ranker (reference S@10 well below dense) and an equal vote lets its errors displace correct dense hits; this is a hypothesis, not tested. The extra BM25 input helps crosslingual (set B mean 0.636 vs A 0.631) but not monolingual (B 0.866 vs A 0.878; the top single run is B at 0.8833 against 0.8824 for the best A run, a difference of about one post-equivalent, so it should not be read as a real preference), which could mean that the two monolingual BM25 variants are too correlated to add information when both are weighted. The grid winner for mono sits on the grid edge (k = 10, wb = 0.3) and the k and wb differences between neighbouring cells are mostly a few tenths of a point on 1891 (mono) / 552 (cross) posts, so the choice of k and exact weight is weakly determined; k has a small effect overall (mono means 0.874 for k = 10 to 0.870 for k = 100). Languages with few dev posts (tha 42, ara 78) show the largest swings and no gain for msa/tha, so per-language deltas should be treated as noisy. The best mono configuration was selected on the same dev set it is reported on, so its +2.2 points is slightly optimistic; the test split has not been touched.

## Monolingual (reference dense_mono_dev_both: S@10 0.8617, MRR 0.6564)

| run | set | k | wb | S@10 | MRR@10 | S@20 | S@50 | S@100 |
|---|---|---|---|---|---|---|---|---|
| rrf_mono_dev_A_k100_wb0.3 | A | 100 | 0.3 | 0.8807 | 0.6676 | 0.9033 | 0.9309 | 0.9412 |
| rrf_mono_dev_A_k100_wb0.5 | A | 100 | 0.5 | 0.8791 | 0.6655 | 0.9095 | 0.9305 | 0.9412 |
| rrf_mono_dev_A_k100_wb1.0 | A | 100 | 1.0 | 0.8731 | 0.6641 | 0.9044 | 0.9346 | 0.9500 |
| rrf_mono_dev_A_k10_wb0.3 | A | 10 | 0.3 | 0.8730 | 0.6697 | 0.8977 | 0.9297 | 0.9476 |
| rrf_mono_dev_A_k10_wb0.5 | A | 10 | 0.5 | 0.8824 | 0.6696 | 0.9071 | 0.9365 | 0.9485 |
| rrf_mono_dev_A_k10_wb1.0 | A | 10 | 1.0 | 0.8781 | 0.6680 | 0.9075 | 0.9346 | 0.9500 |
| rrf_mono_dev_A_k30_wb0.3 | A | 30 | 0.3 | 0.8753 | 0.6702 | 0.8989 | 0.9267 | 0.9466 |
| rrf_mono_dev_A_k30_wb0.5 | A | 30 | 0.5 | 0.8788 | 0.6693 | 0.9100 | 0.9339 | 0.9466 |
| rrf_mono_dev_A_k30_wb1.0 | A | 30 | 1.0 | 0.8817 | 0.6665 | 0.9090 | 0.9346 | 0.9500 |
| rrf_mono_dev_A_k60_wb0.3 | A | 60 | 0.3 | 0.8758 | 0.6694 | 0.9023 | 0.9315 | 0.9412 |
| rrf_mono_dev_A_k60_wb0.5 | A | 60 | 0.5 | 0.8807 | 0.6670 | 0.9095 | 0.9309 | 0.9494 |
| rrf_mono_dev_A_k60_wb1.0 | A | 60 | 1.0 | 0.8778 | 0.6653 | 0.9075 | 0.9346 | 0.9500 |
| rrf_mono_dev_B_k100_wb0.3 | B | 100 | 0.3 | 0.8696 | 0.6589 | 0.9057 | 0.9307 | 0.9475 |
| rrf_mono_dev_B_k100_wb0.5 | B | 100 | 0.5 | 0.8662 | 0.6520 | 0.9009 | 0.9331 | 0.9535 |
| rrf_mono_dev_B_k100_wb1.0 | B | 100 | 1.0 | 0.8504 | 0.6435 | 0.8722 | 0.9057 | 0.9364 |
| rrf_mono_dev_B_k10_wb0.3 | B | 10 | 0.3 | 0.8833 | 0.6687 | 0.9094 | 0.9345 | 0.9507 |
| rrf_mono_dev_B_k10_wb0.5 | B | 10 | 0.5 | 0.8745 | 0.6627 | 0.9062 | 0.9337 | 0.9519 |
| rrf_mono_dev_B_k10_wb1.0 | B | 10 | 1.0 | 0.8511 | 0.6364 | 0.8875 | 0.9344 | 0.9474 |
| rrf_mono_dev_B_k30_wb0.3 | B | 30 | 0.3 | 0.8800 | 0.6640 | 0.9073 | 0.9354 | 0.9508 |
| rrf_mono_dev_B_k30_wb0.5 | B | 30 | 0.5 | 0.8747 | 0.6562 | 0.9082 | 0.9365 | 0.9538 |
| rrf_mono_dev_B_k30_wb1.0 | B | 30 | 1.0 | 0.8463 | 0.6417 | 0.8734 | 0.9270 | 0.9469 |
| rrf_mono_dev_B_k60_wb0.3 | B | 60 | 0.3 | 0.8788 | 0.6619 | 0.9076 | 0.9324 | 0.9449 |
| rrf_mono_dev_B_k60_wb0.5 | B | 60 | 0.5 | 0.8731 | 0.6543 | 0.9061 | 0.9331 | 0.9538 |
| rrf_mono_dev_B_k60_wb1.0 | B | 60 | 1.0 | 0.8481 | 0.6424 | 0.8731 | 0.9067 | 0.9445 |

## Crosslingual (reference dense_cross_dev_eng: S@10 0.5906, MRR 0.3441)

| run | set | k | wb | S@10 | MRR@10 | S@20 | S@50 | S@100 |
|---|---|---|---|---|---|---|---|---|
| rrf_cross_dev_A_k100_wb0.3 | A | 100 | 0.3 | 0.6359 | 0.3921 | 0.6793 | 0.7301 | 0.7554 |
| rrf_cross_dev_A_k100_wb0.5 | A | 100 | 0.5 | 0.6359 | 0.3917 | 0.6812 | 0.7301 | 0.7554 |
| rrf_cross_dev_A_k100_wb1.0 | A | 100 | 1.0 | 0.6322 | 0.3887 | 0.6830 | 0.7355 | 0.7736 |
| rrf_cross_dev_A_k10_wb0.3 | A | 10 | 0.3 | 0.6141 | 0.3681 | 0.6775 | 0.7301 | 0.7754 |
| rrf_cross_dev_A_k10_wb0.5 | A | 10 | 0.5 | 0.6304 | 0.3807 | 0.6884 | 0.7373 | 0.7772 |
| rrf_cross_dev_A_k10_wb1.0 | A | 10 | 1.0 | 0.6304 | 0.3847 | 0.6775 | 0.7337 | 0.7736 |
| rrf_cross_dev_A_k30_wb0.3 | A | 30 | 0.3 | 0.6232 | 0.3853 | 0.6757 | 0.7301 | 0.7627 |
| rrf_cross_dev_A_k30_wb0.5 | A | 30 | 0.5 | 0.6322 | 0.3932 | 0.6830 | 0.7319 | 0.7754 |
| rrf_cross_dev_A_k30_wb1.0 | A | 30 | 1.0 | 0.6413 | 0.3874 | 0.6812 | 0.7355 | 0.7736 |
| rrf_cross_dev_A_k60_wb0.3 | A | 60 | 0.3 | 0.6286 | 0.3904 | 0.6775 | 0.7301 | 0.7554 |
| rrf_cross_dev_A_k60_wb0.5 | A | 60 | 0.5 | 0.6322 | 0.3915 | 0.6793 | 0.7301 | 0.7681 |
| rrf_cross_dev_A_k60_wb1.0 | A | 60 | 1.0 | 0.6359 | 0.3878 | 0.6830 | 0.7355 | 0.7736 |
| rrf_cross_dev_B_k100_wb0.3 | B | 100 | 0.3 | 0.6413 | 0.4019 | 0.6866 | 0.7391 | 0.7772 |
| rrf_cross_dev_B_k100_wb0.5 | B | 100 | 0.5 | 0.6467 | 0.4032 | 0.6957 | 0.7554 | 0.7935 |
| rrf_cross_dev_B_k100_wb1.0 | B | 100 | 1.0 | 0.6232 | 0.3995 | 0.6703 | 0.7174 | 0.7736 |
| rrf_cross_dev_B_k10_wb0.3 | B | 10 | 0.3 | 0.6377 | 0.3909 | 0.6902 | 0.7554 | 0.7917 |
| rrf_cross_dev_B_k10_wb0.5 | B | 10 | 0.5 | 0.6413 | 0.4049 | 0.6920 | 0.7554 | 0.7917 |
| rrf_cross_dev_B_k10_wb1.0 | B | 10 | 1.0 | 0.6286 | 0.3829 | 0.6920 | 0.7391 | 0.7772 |
| rrf_cross_dev_B_k30_wb0.3 | B | 30 | 0.3 | 0.6395 | 0.4036 | 0.6902 | 0.7518 | 0.7917 |
| rrf_cross_dev_B_k30_wb0.5 | B | 30 | 0.5 | 0.6504 | 0.4050 | 0.6957 | 0.7591 | 0.7899 |
| rrf_cross_dev_B_k30_wb1.0 | B | 30 | 1.0 | 0.6178 | 0.3922 | 0.6739 | 0.7409 | 0.7790 |
| rrf_cross_dev_B_k60_wb0.3 | B | 60 | 0.3 | 0.6377 | 0.4009 | 0.6884 | 0.7446 | 0.7844 |
| rrf_cross_dev_B_k60_wb0.5 | B | 60 | 0.5 | 0.6467 | 0.4041 | 0.6975 | 0.7591 | 0.7917 |
| rrf_cross_dev_B_k60_wb1.0 | B | 60 | 1.0 | 0.6232 | 0.3966 | 0.6721 | 0.7174 | 0.7754 |
