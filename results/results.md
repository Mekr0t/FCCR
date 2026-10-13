| run | model | variant | pool | split | track | S@10 | MRR@10 | macro S@10 |
|---|---|---|---|---|---|---|---|---|
| bm25_sample_eng | bm25 | eng | all | sample | - | 0.9574 | 0.7794 | 0.9649 |
| bm25_sample_orig | bm25 | orig | all | sample | - | 0.7021 | 0.5497 | 0.7495 |
| bm25_mono_dev_eng | bm25 | eng | per-language | dev | monolingual | 0.6847 | 0.4942 | 0.6847 |
| bm25_cross_dev_eng | bm25 | eng | all | dev | crosslingual | 0.4130 | 0.2469 | - |
| bm25_mono_dev_orig | bm25 | orig | per-language | dev | monolingual | 0.7003 | 0.5090 | 0.7003 |
| bm25_mono_test_eng | bm25 | eng | per-language | test | monolingual | 0.7714 | 0.6224 | 0.7714 |
| bm25_cross_test_eng | bm25 | eng | all | test | crosslingual | 0.5373 | 0.3391 | - |
