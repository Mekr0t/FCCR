| run | model | variant | pool | split | track | S@10 | MRR@10 | macro S@10 |
|---|---|---|---|---|---|---|---|---|
| bm25_sample_eng | bm25 | eng | all | sample | - | 0.9574 | 0.7794 | 0.9649 |
| bm25_sample_orig | bm25 | orig | all | sample | - | 0.7021 | 0.5497 | 0.7495 |
| bm25_mono_dev_eng | bm25 | eng | per-language | dev | monolingual | 0.6847 | 0.4942 | 0.6847 |
| bm25_cross_dev_eng | bm25 | eng | all | dev | crosslingual | 0.4130 | 0.2469 | - |
| bm25_mono_dev_orig | bm25 | orig | per-language | dev | monolingual | 0.7003 | 0.5090 | 0.7003 |
| bm25_mono_test_eng | bm25 | eng | per-language | test | monolingual | 0.7714 | 0.6224 | 0.7714 |
| bm25_cross_test_eng | bm25 | eng | all | test | crosslingual | 0.5373 | 0.3391 | - |
| dense_sample_eng | mE5-large | eng | all | sample | - | 0.9574 | 0.9220 | 0.9649 |
| dense_mono_dev_eng | mE5-large | eng | per-language | dev | monolingual | 0.8272 | 0.6114 | 0.8272 |
| dense_mono_dev_orig | mE5-large | orig | per-language | dev | monolingual | 0.8205 | 0.5988 | 0.8205 |
| dense_mono_dev_both | mE5-large | both | per-language | dev | monolingual | 0.8617 | 0.6564 | 0.8617 |
| dense_cross_dev_eng | mE5-large | eng | all | dev | crosslingual | 0.5906 | 0.3441 | - |
| dense_cross_dev_orig | mE5-large | orig | all | dev | crosslingual | 0.5344 | 0.3302 | - |
| dense_mono_test_both | mE5-large | both | per-language | test | monolingual | 0.8862 | 0.7250 | 0.8862 |
| dense_cross_test_eng | mE5-large | eng | all | test | crosslingual | 0.6813 | 0.4166 | - |
| dense_cross_dev_both | mE5-large | both | all | dev | crosslingual | 0.5326 | 0.3382 | - |
| bm25_mono_dev_eng_stem | bm25 | eng | per-language | dev | monolingual | 0.7228 | 0.5230 | 0.7228 |
| bm25_mono_dev_eng_stem_stop | bm25 | eng | per-language | dev | monolingual | 0.7729 | 0.5697 | 0.7729 |
| bm25_mono_dev_orig_thai | bm25 | orig | per-language | dev | monolingual | 0.7152 | 0.5395 | 0.7152 |
| bm25_mono_dev_orig_stem_stop_thai | bm25 | orig | per-language | dev | monolingual | 0.7767 | 0.5794 | 0.7767 |
| bm25_mono_dev_both_stem_stop_thai | bm25 | both | per-language | dev | monolingual | 0.7930 | 0.5943 | 0.7930 |
| bm25_cross_dev_eng_stem_stop | bm25 | eng | all | dev | crosslingual | 0.5199 | 0.2931 | - |
| bm25_cross_dev_both_stem_stop_thai | bm25 | both | all | dev | crosslingual | 0.4891 | 0.3108 | - |
| bm25_mono_test_both_stem_stop_thai | bm25 | both | per-language | test | monolingual | 0.8682 | 0.7130 | 0.8682 |
| bm25_cross_test_eng_stem_stop | bm25 | eng | all | test | crosslingual | 0.6225 | 0.3825 | - |
