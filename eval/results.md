Corpus: principles-of-data-science.pdf. Questions: 75 answerable, 20 off-topic.

| Retrieval | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | Median latency |
|---|---|---|---|---|---|---|
| bm25 | 69.3% | 84.0% | 85.3% | 92.0% | 0.766 | 1 ms |
| dense | 72.0% | 85.3% | 94.7% | 97.3% | 0.810 | 8 ms |
| hybrid | 74.7% | 89.3% | 89.3% | 93.3% | 0.817 | 10 ms |
| hybrid+rerank | 76.0% | 89.3% | 93.3% | 97.3% | 0.834 | 564 ms |

| Off-topic threshold | Answerable kept | Off-topic declined |
|---|---|---|
| 0.15 | 100.0% | 0.0% |
| 0.20 | 100.0% | 20.0% |
| 0.25 | 100.0% | 65.0% |
| 0.30 | 100.0% | 85.0% |
| 0.35 | 100.0% | 95.0% |
| 0.40 | 100.0% | 95.0% |
