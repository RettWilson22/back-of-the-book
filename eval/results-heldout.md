Corpus: principles-of-data-science.pdf. Questions: 19 answerable, 15 off-topic.

| Retrieval | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | Median latency |
|---|---|---|---|---|---|---|
| bm25 | 63.2% | 84.2% | 89.5% | 94.7% | 0.759 | 1 ms |
| dense | 68.4% | 84.2% | 94.7% | 94.7% | 0.758 | 8 ms |
| hybrid | 68.4% | 89.5% | 94.7% | 94.7% | 0.782 | 10 ms |
| hybrid+rerank | 84.2% | 89.5% | 89.5% | 94.7% | 0.868 | 679 ms |

| Off-topic threshold | Answerable kept | Off-topic declined |
|---|---|---|
| 0.15 | 100.0% | 0.0% |
| 0.20 | 100.0% | 20.0% |
| 0.25 | 100.0% | 33.3% |
| 0.30 | 100.0% | 80.0% |
| 0.35 | 100.0% | 86.7% |
| 0.40 | 100.0% | 93.3% |
