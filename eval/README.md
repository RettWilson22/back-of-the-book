# Evaluation question sets

| File | Questions | Purpose |
|---|---|---|
| `questions.jsonl` | 75 answerable, 20 off-topic | Main set. Written and committed before any retrieval run. |
| `heldout.jsonl` | 19 answerable, 15 off-topic | Written after the first run, to check the off-topic threshold on questions it wasn't chosen from. |
| `results.md`, `results-heldout.md` | | Output of `backofthebook eval`. |

## How the questions were written

- **Answerable questions:** pages were picked mechanically: one or two per section of the book (the page after the section starts, plus the middle page of longer sections). For each page, a question was written that the page answers, paraphrased so it doesn't reuse the page's distinctive wording. `gold` is the PDF page number (1-indexed). The printed page number is the PDF page minus 10.
- **Off-topic questions:** a mix of clearly unrelated questions (geography, cooking) and *near-domain* technical questions the book doesn't cover (Kubernetes, TCP, Dijkstra's algorithm), which are harder to detect.
- **Authorship:** the questions were drafted with an LLM from the extracted page text, then reviewed and edited by hand. They have not been validated by independent annotators.

## Known limitations

- Each question has a single gold page. When a passage continues onto the next page, a correct retrieval of that next page counts as a miss, so recall is a lower bound. Both remaining hybrid+rerank misses in `questions.jsonl` (q062, q065) are of this kind.
- 75 questions give roughly ±10 percentage points of uncertainty on recall figures, and the held-out set (19 questions) is much noisier. Treat small differences between methods as indicative, not conclusive.

## Attribution and license

The questions are based on *Principles of Data Science* (senior contributing authors Shaun V. Ault, Soohyun Nam Liao, and Larry Musolino), © 2025 Rice University / OpenStax, licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Book: <https://openstax.org/details/books/principles-data-science>.

As a derivative of that work, the files in this folder are also licensed under **CC BY-NC-SA 4.0**. The textbook PDF itself is not included in this repository; `scripts/download_corpus.py` downloads it from OpenStax.
