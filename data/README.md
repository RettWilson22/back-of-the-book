# Sample data

`index/` is a prebuilt CoursePilot index of OpenStax *Principles of Data Science* (senior contributing authors Shaun V. Ault, Soohyun Nam Liao, and Larry Musolino), © 2025 Rice University / OpenStax, licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Book: <https://openstax.org/details/books/principles-data-science>.

It contains the book's text split into passages (`chunks.jsonl`) and their embeddings (`embeddings.npy`), so the demo starts instantly and the evaluation can be reproduced without downloading the PDF. As a derivative of the book, this folder is licensed under **CC BY-NC-SA 4.0**. Changes from the original: the text was extracted from the PDF, running headers and footers were removed, and the text was split into passages.

Rebuild it with:

```bash
python scripts/download_corpus.py
coursepilot --index data/index ingest data/corpus
```

`corpus/` holds the downloaded PDF and isn't committed.
