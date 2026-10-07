# CoursePilot

Ask questions about your course materials and get answers that cite the exact page they came from. Generate practice quizzes from the same materials. And see how accurate the retrieval actually is, measured on a fixed question set instead of assumed.

Works with PDFs (slides, notes, textbooks), PowerPoint decks, Markdown, and plain text. Answers come from **Groq** (free tier) or **Claude**, or from an offline mode that quotes the matching passages with no API key at all.

## Results

Retrieval accuracy on [OpenStax *Principles of Data Science*](https://openstax.org/details/books/principles-data-science) (561 pages, 1,479 passages). Each question is labeled with the page that answers it, and a hit counts only if a retrieved passage comes from that page. The question set was [committed before any retrieval run](eval/README.md).

| Retrieval | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Median latency |
|---|---|---|---|---|---|
| Keyword (BM25) | 69.3% | 85.3% | 92.0% | 0.766 | 1 ms |
| Semantic (embeddings) | 72.0% | 94.7% | 97.3% | 0.810 | 8 ms |
| Hybrid (BM25 + embeddings, RRF) | 74.7% | 89.3% | 93.3% | 0.817 | 10 ms |
| **Hybrid + cross-encoder rerank** (default) | **76.0%** | 93.3% | **97.3%** | **0.834** | 564 ms |

75 questions, two per section of the book. Full tables, including Recall@3: [`eval/results.md`](eval/results.md), held-out set: [`eval/results-heldout.md`](eval/results-heldout.md).

**What the numbers say**
- Reranking gives the best top-1 accuracy and MRR, at the cost of about half a second per query on a laptop CPU.
- Plain hybrid search is *worse* than embeddings alone at Recall@10. BM25 pulls in passages that share words but not meaning. Fusion only paid off once a reranker re-scored the candidates.
- Both remaining reranker misses (out of 75) are next-page continuations of the labeled page, not wrong answers. The strict single-page metric counts them as misses anyway; I didn't relax it after seeing results.

**Declining off-topic questions.** Before calling the LLM, CoursePilot checks how close the best passage is to the question and declines if nothing in the materials is close. The threshold was set on the main question set and then checked on a [held-out set](eval/heldout.jsonl) written afterwards:

| Threshold | Main set: real questions kept / off-topic declined | Held-out: kept / declined |
|---|---|---|
| 0.25 (original guess) | 100% / 65% | 100% / 33% |
| **0.35 (default)** | **100% / 95%** | **100% / 87%** |

Off-topic questions that still get through: "SOLID principles", "public-key cryptography" (the book discusses data security), and, oddly, "Who painted the Mona Lisa?". For those, the prompt tells the model to answer only from the provided passages and say when they don't cover the question.

> These are small evaluation sets: about ±10 points of uncertainty on 75 questions, more on the held-out set. Read differences of a few points as indicative.

## How it works

```
 PDF / PPTX / MD / TXT
        │  documents.py   extract text per page; detect printed page numbers;
        ▼                 strip repeated headers/footers
      pages
        │  chunking.py    sentence-aligned ~180-word chunks, 40-word overlap,
        ▼                 never crossing a page (so every chunk has one citation)
      chunks ──► index.py   embeddings (all-MiniLM-L6-v2) saved as plain files
        │
 question ─► retrieval.py   BM25 + embeddings → reciprocal rank fusion
        │                   → cross-encoder rerank (ms-marco-MiniLM-L6-v2)
        ▼
 answer.py  off-topic check → prompt with labeled passages [S1]..[Sn]
        │   → LLM streams an answer → citations checked against the passages
        ▼
 quiz.py    same retrieval → structured JSON quiz → each question validated
            (4 distinct choices, valid answer, cites a real passage) or dropped
```

| Module | Responsibility |
|---|---|
| [`documents.py`](src/coursepilot/documents.py) | Loaders for each file type. Detects the printed page number from running heads, so a citation says "p. 283" like the book does, not the PDF's 293. |
| [`chunking.py`](src/coursepilot/chunking.py) | Sentence-aware chunking with overlap, scoped to one page. |
| [`index.py`](src/coursepilot/index.py) | Builds, saves, and loads the index (`chunks.jsonl`, `embeddings.npy`, `meta.json`). No database server. |
| [`retrieval.py`](src/coursepilot/retrieval.py) | Four retrieval modes behind one interface, so the eval compares them on identical inputs. |
| [`llm.py`](src/coursepilot/llm.py) | Claude, Groq, and offline providers behind one small interface. |
| [`answer.py`](src/coursepilot/answer.py) | Off-topic check, prompt, streaming, citation validation. |
| [`quiz.py`](src/coursepilot/quiz.py) | Quiz generation and validation. |
| [`evaluation.py`](src/coursepilot/evaluation.py) | Recall@k, MRR, and off-topic metrics. |
| [`app/streamlit_app.py`](app/streamlit_app.py) | Web UI: chat with sources, interactive quiz with grading, eval results. |

## Design decisions

- **Citations are checked, not trusted.** The model can only cite `[S1]`–`[Sn]`. Anything else is reported as an invalid citation instead of being shown as a source.
- **Decline before generating.** An off-topic question never reaches the LLM. That's cheaper, and the model has no chance to answer from its own knowledge.
- **Chunks never cross pages.** Slightly less context per chunk, but every passage maps to exactly one page. That's what makes page-level citations and page-level evaluation possible.
- **Retrieval is evaluated without an LLM.** The metrics are deterministic, free to rerun, and isolate the part of the system most responsible for wrong answers.
- **Quizzes use structured output.** On Claude, the response must match a JSON schema (`output_format`). On Groq, JSON mode plus Pydantic validation, with one automatic repair attempt. Invalid questions are dropped, not shown.
- **Claude requests opt into server-side refusal fallback** (`fallbacks: "default"`) and check `stop_reason` before reading output.

## Getting started

```bash
git clone https://github.com/RettWilson22/coursepilot && cd coursepilot
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[groq,claude,app,dev]"

python scripts/download_corpus.py           # sample textbook from OpenStax (checksum-verified)
coursepilot ingest data/corpus               # about a minute on a laptop
```

Pick an LLM (or skip this for offline, quote-only answers):

```bash
export GROQ_API_KEY=...          # free at console.groq.com
# or
export ANTHROPIC_API_KEY=...     # uses claude-opus-5-5
```

Then:

```bash
streamlit run app/streamlit_app.py                       # web app
coursepilot ask "Why can k-means give different clusters on different runs?"
coursepilot quiz "hypothesis testing" -n 5
coursepilot eval --output eval/results.md                # reproduce the table above
```

Use your own materials with `coursepilot ingest path/to/slides/`, or upload files in the web app's sidebar.

| Setting | Default |
|---|---|
| `COURSEPILOT_LLM` | `groq` if `GROQ_API_KEY` is set, else `claude` if `ANTHROPIC_API_KEY` is set, else `extractive` |
| `COURSEPILOT_GROQ_MODEL` | `openai/gpt-oss-120b` |
| `COURSEPILOT_CLAUDE_MODEL` / `COURSEPILOT_CLAUDE_EFFORT` | `claude-opus-5-5` / `medium` |
| `COURSEPILOT_INDEX` (web app) | `.coursepilot/index` |

## Testing

```bash
pytest                 # 78 tests, about 15 s, no downloads or API keys needed (fake models and LLM)
pytest -m slow         # real embedding/reranking models; includes a guard that fails
                       # if textbook Recall@10 drops below 95% or MRR below 0.80
ruff check . && mypy src app/streamlit_app.py
```

The fast tests cover every module (93% line coverage), the CLI end to end, and the web app through Streamlit's `AppTest`: asking, declining, and taking and grading a quiz. CI runs lint, strict type checking, and tests on every push.

## Limitations

- **Live LLM calls aren't part of the test suite.** Claude and Groq are tested against their SDK interfaces with fake clients (exact request parameters, error handling, refusals). Answer *quality* hasn't been evaluated, only retrieval.
- **Scanned PDFs need OCR**, which isn't included. Image-only pages are skipped.
- **Printed page detection** looks for headers/footers like "12 • Chapter title". Documents without them fall back to PDF page numbers.
- **Evaluation scope:** one textbook, AI-drafted questions reviewed by one person, single-page labels. See [`eval/README.md`](eval/README.md).

## License

Code: MIT. Evaluation questions: CC BY-NC-SA 4.0, as a derivative of the OpenStax textbook ([details](eval/README.md)). The textbook isn't redistributed here.
