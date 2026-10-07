# CoursePilot

[![CI](https://github.com/RettWilson22/coursepilot/actions/workflows/ci.yml/badge.svg)](https://github.com/RettWilson22/coursepilot/actions/workflows/ci.yml)

Ask questions about your course materials and get answers that cite the exact page they came from. Generate practice quizzes from the same materials, or use **AnyQuiz** to get a quiz on any topic at all, at **easy, medium, or hard** difficulty. And see how accurate the retrieval actually is, measured on a fixed question set instead of assumed.

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

## AnyQuiz: a quiz on any topic

Type any topic (*Super Mario Galaxy*, *the French Revolution*, *photosynthesis*), pick **Easy**, **Medium**, or **Hard**, and the AI writes a multiple-choice quiz from its own knowledge. Take it in the app and get graded, with an explanation for every answer.

| Difficulty | What the questions test |
|---|---|
| Easy | Recall of basic facts and definitions; wrong choices are clearly wrong |
| Medium | Understanding and application; plausible wrong choices |
| Hard | Multi-step reasoning and edge cases; wrong choices based on common misconceptions |

**Double-check.** A quiz written from memory can have a wrong answer key. That's worse than useless when you're studying. So after writing the quiz, AnyQuiz runs a second pass that answers every question **without seeing the key**. Questions where the two passes disagree are left out, and the app says how many. The quiz asks for a couple of spare questions so you still get the number you asked for. You can turn the check off for a faster quiz.

AnyQuiz is clearly labeled as written from general knowledge, separate from the **Course quiz**, which only uses your materials and cites a source for every question. If you ask the course quiz about something your materials don't cover, it declines and points you to AnyQuiz.

## Error codes

Every failure is a `CoursePilotError` ([`errors.py`](src/coursepilot/errors.py)) with a stable **code**, a **message** that's safe to show users, a **retryable** flag, and optional machine-readable **details** (such as the provider's HTTP status). The web app shows the message and the code. The CLI prints `Error [CODE]: message` and exits with the category's exit code.

| Code | When | Retryable | CLI exit |
|---|---|---|---|
| `EMPTY_TOPIC` | No topic given | no | 2 |
| `TOPIC_TOO_LONG` | Topic over 200 characters | no | 2 |
| `INVALID_QUESTION_COUNT` | Not 1–10 questions | no | 2 |
| `INVALID_DIFFICULTY` | Not easy / medium / hard | no | 2 |
| `TOPIC_NOT_COVERED` | Course quiz topic isn't in the loaded materials | no | 3 |
| `NO_LLM_CONFIGURED` | No `GROQ_API_KEY` or `ANTHROPIC_API_KEY` | no | 4 |
| `LLM_AUTH_FAILED` | API key rejected | no | 4 |
| `LLM_RATE_LIMITED` | Provider rate limit | yes | 5 |
| `LLM_UNAVAILABLE` | Network error or provider 5xx | yes | 5 |
| `LLM_REQUEST_REJECTED` | Provider 4xx (e.g. invalid model) | no | 5 |
| `LLM_REFUSED` | Model declined the request | no | 5 |
| `LLM_BAD_RESPONSE` | Malformed or truncated response, even after one repair attempt | yes | 5 |
| `NO_VALID_QUESTIONS` | Nothing usable survived validation and the double-check | yes | 6 |
| `INTERNAL_ERROR` | An unexpected bug (logged with a traceback) | no | 1 |

Invalid input is rejected **before** any API call. A test checks that every code has a catalog entry.

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
 quiz.py    course quiz: same retrieval → structured JSON quiz → each question validated
            (4 distinct choices, valid answer, cites a real passage) or dropped
            AnyQuiz: topic + difficulty → quiz from model knowledge → double-check pass
```

| Module | Responsibility |
|---|---|
| [`documents.py`](src/coursepilot/documents.py) | Loaders for each file type. Detects the printed page number from running heads, so a citation says "p. 283" like the book does, not the PDF's 293. |
| [`chunking.py`](src/coursepilot/chunking.py) | Sentence-aware chunking with overlap, scoped to one page. |
| [`index.py`](src/coursepilot/index.py) | Builds, saves, and loads the index (`chunks.jsonl`, `embeddings.npy`, `meta.json`). No database server. |
| [`retrieval.py`](src/coursepilot/retrieval.py) | Four retrieval modes behind one interface, so the eval compares them on identical inputs. |
| [`llm.py`](src/coursepilot/llm.py) | Claude, Groq, and offline providers behind one small interface. |
| [`answer.py`](src/coursepilot/answer.py) | Off-topic check, prompt, streaming, citation validation. |
| [`quiz.py`](src/coursepilot/quiz.py) | Course quiz and AnyQuiz, difficulty levels, validation, double-check. |
| [`errors.py`](src/coursepilot/errors.py) | Error codes, messages, retryability, and CLI exit codes. |
| [`evaluation.py`](src/coursepilot/evaluation.py) | Recall@k, MRR, and off-topic metrics. |
| [`app/streamlit_app.py`](app/streamlit_app.py) | Web UI: chat with sources, course quiz, AnyQuiz, grading, and a plain-language About page. |

## Design decisions

- **Citations are checked, not trusted.** The model can only cite `[S1]`–`[Sn]`. Anything else is reported as an invalid citation instead of being shown as a source.
- **Decline before generating.** An off-topic question never reaches the LLM. That's cheaper, and the model has no chance to answer from its own knowledge.
- **Chunks never cross pages.** Slightly less context per chunk, but every passage maps to exactly one page. That's what makes page-level citations and page-level evaluation possible.
- **Retrieval is evaluated without an LLM.** The metrics are deterministic, free to rerun, and isolate the part of the system most responsible for wrong answers.
- **Quizzes use structured output.** On Claude, the response must match a JSON schema (`output_format`). On Groq, JSON mode plus Pydantic validation, with one automatic repair attempt. Invalid questions are dropped, not shown.
- **Claude requests opt into server-side refusal fallback** (`fallbacks: "default"`) and check `stop_reason` before reading output.

## Try it

**Live demo: https://coursepilot-rettwilson.streamlit.app**

It's preloaded with the sample textbook. Ask a question, or open **Practice quiz** and pick a topic. You can also upload your own slides or notes in the sidebar; they stay in your session only.

## Getting started

```bash
git clone https://github.com/RettWilson22/coursepilot && cd coursepilot
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[groq,claude,app,dev]"
```

The repo ships a prebuilt index of the sample textbook (`data/index`, 3.7 MB), so there's nothing to download or build before you start.

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
coursepilot quiz "hypothesis testing" -n 5 -d hard       # from your materials
coursepilot anyquiz "Super Mario Galaxy" -d easy          # any topic
coursepilot eval                                         # reproduce the table above, in seconds
```

Use your own materials with `coursepilot ingest path/to/slides/`, or upload files in the web app's sidebar.

| Setting | Default |
|---|---|
| `COURSEPILOT_LLM` | `groq` if `GROQ_API_KEY` is set, else `claude` if `ANTHROPIC_API_KEY` is set, else `extractive` |
| `COURSEPILOT_GROQ_MODEL` | `openai/gpt-oss-120b` |
| `COURSEPILOT_CLAUDE_MODEL` / `COURSEPILOT_CLAUDE_EFFORT` | `claude-opus-5-5` / `medium` |
| `COURSEPILOT_INDEX` (web app) | your `.coursepilot/index` if you've run `ingest`, else the bundled `data/index` |

### Deploy your own (Streamlit Community Cloud, free)

1. Fork this repo, then at [share.streamlit.io](https://share.streamlit.io) choose **Create app → Deploy a public app from GitHub**.
2. Repository: your fork, branch `main`, main file `app/streamlit_app.py`. Under **Advanced settings**, pick Python 3.12 and add `GROQ_API_KEY = "..."` as a secret.
3. Deploy. The bundled index loads at startup, so there's no setup step. Peak memory is about 0.6 GB.

## Testing

```bash
pytest                 # 118 tests, about 15 s, no downloads or API keys needed (fake models and LLM)
pytest -m slow         # real embedding/reranking models; includes a guard that fails
                       # if textbook Recall@10 drops below 95% or MRR below 0.80
ruff check . && mypy src app/streamlit_app.py
```

The fast tests cover every module (96% line coverage), every error code, the CLI end to end including exit codes, and the web app through Streamlit's `AppTest`: asking, declining, taking and grading both kinds of quiz, and how errors are shown. CI runs lint, strict type checking, and tests on every push.

## Limitations

- **Live LLM calls aren't part of the test suite.** Claude and Groq are tested against their SDK interfaces with fake clients (exact request parameters, error handling, refusals). Answer *quality* hasn't been evaluated, only retrieval.
- **AnyQuiz's double-check reduces wrong answer keys but doesn't eliminate them.** If the model is confidently wrong both times, the error survives. Its accuracy hasn't been measured.
- **Scanned PDFs need OCR**, which isn't included. Image-only pages are skipped.
- **Printed page detection** looks for headers/footers like "12 • Chapter title". Documents without them fall back to PDF page numbers.
- **Evaluation scope:** one textbook, AI-drafted questions reviewed by one person, single-page labels. See [`eval/README.md`](eval/README.md).

## License

Code: MIT. Evaluation questions and the bundled sample index: CC BY-NC-SA 4.0, as derivatives of the OpenStax textbook ([eval](eval/README.md), [data](data/README.md)). The textbook PDF itself isn't redistributed here.
