"""Back of the Book web app: ask questions with citations, take practice quizzes, see the eval.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import logging
import os
import tempfile
import urllib.parse
from collections.abc import Callable
from pathlib import Path

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from backofthebook.answer import AnswerEngine
from backofthebook.chunking import chunk_pages
from backofthebook.documents import SUPPORTED_SUFFIXES, load_document
from backofthebook.errors import BackOfTheBookError, ErrorCode
from backofthebook.index import (
    CorpusIndex,
    Embedder,
    IndexBuildError,
    SentenceTransformerEmbedder,
    default_index_dir,
)
from backofthebook.llm import make_provider
from backofthebook.quiz import (
    MAX_QUESTIONS,
    MAX_TOPIC_CHARS,
    Difficulty,
    Quiz,
    QuizQuestion,
    generate_anyquiz,
    generate_quiz,
)
from backofthebook.retrieval import CrossEncoderReranker, Hit, Retriever
from backofthebook.sample import build_sample_index

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = Path(os.environ.get("BACKOFTHEBOOK_INDEX") or default_index_dir(ROOT))
# The repo ships a prebuilt sample index, so this only runs if it was deleted (0 to skip).
USE_SAMPLE = os.environ.get("BACKOFTHEBOOK_SAMPLE", "1") != "0"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(
    page_title="Back of the Book",
    page_icon=str(ROOT / "app" / "static" / "favicon.png"),
    layout="wide",
)
logger = logging.getLogger("backofthebook.app")

STYLE = """
<style>
/* Hide Streamlit's own chrome so the page reads like a normal website. */
header[data-testid="stHeader"], footer, [data-testid="stToolbar"] { display: none; }
.block-container { padding-top: 0; max-width: 1100px; }

.bb-masthead {
  background: #22313f; color: #f6f3ec; margin: 0 -100vw 1.4rem; padding: 1.1rem 100vw 1rem;
  border-bottom: 4px solid #2a6496;
}
.bb-brand { font-family: Georgia, "Times New Roman", serif; font-size: 2rem; font-weight: bold;
  letter-spacing: 0.5px; }
.bb-tagline { font-size: 0.95rem; color: #c9d3dd; margin-top: 0.1rem; }

/* Classic rectangular tabs with a rule underneath. */
[data-testid="stTabs"] [role="tablist"] { gap: 4px; border-bottom: 1px solid #cfc6b4; }
[data-testid="stTab"] { background: #ebe5d8; border: 1px solid #cfc6b4; border-bottom: none;
  padding: 0.45rem 1.1rem; margin-bottom: -1px; }
[data-testid="stTab"] p { font-weight: 600; }
[data-testid="stTab"][aria-selected="true"] { background: #f6f3ec; color: #2a6496;
  box-shadow: inset 0 3px 0 #2a6496; }

.bb-mark { display: inline-block; padding: 0 0.45rem; border: 1px solid; font-size: 0.8rem;
  font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px; margin-right: 0.4rem; }
.bb-right { color: #2e6b30; background: #e6f2e4; border-color: #9cc49a; }
.bb-wrong { color: #8c2a24; background: #f7e4e1; border-color: #d9a29b; }

.bb-footer { margin-top: 3rem; padding: 1rem 0; border-top: 1px solid #cfc6b4;
  font-size: 0.85rem; color: #6b6458; }
</style>
"""

MASTHEAD = """
<div class="bb-masthead">
  <div class="bb-brand">Back of the Book</div>
  <div class="bb-tagline">Answers with sources, practice quizzes, and a quiz on anything.</div>
</div>
"""

FOOTER = """
<div class="bb-footer">
  Back of the Book, built by Rett Wilson &middot;
  <a href="https://github.com/RettWilson22/back-of-the-book">Source code on GitHub</a> &middot;
  Sample textbook: OpenStax <i>Principles of Data Science</i> (CC BY-NC-SA 4.0) &middot;
  AnyQuiz text from Wikipedia (CC BY-SA 4.0)
</div>
"""

st.markdown(STYLE, unsafe_allow_html=True)
st.markdown(MASTHEAD, unsafe_allow_html=True)

# Errors caused by what the user typed are shown as information, not as failures.
_INPUT_ERRORS = {
    ErrorCode.EMPTY_TOPIC,
    ErrorCode.TOPIC_TOO_LONG,
    ErrorCode.INVALID_QUESTION_COUNT,
    ErrorCode.INVALID_DIFFICULTY,
    ErrorCode.TOPIC_NOT_COVERED,
    ErrorCode.SOURCE_NOT_FOUND,
}


@st.cache_resource(show_spinner="Loading models...")
def load_models(model_name: str) -> tuple[SentenceTransformerEmbedder, CrossEncoderReranker]:
    return SentenceTransformerEmbedder(model_name), CrossEncoderReranker()


@st.cache_resource(show_spinner=False)
def build_sample(index_dir: Path) -> str | None:
    """Build the sample index once per server process; concurrent sessions wait for it."""
    try:
        build_sample_index(ROOT / "data" / "corpus", index_dir, SentenceTransformerEmbedder())
    except Exception as e:  # e.g. no network; the app still works with uploads
        return str(e)
    return None


@st.cache_resource(show_spinner="Loading course index...")
def load_base_index() -> CorpusIndex | None:
    try:
        return CorpusIndex.load(INDEX_DIR)
    except IndexBuildError:
        return None


def add_uploads(
    index: CorpusIndex | None, files: list[UploadedFile], embedder: Embedder
) -> CorpusIndex | None:
    """Index uploaded files under their original names so citations stay readable."""
    pages = []
    with tempfile.TemporaryDirectory() as tmp:
        for upload in files:
            path = Path(tmp) / upload.name
            path.write_bytes(upload.getvalue())
            try:
                pages.extend(load_document(path))
            except Exception as e:  # a bad upload shouldn't take down the app
                st.sidebar.error(f"Couldn't read {upload.name}: {e}")
    chunks = chunk_pages(pages)
    if not chunks:
        return index
    if index is None:
        return CorpusIndex.build(chunks, embedder)
    return index.add(chunks, embedder)


def render_sources(numbered_hits: list[tuple[int, Hit]]) -> None:
    for n, hit in numbered_hits:
        with st.expander(f"[S{n}] {hit.chunk.citation}"):
            st.write(hit.chunk.text)


# --- Sidebar: materials and settings -----------------------------------------------------

st.sidebar.header("Your materials")
if USE_SAMPLE and not (INDEX_DIR / "meta.json").exists():
    with st.spinner(
        "First start: downloading the sample textbook and indexing it. This takes about a minute."
    ):
        if error := build_sample(INDEX_DIR):
            st.warning(f"Couldn't set up the sample textbook ({error}). You can upload files.")
base_index = load_base_index()
embedder, reranker = load_models(base_index.embedding_model if base_index else DEFAULT_MODEL)

uploads = st.sidebar.file_uploader(
    "Add your own course materials",
    type=[s.lstrip(".") for s in SUPPORTED_SUFFIXES],
    accept_multiple_files=True,
    help="PDF slides or notes, PowerPoint decks, Markdown, or text. Stays in this session only.",
)
upload_key = tuple(sorted((u.name, u.size) for u in uploads or []))
if st.session_state.get("upload_key") != upload_key:
    with st.spinner("Indexing your files..."):
        st.session_state.index = add_uploads(base_index, uploads or [], embedder)
    st.session_state.upload_key = upload_key
index: CorpusIndex | None = st.session_state.index

if index is None:
    st.info(
        "No course materials loaded yet. Upload files in the sidebar, or build the sample "
        "index with `python scripts/download_corpus.py && backofthebook ingest data/corpus`."
    )
    st.stop()

st.sidebar.caption(
    f"{len(index.chunks):,} passages from {len(index.sources)} file(s): " + ", ".join(index.sources)
)

provider_names = {
    "Auto": None,
    "Groq": "groq",
    "Claude": "claude",
    "No LLM (quotes only)": "extractive",
}
choice = st.sidebar.selectbox("Answer with", list(provider_names), index=0)
try:
    llm = make_provider(provider_names[choice])
except Exception as e:
    st.sidebar.error(f"Couldn't start {choice}: {e}")
    llm = make_provider("extractive")
st.sidebar.caption(f"Using: **{llm.name}**")

retriever = Retriever(index, embedder, reranker)
engine = AnswerEngine(retriever, llm)


def show_error(error: BackOfTheBookError) -> None:
    hint = " Trying again may work." if error.retryable else ""
    (st.info if error.code in _INPUT_ERRORS else st.error)(error.message + hint)
    st.caption(f"Error code: `{error.code}`")


def guarded(action: Callable[[], Quiz]) -> Quiz | None:
    """Run a quiz request, turning every failure into a coded, user-facing message."""
    try:
        return action()
    except BackOfTheBookError as e:
        logger.info("quiz request failed: %r", e)
        show_error(e)
    except Exception:
        logger.exception("unexpected error while generating a quiz")
        show_error(
            BackOfTheBookError(
                ErrorCode.INTERNAL_ERROR, "Something went wrong on our side. Please try again."
            )
        )
    return None


REPORT_URL = "https://github.com/RettWilson22/back-of-the-book/issues/new"


def report_link(quiz: Quiz, question: QuizQuestion) -> str:
    """A pre-filled GitHub issue for reporting a wrong or unclear question."""
    choices = "\n".join(f"- {c}" for c in question.choices)
    sources = "\n".join(f"- {p.citation} {p.url or ''}" for p in quiz.sources_for(question))
    body = (
        f"**Topic:** {quiz.topic} ({quiz.difficulty.value})\n\n"
        f"**Question:** {question.question}\n\n**Choices:**\n{choices}\n\n"
        f"**Answer key:** {question.choices[question.answer_index]}\n\n"
        f"**Sources:**\n{sources}\n\n**What's wrong:** "
    )
    query = urllib.parse.urlencode(
        {"title": f"Quiz problem: {question.question[:80]}", "body": body}
    )
    return f"{REPORT_URL}?{query}"


def source_links(quiz: Quiz, question: QuizQuestion) -> str:
    links = []
    for passage in quiz.sources_for(question):
        links.append(f"[{passage.citation}]({passage.url})" if passage.url else passage.citation)
    return ", ".join(dict.fromkeys(links))  # de-duplicate, keep order


def quiz_tab(key: str, placeholder: str, generate: Callable[[str, int, Difficulty], Quiz]) -> None:
    """Form, quiz, and grading. `key` keeps each tab's quiz state separate."""
    with st.form(f"{key}_form"):
        topic = st.text_input("Quiz me on", placeholder=placeholder, max_chars=MAX_TOPIC_CHARS)
        level = st.radio(
            "Difficulty", [d.value.title() for d in Difficulty], index=1, horizontal=True
        )
        count = st.slider("Number of questions", 3, MAX_QUESTIONS, 5)
        submitted = st.form_submit_button("Generate quiz", type="primary")
    if submitted:
        with st.spinner("Reading the sources and writing your quiz..."):
            quiz = guarded(lambda: generate(topic, count, Difficulty(level.lower())))
        if quiz is not None:
            st.session_state[f"{key}_quiz"] = quiz
            st.session_state[f"{key}_id"] = st.session_state.get(f"{key}_id", 0) + 1
            st.session_state[f"{key}_graded"] = False

    quiz = st.session_state.get(f"{key}_quiz")
    if quiz is None:
        return
    quiz_id = st.session_state[f"{key}_id"]
    st.subheader(f"{quiz.topic} · {quiz.difficulty.value.title()}")
    source = f"[{quiz.source_title}]({quiz.source_url})" if quiz.source_url else quiz.source_title
    note = f"Written from {source}."
    if quiz.checked:
        note += " Every answer was checked against its source passage."
    if quiz.source_url:
        note += " Wikipedia text is available under CC BY-SA 4.0."
    st.caption(note)
    picks = []
    for i, q in enumerate(quiz.questions, start=1):
        with st.container(border=True):
            picks.append(
                st.radio(
                    f"**{i}. {q.question}**", q.choices, index=None, key=f"{key}_{quiz_id}_q{i}"
                )
            )
    if st.button("Check answers", key=f"{key}_check", type="primary"):
        st.session_state[f"{key}_graded"] = True
    if st.session_state.get(f"{key}_graded"):
        pairs = list(zip(picks, quiz.questions, strict=True))
        score = sum(p == q.choices[q.answer_index] for p, q in pairs)
        st.subheader(f"Score: {score} / {len(quiz.questions)}")
        for i, (pick, q) in enumerate(pairs, start=1):
            correct = q.choices[q.answer_index]
            mark = (
                '<span class="bb-mark bb-right">Correct</span>'
                if pick == correct
                else '<span class="bb-mark bb-wrong">Incorrect</span>'
            )
            st.markdown(
                f"{mark} **{i}.** Answer: *{correct}*. {q.explanation}", unsafe_allow_html=True
            )
            st.caption(
                f"Source: {source_links(quiz, q)} · [Report a problem]({report_link(quiz, q)})"
            )
    removed = []
    if quiz.dropped:
        removed.append(f"{quiz.dropped} malformed or repeated")
    if quiz.failed_check:
        removed.append(f"{quiz.failed_check} whose answer didn't match its source")
    if removed:
        st.caption("Left out " + " and ".join(removed) + " question(s).")


ask_tab, course_tab, anyquiz_tab, about_tab = st.tabs(["Ask", "Course quiz", "AnyQuiz", "About"])

# --- Ask -----------------------------------------------------------------------------------

with ask_tab:
    st.session_state.setdefault("history", [])
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["text"])
            render_sources(turn["sources"])

    if question := st.chat_input("Ask about your course materials...", max_chars=500):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            answer, tokens = engine.stream(question)
            try:
                st.write_stream(tokens)
            except BackOfTheBookError as e:
                show_error(e)
            else:
                render_sources(answer.cited_hits)
                if answer.invalid_citations:
                    st.caption(
                        "Ignored citations to sources that weren't provided: "
                        f"{answer.invalid_citations}"
                    )
                st.session_state.history.append(
                    {"question": question, "text": answer.text, "sources": answer.cited_hits}
                )

# --- Quizzes -------------------------------------------------------------------------------

with course_tab:
    st.markdown("Questions written **only from your course materials**, each with its source.")
    quiz_tab(
        "course",
        "e.g. hypothesis testing, k-means clustering",
        lambda topic, n, level: generate_quiz(retriever, llm, topic, n, level),
    )

with anyquiz_tab:
    st.markdown(
        "**Any topic you like.** AnyQuiz looks the topic up on Wikipedia and writes the quiz "
        "from that article, so every answer comes with a source you can check."
    )
    quiz_tab(
        "anyquiz",
        "e.g. Super Mario Galaxy, the French Revolution, photosynthesis",
        lambda topic, n, level: generate_anyquiz(llm, topic, n, level),
    )

# --- Evaluation ----------------------------------------------------------------------------

with about_tab:
    st.markdown(
        """
### How it works

**Ask.** Your question is matched against your course materials (or the sample textbook).
The closest passages are given to the AI, which answers using only those passages and cites
them. Open a source under any answer to read the original text and see its page number.
If nothing in your materials is close to the question, it says so instead of guessing.

**Course quiz.** Questions are written only from passages in your materials, and every
question lists the passage it came from.

**AnyQuiz.** For any topic, it finds the matching Wikipedia article and writes the quiz from
that article. Each question links to the section its answer comes from.

**Answer checking.** After a quiz is written, a second, separate pass looks at each
question and its source passage, without seeing the answer key, and picks the answer the
source supports. If the two don't match, the question is left out.

### How reliable is it?

We tested it on a 561-page data science textbook with questions whose correct page we
knew in advance:

- For **70 of 75** questions, the right page was among the five passages it read first.
- It answered all **94** real questions and declined **32 of 35** questions that had
  nothing to do with the book.

### Things to keep in mind

- It can still make mistakes, so check the source for anything important. Every answer
  and quiz question shows where it came from so you can.
- If a quiz question looks wrong, use **Report a problem** under it.
- Scanned PDFs (pictures of pages) can't be read.

The full test results and method are in the
[project README](https://github.com/RettWilson22/back-of-the-book#results).
"""
    )

st.markdown(FOOTER, unsafe_allow_html=True)
