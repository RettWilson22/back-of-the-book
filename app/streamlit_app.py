"""CoursePilot web app: ask questions with citations, take practice quizzes, see the eval.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from coursepilot.answer import AnswerEngine
from coursepilot.chunking import chunk_pages
from coursepilot.documents import SUPPORTED_SUFFIXES, load_document
from coursepilot.errors import CoursePilotError, ErrorCode
from coursepilot.index import (
    CorpusIndex,
    Embedder,
    IndexBuildError,
    SentenceTransformerEmbedder,
    default_index_dir,
)
from coursepilot.llm import make_provider
from coursepilot.quiz import (
    MAX_QUESTIONS,
    MAX_TOPIC_CHARS,
    Difficulty,
    Quiz,
    generate_anyquiz,
    generate_quiz,
)
from coursepilot.retrieval import CrossEncoderReranker, Hit, Retriever
from coursepilot.sample import build_sample_index

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = Path(os.environ.get("COURSEPILOT_INDEX") or default_index_dir(ROOT))
# The repo ships a prebuilt sample index, so this only runs if it was deleted (0 to skip).
USE_SAMPLE = os.environ.get("COURSEPILOT_SAMPLE", "1") != "0"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="CoursePilot", page_icon="🎓", layout="wide")
logger = logging.getLogger("coursepilot.app")

# Errors caused by what the user typed are shown as information, not as failures.
_INPUT_ERRORS = {
    ErrorCode.EMPTY_TOPIC,
    ErrorCode.TOPIC_TOO_LONG,
    ErrorCode.INVALID_QUESTION_COUNT,
    ErrorCode.INVALID_DIFFICULTY,
    ErrorCode.TOPIC_NOT_COVERED,
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

st.sidebar.title("🎓 CoursePilot")
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
        "index with `python scripts/download_corpus.py && coursepilot ingest data/corpus`."
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


def show_error(error: CoursePilotError) -> None:
    hint = " Trying again may work." if error.retryable else ""
    (st.info if error.code in _INPUT_ERRORS else st.error)(error.message + hint)
    st.caption(f"Error code: `{error.code}`")


def guarded(action: Callable[[], Quiz]) -> Quiz | None:
    """Run a quiz request, turning every failure into a coded, user-facing message."""
    try:
        return action()
    except CoursePilotError as e:
        logger.info("quiz request failed: %r", e)
        show_error(e)
    except Exception:
        logger.exception("unexpected error while generating a quiz")
        show_error(
            CoursePilotError(
                ErrorCode.INTERNAL_ERROR, "Something went wrong on our side. Please try again."
            )
        )
    return None


def quiz_tab(key: str, placeholder: str, generate: Callable[[str, int, Difficulty], Quiz]) -> None:
    """Form, quiz, and grading. `key` keeps each tab's quiz state separate."""
    with st.form(f"{key}_form"):
        topic = st.text_input("Quiz me on", placeholder=placeholder, max_chars=MAX_TOPIC_CHARS)
        level = st.radio(
            "Difficulty", [d.value.title() for d in Difficulty], index=1, horizontal=True
        )
        count = st.slider("Number of questions", 3, MAX_QUESTIONS, 5)
        submitted = st.form_submit_button("Generate quiz")
    if submitted:
        with st.spinner("Writing your quiz..."):
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
    if not quiz.grounded:
        note = "Written by AI from general knowledge, not from your course materials."
        if quiz.checked:
            note += " Answers were double-checked by a second, independent pass."
        st.caption(note)
    picks = [
        st.radio(f"**{i}. {q.question}**", q.choices, index=None, key=f"{key}_{quiz_id}_q{i}")
        for i, q in enumerate(quiz.questions, start=1)
    ]
    if st.button("Check answers", key=f"{key}_check"):
        st.session_state[f"{key}_graded"] = True
    if st.session_state.get(f"{key}_graded"):
        pairs = list(zip(picks, quiz.questions, strict=True))
        score = sum(p == q.choices[q.answer_index] for p, q in pairs)
        st.subheader(f"Score: {score} / {len(quiz.questions)}")
        for i, (pick, q) in enumerate(pairs, start=1):
            correct = q.choices[q.answer_index]
            mark = "✅" if pick == correct else "❌"
            st.markdown(f"{mark} **{i}.** Correct answer: *{correct}*. {q.explanation}")
            if quiz.grounded:
                st.caption("Source: " + ", ".join(h.chunk.citation for h in quiz.sources_for(q)))
    removed = []
    if quiz.dropped:
        removed.append(f"{quiz.dropped} malformed or repeated")
    if quiz.failed_check:
        removed.append(f"{quiz.failed_check} whose answer the double-check disagreed with")
    if removed:
        st.caption("Left out " + " and ".join(removed) + " question(s).")


ask_tab, course_tab, anyquiz_tab, eval_tab = st.tabs(
    ["💬 Ask", "📝 Course quiz", "🎲 AnyQuiz", "📊 How accurate is it?"]
)

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
            except CoursePilotError as e:
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
    st.markdown("**Any topic you like.** The AI writes the quiz from what it knows.")
    double_check = st.toggle(
        "Double-check answers",
        value=True,
        help="A second pass answers each question without seeing the answer key. Questions "
        "where the two disagree are left out. Takes a little longer.",
    )
    quiz_tab(
        "anyquiz",
        "e.g. Super Mario Galaxy, the French Revolution, photosynthesis",
        lambda topic, n, level: generate_anyquiz(llm, topic, n, level, verify=double_check),
    )

# --- Evaluation ----------------------------------------------------------------------------

with eval_tab:
    st.markdown(
        "Retrieval accuracy on a fixed question set: each question is labeled with the textbook "
        "page that answers it. See the README for how the questions were written."
    )
    for title, name in [
        ("Main question set", "results.md"),
        ("Held-out set", "results-heldout.md"),
    ]:
        path = ROOT / "eval" / name
        if path.exists():
            st.subheader(title)
            st.markdown(path.read_text())
