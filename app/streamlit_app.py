"""Back of the Book web app: ask questions with citations and take practice quizzes.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import functools
import html
import logging
import os
import sys
import tempfile
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Always import the package from this repo's src/, not a copy installed earlier. Hosts like
# Streamlit Community Cloud install requirements once and only reinstall when requirements.txt
# changes, so an installed copy can fall behind the app code after a push.
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit.errors import StreamlitSecretNotFoundError
from streamlit.runtime.uploaded_file_manager import UploadedFile


def reload_package_if_updated() -> None:
    """After a deploy, drop stale copies of the package so the new code is imported.

    Streamlit keeps imported modules in memory between runs and only reloads files it is
    watching, so a push that changes src/ could keep running old code. The package records
    its source files' newest timestamp when imported (SOURCE_STAMP); if the files on disk
    are newer, or the loaded copy predates that attribute, its modules are removed from
    sys.modules (re-imported fresh below) and cached models and indexes are cleared.
    """
    package = sys.modules.get("backofthebook")
    if package is None:
        return  # not imported yet in this process, so it will load from the current files
    newest = max(p.stat().st_mtime for p in (SRC / "backofthebook").glob("*.py"))
    if getattr(package, "SOURCE_STAMP", None) == newest:
        return
    for name in [m for m in sys.modules if m == "backofthebook" or m.startswith("backofthebook.")]:
        del sys.modules[name]
    st.cache_resource.clear()


reload_package_if_updated()

from backofthebook.answer import Answer, AnswerEngine, Turn, escape_markdown, tidy_markdown
from backofthebook.chunking import chunk_pages
from backofthebook.documents import SUPPORTED_SUFFIXES, UnsupportedFileError, read_document
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
from backofthebook.sample import FILENAME as SAMPLE_FILE
from backofthebook.sample import build_sample_index

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = Path(os.environ.get("BACKOFTHEBOOK_INDEX") or default_index_dir(ROOT))
# The repo ships a prebuilt sample index, so this only runs if it was deleted (0 to skip).
USE_SAMPLE = os.environ.get("BACKOFTHEBOOK_SAMPLE", "1") != "0"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Limits on what one visitor can upload. The server also refuses files over maxUploadSize
# (.streamlit/config.toml); the size is checked again here.
MAX_UPLOAD_MB = 10
MAX_DOCUMENTS = 5
MAX_PDF_PAGES = 300
MAX_CHARS_PER_FILE = 2_000_000
MAX_SESSION_CHUNKS = 3_000

st.set_page_config(
    page_title="Back of the Book",
    page_icon=str(ROOT / "app" / "static" / "favicon.png"),
    layout="wide",
)
logger = logging.getLogger("backofthebook.app")


def load_api_keys_from_secrets() -> None:
    """On Streamlit Community Cloud, API keys are set in the app's Secrets settings. Copy them
    into the environment, where the Groq and Anthropic clients look for them."""
    for name in ("GROQ_API_KEY", "ANTHROPIC_API_KEY"):
        if os.environ.get(name):
            continue
        try:
            value = st.secrets.get(name)
        except (FileNotFoundError, StreamlitSecretNotFoundError):
            return  # running locally with no secrets file
        if value:
            os.environ[name] = str(value)


load_api_keys_from_secrets()

STYLE = """
<style>
/* Hide Streamlit's own chrome so the page reads like a normal website. */
header[data-testid="stHeader"], footer, [data-testid="stToolbar"] { display: none; }
.block-container { padding-top: 0; max-width: 1100px; }

.bb-masthead {
  background: #22313f; color: #fdfaf2; margin: 0 -100vw 1.6rem; padding: 1.3rem 100vw 1.2rem;
  border-bottom: 5px solid #c9a24a;
}
.bb-logo { display: flex; align-items: center; gap: 16px; }
.bb-icon { width: 54px; height: 54px; flex: none; }
.bb-brand { font-family: Georgia, "Times New Roman", serif; font-size: 2.7rem; font-weight: 700;
  line-height: 1; letter-spacing: 0.2px; }
.bb-brand em { font-weight: 400; font-size: 0.78em; color: #c9a24a; }
.bb-tagline { font-size: 0.78rem; color: #b9c4cf; margin-top: 0.45rem; text-transform: uppercase;
  letter-spacing: 1.6px; }

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
.bb-skip { color: #5c5a52; background: #ecebe4; border-color: #c4c1b5; }
/* A graded answer: its mark and its explanation are separate elements on one line. */
[class*="st-key-bb_result_"] { flex-direction: row; align-items: baseline; gap: 0; }
[class*="st-key-bb_result_"] > div:first-child { flex: none; width: auto; }
[class*="st-key-bb_result_"] > div:last-child { flex: 1 1 0; min-width: 0; }

/* Chat that reads like a modern assistant: your messages in bubbles on the right,
   replies as plain text on the left. */
[data-testid="stChatMessage"] { background: transparent; padding: 0.35rem 0; gap: 0.6rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  flex-direction: row-reverse; justify-content: flex-start; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] {
  background: #e4ddcd; border-radius: 18px; padding: 0.55rem 1rem; max-width: 75%;
  flex: 0 1 auto; margin: 0 !important; }
[data-testid="stChatMessageAvatarUser"] { display: none; }
[data-testid="stChatMessageAvatarAssistant"] { background: #22313f; color: #c9a24a; }
.bb-thought { color: #6b6458; font-size: 0.88rem; white-space: pre-wrap; }

/* Keep each uploaded document's remove button on the same line, even on phones. */
.st-key-doc_list [data-testid="stHorizontalBlock"] { flex-wrap: nowrap; gap: 0.5rem; }
.st-key-doc_list [data-testid="stColumn"] { min-width: 0; }
.st-key-doc_list [data-testid="stColumn"]:last-child { flex: 0 0 auto; width: auto; }

.bb-footer { margin-top: 3rem; padding: 1rem 0; border-top: 1px solid #cfc6b4;
  font-size: 0.85rem; color: #6b6458; }
/* The full-width header bar must never let the page scroll sideways. */
html, body, .stApp, [data-testid="stMain"] { overflow-x: hidden; }

@media (max-width: 640px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .bb-masthead { padding-top: 0.9rem; padding-bottom: 0.85rem; margin-bottom: 1.1rem; }
  .bb-logo { gap: 10px; }
  .bb-icon { width: 34px; height: 34px; }
  .bb-brand { font-size: 1.65rem; white-space: nowrap; }
  .bb-tagline { font-size: 0.66rem; letter-spacing: 1px; margin-top: 0.3rem; }
  [data-testid="stTabs"] [role="tablist"] { gap: 2px; }
  [data-testid="stTab"] { padding: 0.4rem 0.6rem; }
  [data-testid="stTab"] p { font-size: 0.85rem; }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] { max-width: 88%; }
  .bb-footer { font-size: 0.78rem; }
}
</style>
"""

MASTHEAD = """
<div class="bb-masthead">
  <div class="bb-logo">
    <svg class="bb-icon" viewBox="0 0 48 48" fill="none" stroke="#c9a24a" stroke-width="2.4"
         stroke-linejoin="round" stroke-linecap="round" aria-hidden="true">
      <path d="M24 13c-4.5-3.2-11-4.2-19-3.2v27c8-1 14.5 0 19 3.2 4.5-3.2 11-4.2 19-3.2v-27
               c-8-1-14.5 0-19 3.2z"/>
      <path d="M24 13v27"/>
      <path d="M10 17c3.5-.4 7 0 9.5 1.2M10 23c3.5-.4 7 0 9.5 1.2M28.5 18.2c2.5-1.2 6-1.6 9.5-1.2"/>
    </svg>
    <div>
      <div class="bb-brand">Back of <em>the</em> Book</div>
      <div class="bb-tagline">Create a quiz on any topic</div>
    </div>
  </div>
</div>
"""

FOOTER = """
<div class="bb-footer">
  Back of the Book, built by Rett Wilson &middot;
  <a href="https://github.com/RettWilson22/back-of-the-book">Source code on GitHub</a> &middot;
  Sample textbook: OpenStax <i>Principles of Data Science</i> (CC BY-NC-SA 4.0) &middot;
  Quiz generator text from Wikipedia (CC BY-SA 4.0)
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


@dataclass
class Document:
    """One uploaded file, processed once and kept only in this visitor's session."""

    name: str
    index: CorpusIndex
    pages: int
    notes: list[str]


def upload_name(raw: str) -> str:
    """The name an uploaded file is listed and cited under. A path in it is never trusted, and
    it can't form a Markdown image, since labels that show it render Markdown."""
    return Path(raw).name.replace("![", "! [")


def process_upload(upload: UploadedFile, embedder: Embedder, room: int) -> Document | str:
    """Read and index one uploaded file. Returns the document, or an error message.

    `room` is how many more passages this session can hold.
    """
    name = upload_name(upload.name)
    if upload.size > MAX_UPLOAD_MB * 1024 * 1024:
        return f"{name} is larger than {MAX_UPLOAD_MB} MB, so it can't be added."
    with tempfile.TemporaryDirectory() as tmp:  # deleted as soon as the file has been read
        path = Path(tmp) / name
        path.write_bytes(upload.getvalue())
        try:
            text = read_document(path, max_pages=MAX_PDF_PAGES, max_chars=MAX_CHARS_PER_FILE)
        except UnsupportedFileError as e:
            return str(e)
        except Exception:  # a bad upload shouldn't take down the app
            logger.exception("couldn't read upload %s", name)
            return f"Couldn't read {name}. The file may be damaged or password-protected."
    chunks = chunk_pages(text.pages)
    if not chunks:
        return f"Couldn't find any text in {name}. If it's a scan, it can't be read yet."
    if len(chunks) > room:
        return (
            f"{name} is too long to add alongside your other documents. Remove one, or "
            "upload a shorter file."
        )
    notes = []
    if text.unreadable:
        listed = ", ".join(str(n) for n in text.unreadable[:10])
        notes.append(f"Page(s) {listed} have no readable text (probably scanned images).")
    return Document(name, CorpusIndex.build(chunks, embedder), len(text.pages), notes)


def add_uploads(uploads: list[UploadedFile], docs: dict[str, Document]) -> list[str]:
    """Process new uploads into `docs` within the session limits. Returns error messages."""
    errors = []
    for upload in uploads:
        name = upload_name(upload.name)
        if len(docs) >= MAX_DOCUMENTS and name not in docs:
            errors.append(
                f"You can add up to {MAX_DOCUMENTS} documents, so {name} wasn't added. "
                "Remove one to make room."
            )
            continue
        used = sum(len(d.index.chunks) for d in docs.values() if d.name != name)
        result = process_upload(upload, embedder, MAX_SESSION_CHUNKS - used)
        if isinstance(result, str):
            errors.append(result)
        else:
            docs[result.name] = result
    return errors


def label(source: str) -> str:
    """How a document is named on screen."""
    return "Sample textbook" if source == SAMPLE_FILE else source


def set_sample_hidden(hidden: bool) -> None:
    st.session_state.sample_hidden = hidden
    picked = st.session_state.get("referenced_docs") or []
    st.session_state.referenced_docs = [n for n in picked if n != SAMPLE_FILE]


def document_row(
    key: str, title: str, detail: str, button: str, action: Callable[[], None], help: str
) -> None:
    """One document in the list: its name and details, with its button on the same line."""
    with st.container(border=True):
        name_col, button_col = st.columns([6, 1], vertical_alignment="center")
        name_col.markdown(f"**{title}**")
        name_col.caption(detail)
        button_col.button(button, key=key, help=help, on_click=action)


def remove_document(name: str) -> None:
    st.session_state.docs.pop(name, None)
    picked = st.session_state.get("referenced_docs") or []
    st.session_state.referenced_docs = [n for n in picked if n != name]


def render_sources(numbered_hits: list[tuple[int, Hit]]) -> None:
    for n, hit in numbered_hits:
        with st.expander(f"[S{n}] {hit.chunk.citation}"):
            st.text(hit.chunk.text)  # document text is shown as written, never as Markdown


def thought_html(thinking: str) -> str:
    """The model's reasoning as one escaped HTML block. It has no line breaks, because a blank
    line would end the HTML block and the rest would be rendered as Markdown."""
    lines = html.escape(thinking).splitlines()
    return f'<div class="bb-thought">{"<br>".join(lines)}</div>'


def thought_label(seconds: float) -> str:
    whole = round(seconds)
    if whole < 1:
        return "Thought for a moment"
    return f"Thought for {whole} second{'s' if whole != 1 else ''}"


def render_answer_footer(turn: dict[str, object]) -> None:
    hits = turn["sources"]
    assert isinstance(hits, list)
    if not turn["grounded"]:
        st.caption("Answered from general knowledge, not from your materials.")
    elif hits:
        st.caption("Sources")
        render_sources(hits)
    invalid = turn.get("invalid")
    if invalid:
        st.caption(f"Ignored citations to sources that weren't provided: {invalid}")


def turn_record(answer: Answer) -> dict[str, object]:
    return {
        "question": answer.question,
        "text": answer.text,
        "thinking": answer.thinking,
        "seconds": answer.thinking_seconds,
        "sources": answer.cited_hits,
        "grounded": answer.grounded,
        "invalid": answer.invalid_citations,
    }


# --- Layout: tabs first, so the upload box can sit inside "Ask" ----------------------------

ask_tab, course_tab, anyquiz_tab, about_tab = st.tabs(
    ["Ask", "Course quiz", "Quiz generator", "About"]
)
about_body, about_settings = about_tab.container(), about_tab.container()

if USE_SAMPLE and not (INDEX_DIR / "meta.json").exists():
    with st.spinner(
        "First start: downloading the sample textbook and indexing it. This takes about a minute."
    ):
        if error := build_sample(INDEX_DIR):
            st.warning(f"Couldn't set up the sample textbook ({error}). You can upload files.")
base_index = load_base_index()
embedder, reranker = load_models(base_index.embedding_model if base_index else DEFAULT_MODEL)

st.session_state.setdefault("docs", {})
st.session_state.setdefault("uploader_round", 0)
docs: dict[str, Document] = st.session_state.docs

with ask_tab:
    uploads = st.file_uploader(
        "Add your own documents (optional)",
        type=[s.lstrip(".") for s in SUPPORTED_SUFFIXES],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_round}",
        help="PDF slides or notes, PowerPoint decks, Markdown, or text: up to "
        f"{MAX_DOCUMENTS} files and {MAX_PDF_PAGES} pages each. Your files are only "
        "visible to you and are gone when you close or refresh the page.",
    )
    if uploads:
        with st.spinner("Reading your files..."):
            st.session_state.upload_errors = add_uploads(uploads, docs)
        st.session_state.uploader_round += 1  # clears the upload box for the next file
        st.rerun()
    for error in st.session_state.pop("upload_errors", []):
        st.error(error)

    # The sample textbook can be hidden only while there are uploads to answer from instead.
    hide_sample = bool(st.session_state.get("sample_hidden")) and bool(docs)
    show_sample = base_index is not None and not hide_sample
    if docs or show_sample:
        st.markdown("**Documents**")
        with st.container(key="doc_list"):
            for doc in docs.values():
                detail = f"{doc.pages} page{'s' if doc.pages != 1 else ''} · only visible to you"
                document_row(
                    f"remove_{doc.name}",
                    doc.name,
                    " · ".join([detail, *doc.notes]),
                    "Remove",
                    functools.partial(remove_document, doc.name),
                    f"Remove {doc.name}",
                )
            if show_sample and base_index is not None:
                built_in = base_index.sources
                title = (
                    "Sample textbook: OpenStax Principles of Data Science"
                    if built_in == [SAMPLE_FILE]
                    else ", ".join(built_in)
                )
                detail = "Built in for every visitor, so you can try the app right away."
                if docs:
                    document_row(
                        "hide_sample",
                        title,
                        detail,
                        "Hide",
                        lambda: set_sample_hidden(True),
                        "Stop using the sample textbook (just for you)",
                    )
                else:
                    with st.container(border=True):
                        st.markdown(f"**{title}**")
                        st.caption(detail)
    if hide_sample:
        st.button(
            "Use the sample textbook again",
            key="show_sample",
            on_click=lambda: set_sample_hidden(False),
        )

parts = ([base_index] if base_index and not hide_sample else []) + [d.index for d in docs.values()]
index: CorpusIndex | None = CorpusIndex.merge(parts) if parts else None

if index is None:
    with ask_tab:
        st.info("Upload your notes, slides, or textbook above to get started.")
    st.stop()

with ask_tab:
    referenced = st.pills(
        "Reference a document",
        index.sources,
        format_func=label,
        selection_mode="multi",
        key="referenced_docs",
        help="Click a document to answer only from it. Click again to go back to the default.",
    )
    # Default scope: the student's own uploads if there are any, otherwise everything loaded.
    uploaded_names = sorted(docs)
    scope_sources: list[str] | None = list(referenced) or uploaded_names or None
    scope = ", ".join(map(label, scope_sources)) if scope_sources else "all loaded documents"
    st.caption(f"Answering from {scope}.")

provider_names = {
    "Automatic": None,
    "Groq": "groq",
    "Claude": "claude",
    "No AI (show matching passages only)": "extractive",
}
with about_settings:
    st.markdown("### Settings")
    choice = st.selectbox("Answers and quizzes are written by", list(provider_names), index=0)
    try:
        llm = make_provider(provider_names[choice])
    except Exception:
        logger.exception("couldn't start provider %s", choice)
        st.error(f"Couldn't start {choice}. Using matching passages only for now.")
        llm = make_provider("extractive")
    friendly = {"groq": "Groq", "claude": "Claude", "extractive": "No AI (matching passages only)"}
    st.caption(f"Currently using: {friendly.get(llm.name, llm.name)}")

retriever = Retriever(index, embedder, reranker)
engine = AnswerEngine(retriever, llm)


def show_error(error: BackOfTheBookError) -> None:
    hint = " Trying again may work." if error.retryable else ""
    if error.code in _INPUT_ERRORS:  # the user's own typo: no need for a code
        st.info(error.message)
        return
    st.error(error.message + hint)
    st.caption(f"Error code: `{error.code}`")


def guarded(action: Callable[[], Quiz], slot: DeltaGenerator) -> Quiz | None:
    """Run a quiz request, turning every failure into a coded, user-facing message in `slot`."""
    try:
        return action()
    except BackOfTheBookError as e:
        logger.info("quiz request failed: %r", e)
        error = e
    except Exception:
        logger.exception("unexpected error while generating a quiz")
        error = BackOfTheBookError(
            ErrorCode.INTERNAL_ERROR, "Something went wrong on our side. Please try again."
        )
    with slot.container():
        show_error(error)
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


# Constant HTML for the grade marks. Model-written text is never put into HTML.
MARKS = {
    "right": '<span class="bb-mark bb-right">Correct</span>',
    "skip": '<span class="bb-mark bb-skip">Skipped</span>',
    "wrong": '<span class="bb-mark bb-wrong">Incorrect</span>',
}


def quiz_tab(key: str, placeholder: str, generate: Callable[[str, int, Difficulty], Quiz]) -> None:
    """Form, quiz, and grading. `key` keeps each tab's quiz state separate."""
    with st.form(f"{key}_form"):
        topic = st.text_input("Quiz me on", placeholder=placeholder, max_chars=MAX_TOPIC_CHARS)
        level = st.radio(
            "Difficulty", [d.value.title() for d in Difficulty], index=1, horizontal=True
        )
        count = st.slider("Number of questions", 3, MAX_QUESTIONS, 5)
        submitted = st.form_submit_button("Generate quiz", type="primary")
    message = st.empty()  # replaces an old error as soon as the next request starts
    if submitted:
        with st.spinner("Reading the sources and writing your quiz..."):
            quiz = guarded(lambda: generate(topic, count, Difficulty(level.lower())), message)
        if quiz is not None:
            st.session_state[f"{key}_quiz"] = quiz
            st.session_state[f"{key}_id"] = st.session_state.get(f"{key}_id", 0) + 1
            st.session_state[f"{key}_graded"] = False

    quiz = st.session_state.get(f"{key}_quiz")
    if quiz is None:
        return
    quiz_id = st.session_state[f"{key}_id"]
    title = quiz.topic[:1].upper() + quiz.topic[1:]
    st.subheader(f"{title} · {quiz.difficulty.value.title()}")
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
                    f"**{i}. {escape_markdown(q.question)}**",
                    q.choices,
                    index=None,
                    format_func=escape_markdown,
                    key=f"{key}_{quiz_id}_q{i}",
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
            chose = ""
            if pick == correct:
                mark = MARKS["right"]
            elif pick is None:
                mark = MARKS["skip"]
            else:
                mark = MARKS["wrong"]
                chose = f" You chose *{escape_markdown(pick)}*."
            with st.container(key=f"bb_result_{key}_{i}"):
                st.markdown(mark, unsafe_allow_html=True)
                st.markdown(
                    f"**{i}.**{chose} Answer: *{escape_markdown(correct)}*. "
                    + escape_markdown(q.explanation)
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


# --- Ask -----------------------------------------------------------------------------------


def stream_reply(question: str, sources: list[str] | None) -> dict[str, object] | None:
    """Stream one reply with a live "Thinking" status, then the answer and its sources."""
    history = [Turn(str(t["question"]), str(t["text"])) for t in st.session_state.turns]
    answer, events = engine.stream(question, history, sources)
    status = None if llm.name == "extractive" else st.status("Thinking...", expanded=False)
    thought_box = status.empty() if status else None
    reply_box = st.empty()
    thinking, text = "", ""
    try:
        for event in events:
            if event.kind == "thinking" and thought_box is not None:
                thinking += event.text
                thought_box.markdown(thought_html(thinking), unsafe_allow_html=True)
            elif event.kind == "text":
                if status is not None and not text:
                    status.update(label=thought_label(answer.thinking_seconds), state="complete")
                text += event.text
                reply_box.markdown(tidy_markdown(text) + " ▌")
    except BackOfTheBookError as e:
        if status is not None:
            status.update(label="Something went wrong", state="error")
        show_error(e)
        return None
    if status is not None and not text:
        status.update(label=thought_label(answer.thinking_seconds), state="complete")
    reply_box.markdown(answer.text)
    record = turn_record(answer)
    render_answer_footer(record)
    return record


def ask_example(question: str) -> None:
    st.session_state.pending_question = question


with ask_tab:
    st.session_state.setdefault("turns", [])
    if st.session_state.turns and st.button("New chat", key="new_chat"):
        st.session_state.turns = []
        st.rerun()

    for turn in st.session_state.turns:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            if turn["thinking"]:
                with st.expander(thought_label(turn["seconds"])):
                    st.markdown(thought_html(str(turn["thinking"])), unsafe_allow_html=True)
            st.markdown(turn["text"])
            render_answer_footer(turn)

    if not st.session_state.turns and "pending_question" not in st.session_state:
        if docs:
            st.markdown("Ask anything about your documents, or anything else.")
        else:
            st.markdown(
                "A sample data science textbook is loaded so you can try it right away. Ask "
                "about it, ask anything else, or add your own notes and slides above."
            )
            examples = [
                "What is the difference between mean and median?",
                "Explain a p-value in simple terms",
                "Solve: what is the standard deviation of 2, 4, 4, 4, 5, 5, 7, 9?",
            ]
            with st.container(key="examples"):
                for n, example in enumerate(examples):
                    st.button(example, key=f"example_{n}", on_click=ask_example, args=(example,))

    new_turn = st.container()  # keeps the newest exchange above the input box
    typed = st.chat_input("Ask anything", max_chars=2000)
    question = typed or st.session_state.pop("pending_question", None)
    if question:
        with new_turn:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                record = stream_reply(question, scope_sources)
        if record is not None:
            st.session_state.turns.append(record)

# --- Quizzes -------------------------------------------------------------------------------

with course_tab:
    st.markdown(
        "Questions written **only from your course materials**, each with its source. Uses the "
        "sample data science textbook plus any documents you add in **Ask**."
    )
    quiz_tab(
        "course",
        "e.g. hypothesis testing, k-means clustering",
        lambda topic, n, level: generate_quiz(retriever, llm, topic, n, level),
    )

with anyquiz_tab:
    st.markdown(
        "**Any topic you like.** The Quiz generator looks up a reliable source on the topic and "
        "writes the quiz from it, so every answer comes with a source you can check."
    )
    quiz_tab(
        "anyquiz",
        "e.g. Pokémon, the NFL, the French Revolution",
        lambda topic, n, level: generate_anyquiz(llm, topic, n, level),
    )

# --- About ----------------------------------------------------------------------------

with about_body:
    st.markdown(
        """
### How it works

**Ask.** Chat with the AI like any assistant: it remembers the conversation, works through
problems step by step, and shows what it was thinking. When your materials cover the
question, it uses the closest passages and cites them, so you can open a source and see the
original text and page number. When you upload files, it answers from your files. Short
documents (like a syllabus or an assignment) are read in full, and for longer ones it always
reads the first page plus the passages that best match your question. Click a document above
the chat to answer only from that document. When your materials don't cover a question, it
answers from general knowledge and says so.

**Course quiz.** Questions are written only from passages in your materials, and every
question lists the passage it came from.

**Quiz generator.** For any topic, it finds the matching Wikipedia article and writes the quiz from
that article. Each question links to the section its answer comes from.

**Answer checking.** After a quiz is written, a second, separate pass looks at each
question and its source passage, without seeing the answer key, and picks the answer the
source supports. If the two don't match, the question is left out.

### How reliable is it?

We tested it on a 561-page data science textbook with questions whose correct page we
knew in advance:

- For **70 of 75** questions, the right page was among the five passages it read first.
- It recognized that **32 of 35** questions that had nothing to do with the book weren't
  in it, and labeled those answers as general knowledge instead of citing the book.

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
