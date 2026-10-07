"""CoursePilot web app: ask questions with citations, take practice quizzes, see the eval.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from coursepilot.answer import AnswerEngine
from coursepilot.chunking import chunk_pages
from coursepilot.documents import SUPPORTED_SUFFIXES, load_document
from coursepilot.index import CorpusIndex, Embedder, IndexBuildError, SentenceTransformerEmbedder
from coursepilot.llm import LLMError, make_provider
from coursepilot.quiz import generate_quiz
from coursepilot.retrieval import CrossEncoderReranker, Hit, Retriever

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = Path(os.environ.get("COURSEPILOT_INDEX", ROOT / ".coursepilot" / "index"))

st.set_page_config(page_title="CoursePilot", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def load_models(model_name: str) -> tuple[SentenceTransformerEmbedder, CrossEncoderReranker]:
    return SentenceTransformerEmbedder(model_name), CrossEncoderReranker()


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
base_index = load_base_index()
embedder, reranker = load_models(
    base_index.embedding_model if base_index else "sentence-transformers/all-MiniLM-L6-v2"
)

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

ask_tab, quiz_tab, eval_tab = st.tabs(["💬 Ask", "📝 Practice quiz", "📊 How accurate is it?"])

# --- Ask -----------------------------------------------------------------------------------

with ask_tab:
    st.session_state.setdefault("history", [])
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["text"])
            render_sources(turn["sources"])

    if question := st.chat_input("Ask about your course materials..."):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            answer, tokens = engine.stream(question)
            try:
                st.write_stream(tokens)
            except LLMError as e:
                st.error(str(e))
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

# --- Quiz ----------------------------------------------------------------------------------

with quiz_tab:
    with st.form("quiz_form"):
        topic = st.text_input(
            "Quiz me on", placeholder="e.g. hypothesis testing, k-means clustering"
        )
        count = st.slider("Number of questions", 3, 10, 5)
        make_quiz = st.form_submit_button("Generate quiz")
    if make_quiz and topic.strip():
        with st.spinner("Writing questions from your materials..."):
            try:
                st.session_state.quiz = generate_quiz(retriever, llm, topic, n=count)
                st.session_state.quiz_checked = False
            except LLMError as e:
                st.error(str(e))

    quiz = st.session_state.get("quiz")
    if quiz:
        if not quiz.questions:
            st.warning("No valid questions came back. Try a broader topic.")
        picks = []
        for i, q in enumerate(quiz.questions):
            st.markdown(f"**{i + 1}. {q.question}**")
            picks.append(
                st.radio("Answer", q.choices, index=None, key=f"q{i}", label_visibility="collapsed")
            )
        if quiz.questions and st.button("Check answers"):
            st.session_state.quiz_checked = True
        if st.session_state.get("quiz_checked"):
            score = sum(
                p == q.choices[q.answer_index] for p, q in zip(picks, quiz.questions, strict=True)
            )
            st.subheader(f"Score: {score} / {len(quiz.questions)}")
            for i, (pick, q) in enumerate(zip(picks, quiz.questions, strict=True)):
                correct = q.choices[q.answer_index]
                mark = "✅" if pick == correct else "❌"
                st.markdown(f"{mark} **{i + 1}.** Correct answer: *{correct}*. {q.explanation}")
                st.caption("Source: " + ", ".join(h.chunk.citation for h in quiz.sources_for(q)))
        if quiz.dropped:
            st.caption(f"{quiz.dropped} generated question(s) failed validation and were left out.")

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
