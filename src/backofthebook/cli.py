"""Command-line interface: `backofthebook ingest | ask | quiz | eval`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from backofthebook.errors import BackOfTheBookError
from backofthebook.index import USER_INDEX, default_index_dir

if TYPE_CHECKING:
    from backofthebook.quiz import Quiz
    from backofthebook.retrieval import Retriever


def _retriever(index_dir: Path, rerank: bool = True) -> Retriever:
    from backofthebook.index import CorpusIndex, SentenceTransformerEmbedder
    from backofthebook.retrieval import CrossEncoderReranker, Retriever

    index = CorpusIndex.load(index_dir)
    return Retriever(
        index,
        SentenceTransformerEmbedder(index.embedding_model),
        CrossEncoderReranker() if rerank else None,
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    from backofthebook.chunking import chunk_pages
    from backofthebook.documents import find_documents, load_document
    from backofthebook.index import CorpusIndex, SentenceTransformerEmbedder

    files = find_documents([Path(p) for p in args.paths])
    if not files:
        print("No supported documents found (.pdf, .pptx, .md, .txt).", file=sys.stderr)
        return 1
    pages = []
    for path in files:
        loaded = load_document(path)
        print(f"  {path.name}: {len(loaded)} pages")
        pages.extend(loaded)
    chunks = chunk_pages(pages, max_words=args.chunk_words, overlap_words=args.overlap_words)
    print(f"Embedding {len(chunks)} chunks...")
    index = CorpusIndex.build(chunks, SentenceTransformerEmbedder(args.embedding_model))
    destination = args.index or USER_INDEX
    index.save(destination)
    print(f"Saved index to {destination}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from backofthebook.answer import AnswerEngine
    from backofthebook.llm import make_provider

    engine = AnswerEngine(
        _retriever(args.index or default_index_dir()), make_provider(args.llm), k=args.k
    )
    answer, tokens = engine.stream(args.question)
    for token in tokens:
        print(token, end="", flush=True)
    print("\n")
    for n, hit in answer.cited_hits:
        print(f"  [S{n}] {hit.chunk.citation}")
    if answer.invalid_citations:
        print(f"  (ignored citations to nonexistent sources: {answer.invalid_citations})")
    return 0


def print_quiz(quiz: Quiz) -> None:
    print(f"{quiz.topic} ({quiz.difficulty.value})")
    checked = "; every answer checked against its source" if quiz.checked else ""
    print(f"Written from {quiz.source_title}{checked}.")
    for i, q in enumerate(quiz.questions, start=1):
        print(f"\n{i}. {q.question}")
        for letter, choice in zip("ABCD", q.choices, strict=True):
            print(f"   {letter}) {choice}")
        sources = ", ".join(dict.fromkeys(p.citation for p in quiz.sources_for(q)))
        print(f"   Answer: {'ABCD'[q.answer_index]}. {q.explanation} ({sources})")
    if quiz.dropped or quiz.failed_check:
        print(
            f"\n(left out {quiz.dropped} malformed or repeated question(s) and "
            f"{quiz.failed_check} that failed the source check)"
        )


def cmd_quiz(args: argparse.Namespace) -> int:
    from backofthebook.llm import make_provider
    from backofthebook.quiz import generate_quiz

    retriever = _retriever(args.index or default_index_dir())
    print_quiz(
        generate_quiz(retriever, make_provider(args.llm), args.topic, args.n, args.difficulty)
    )
    return 0


def cmd_anyquiz(args: argparse.Namespace) -> int:
    from backofthebook.llm import make_provider
    from backofthebook.quiz import generate_anyquiz

    quiz = generate_anyquiz(
        make_provider(args.llm), args.topic, args.n, args.difficulty, check=not args.no_check
    )
    print_quiz(quiz)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from backofthebook.evaluation import (
        evaluate_mode,
        evaluate_scope,
        format_report,
        load_questions,
    )
    from backofthebook.retrieval import Mode

    questions = load_questions(args.questions)
    retriever = _retriever(args.index or default_index_dir())
    results = []
    for mode in Mode:
        print(f"Evaluating {mode.value}...", file=sys.stderr)
        results.append(evaluate_mode(retriever, questions, mode))
    scope = evaluate_scope(retriever, questions, [0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    report = format_report(
        results,
        scope,
        sum(q.answerable for q in questions),
        sum(not q.answerable for q in questions),
        ", ".join(retriever.index.sources),
    )
    print(report)
    if args.output:
        args.output.write_text(report)
    for r in results:
        if r.misses:
            print(f"{r.mode.value} misses@10: {', '.join(r.misses)}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="backofthebook", description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        help="index directory (default: .backofthebook/index if it exists, else the bundled "
        "sample index in data/index)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="index course materials (PDF, PPTX, Markdown, text)")
    p.add_argument("paths", nargs="+", help="files or folders")
    p.add_argument("--chunk-words", type=int, default=180)
    p.add_argument("--overlap-words", type=int, default=40)
    p.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="answer a question with citations")
    p.add_argument("question")
    p.add_argument("--llm", choices=["claude", "groq", "extractive"])
    p.add_argument("-k", type=int, default=6, help="passages to give the model")
    p.set_defaults(func=cmd_ask)

    for name, func, help_text in [
        ("quiz", cmd_quiz, "practice quiz written only from your course materials"),
        ("anyquiz", cmd_anyquiz, "quiz on any topic, written from its Wikipedia article"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("topic")
        p.add_argument("-n", type=int, default=5, help="number of questions (1-10)")
        p.add_argument("-d", "--difficulty", default="medium", help="easy, medium, or hard")
        p.add_argument("--llm", choices=["claude", "groq"])
        p.add_argument(
            "--no-check", action="store_true", help="skip checking answers against sources"
        )
        p.set_defaults(func=func)

    p = sub.add_parser("eval", help="measure retrieval accuracy on a labeled question set")
    p.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    p.add_argument("--output", type=Path, help="write the Markdown report here")
    p.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a command. Coded errors print as `Error [CODE]: message` and exit with the
    category's exit code (see errors.py): 2 invalid input, 3 not in materials,
    4 configuration, 5 LLM service, 6 generation."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except BackOfTheBookError as e:
        retry = " (you can try again)" if e.retryable else ""
        print(f"\nError [{e.code}]: {e.message}{retry}", file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
