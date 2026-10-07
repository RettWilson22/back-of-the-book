"""Command-line interface: `coursepilot ingest | ask | quiz | eval`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coursepilot.retrieval import Retriever

DEFAULT_INDEX = Path(".coursepilot/index")


def _retriever(index_dir: Path, rerank: bool = True) -> Retriever:
    from coursepilot.index import CorpusIndex, SentenceTransformerEmbedder
    from coursepilot.retrieval import CrossEncoderReranker, Retriever

    index = CorpusIndex.load(index_dir)
    return Retriever(
        index,
        SentenceTransformerEmbedder(index.embedding_model),
        CrossEncoderReranker() if rerank else None,
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    from coursepilot.chunking import chunk_pages
    from coursepilot.documents import find_documents, load_document
    from coursepilot.index import CorpusIndex, SentenceTransformerEmbedder

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
    index.save(args.index)
    print(f"Saved index to {args.index}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from coursepilot.answer import AnswerEngine
    from coursepilot.llm import LLMError, make_provider

    engine = AnswerEngine(_retriever(args.index), make_provider(args.llm), k=args.k)
    answer, tokens = engine.stream(args.question)
    try:
        for token in tokens:
            print(token, end="", flush=True)
    except LLMError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    print("\n")
    for n, hit in answer.cited_hits:
        print(f"  [S{n}] {hit.chunk.citation}")
    if answer.invalid_citations:
        print(f"  (ignored citations to nonexistent sources: {answer.invalid_citations})")
    return 0


def cmd_quiz(args: argparse.Namespace) -> int:
    from coursepilot.llm import LLMError, make_provider
    from coursepilot.quiz import generate_quiz

    try:
        quiz = generate_quiz(_retriever(args.index), make_provider(args.llm), args.topic, args.n)
    except LLMError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    for i, q in enumerate(quiz.questions, start=1):
        print(f"\n{i}. {q.question}")
        for letter, choice in zip("ABCD", q.choices, strict=True):
            print(f"   {letter}) {choice}")
        sources = ", ".join(h.chunk.citation for h in quiz.sources_for(q))
        print(f"   Answer: {'ABCD'[q.answer_index]}. {q.explanation} ({sources})")
    if quiz.dropped:
        print(f"\n({quiz.dropped} generated question(s) failed validation and were dropped)")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from coursepilot.evaluation import evaluate_mode, evaluate_scope, format_report, load_questions
    from coursepilot.retrieval import Mode

    questions = load_questions(args.questions)
    retriever = _retriever(args.index)
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
    parser = argparse.ArgumentParser(prog="coursepilot", description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="index directory")
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

    p = sub.add_parser("quiz", help="generate a practice quiz on a topic")
    p.add_argument("topic")
    p.add_argument("-n", type=int, default=5, help="number of questions")
    p.add_argument("--llm", choices=["claude", "groq"])
    p.set_defaults(func=cmd_quiz)

    p = sub.add_parser("eval", help="measure retrieval accuracy on a labeled question set")
    p.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    p.add_argument("--output", type=Path, help="write the Markdown report here")
    p.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
