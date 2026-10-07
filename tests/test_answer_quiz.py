import pytest
from conftest import FakeLLM

from backofthebook.answer import (
    NOT_FOUND_MESSAGE,
    AnswerEngine,
    Turn,
    build_prompt,
    extract_citations,
    retrieval_query,
)
from backofthebook.llm import ExtractiveProvider
from backofthebook.quiz import (
    Quiz,
    QuizDraft,
    QuizQuestion,
    TopicNotCovered,
    generate_quiz,
    validate_question,
)
from backofthebook.retrieval import Mode, Retriever


def test_extract_citations_splits_valid_and_invalid_and_dedupes():
    valid, invalid = extract_citations("A [S2]. B [S1][S2]. C [S9]. D [S0].", num_sources=3)
    assert valid == [2, 1]
    assert invalid == [9, 0]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Seeds matter (S3). Points move (S1, S2).", [3, 1, 2]),  # format seen from Groq
        ("Seeds matter [S2, S4] and [S1; S3].", [2, 4, 1, 3]),
        ("No citations, S1 mentioned bare, and k-means (k) in parentheses.", []),
    ],
)
def test_extract_citations_accepts_common_formats(text: str, expected: list[int]):
    assert extract_citations(text, num_sources=5)[0] == expected


def test_prompt_labels_sources_with_citations(retriever: Retriever):
    hits = retriever.search("median", k=2, mode=Mode.HYBRID)
    prompt = build_prompt("What is the median?", hits)
    assert prompt.startswith("Course-material excerpts:")
    assert f"[S1] ({hits[0].chunk.citation})" in prompt
    assert prompt.endswith("Student question: What is the median?")


def test_off_topic_question_is_answered_from_general_knowledge(retriever: Retriever):
    llm = FakeLLM("This isn't covered in your materials, so here's a general answer.")
    answer = AnswerEngine(retriever, llm, min_similarity=0.3).ask("volcano eruption lava")

    assert not answer.grounded
    assert answer.sources == [] and answer.cited == []
    assert "No relevant excerpts were found" in llm.prompts[0][1]


def test_off_topic_question_without_an_llm_says_it_is_not_in_the_materials(retriever: Retriever):
    answer = AnswerEngine(retriever, ExtractiveProvider(), min_similarity=0.3).ask("volcano lava")
    assert not answer.grounded
    assert answer.text == NOT_FOUND_MESSAGE


def test_answer_streams_thinking_and_text_and_checks_citations(retriever: Retriever):
    llm = FakeLLM("The median is the middle value [S1]. Also see [S7].", thinking="Look at S1.")
    engine = AnswerEngine(retriever, llm, k=3, min_similarity=0.0)

    answer, events = engine.stream("what is the median of sorted data")
    kinds = [e.kind for e in events]

    assert kinds[0] == "thinking" and set(kinds[1:]) == {"text"}
    assert answer.text == llm.reply
    assert answer.thinking == "Look at S1."
    assert answer.thinking_seconds >= 0
    assert answer.grounded
    assert answer.cited == [1]
    assert answer.invalid_citations == [7]
    assert answer.cited_hits[0][1].chunk.page == 2
    assert "Student question: what is the median of sorted data" in llm.prompts[0][1]


def test_follow_up_questions_see_the_conversation_and_reuse_context(retriever: Retriever):
    llm = FakeLLM("It's the middle value [S1].")
    engine = AnswerEngine(retriever, llm, min_similarity=0.0)
    history = [Turn("What is the median of sorted data?", "The middle value.")]

    engine.ask("why?", history)

    conversation = llm.conversations[0]
    assert [m["role"] for m in conversation] == ["user", "assistant", "user"]
    assert conversation[0]["content"] == "What is the median of sorted data?"
    assert retrieval_query("why?", history) == "What is the median of sorted data? why?"
    assert retrieval_query("Explain how k-means picks its centroids", history).startswith("Explain")


def test_referencing_a_document_limits_sources_to_it(retriever: Retriever):
    llm = FakeLLM("Answer [S1].")
    answer = AnswerEngine(retriever, llm, min_similarity=0.0).ask(
        "decision tree questions", sources=["stats.pdf"]
    )
    assert answer.sources and {h.chunk.source for h in answer.sources} == {"stats.pdf"}


def test_engine_falls_back_to_hybrid_without_a_reranker(retriever: Retriever):
    retriever.reranker = None
    assert AnswerEngine(retriever, FakeLLM()).mode is Mode.HYBRID


def question(**overrides) -> QuizQuestion:
    fields = {
        "question": "What does the median measure?",
        "choices": ["Middle value", "Average", "Spread", "Mode"],
        "answer_index": 0,
        "explanation": "It is the middle of sorted data.",
        "sources": ["S1"],
    }
    return QuizQuestion(**{**fields, **overrides})


def test_validate_question_accepts_a_good_question():
    assert validate_question(question(), num_sources=2)


def test_validate_question_rejects_bad_questions():
    assert not validate_question(question(choices=["A", "B", "C"]), 2)
    assert not validate_question(question(choices=["A", "a", "B", "C"]), 2)
    assert not validate_question(question(answer_index=4), 2)
    assert not validate_question(question(sources=["S3"]), 2)
    assert not validate_question(question(sources=[]), 2)
    assert not validate_question(question(explanation=" "), 2)


def test_generate_quiz_drops_invalid_questions_and_maps_sources(retriever: Retriever):
    draft = QuizDraft(
        questions=[
            question(),
            question(answer_index=9),
            question(question="Which value is in the middle?", sources=["s2"]),
            question(question="What does the median measure? "),  # repeat of the first
        ]
    )
    quiz = generate_quiz(retriever, FakeLLM(structured=draft), "median", n=5, check=False, k=3)

    assert isinstance(quiz, Quiz)
    assert len(quiz.questions) == 2
    assert quiz.dropped == 2
    assert quiz.sources_for(quiz.questions[1]) == [quiz.passages[1]]
    assert quiz.source_title == "your course materials"


def test_quiz_on_a_topic_outside_the_materials_never_calls_the_llm(retriever: Retriever):
    llm = FakeLLM(structured=QuizDraft(questions=[question()]))
    with pytest.raises(TopicNotCovered, match="don't cover"):
        generate_quiz(retriever, llm, "mario galaxy", min_similarity=0.3)
    assert llm.prompts == []
