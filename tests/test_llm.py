"""Provider tests use fake SDK clients, so they check the exact requests without network calls."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import groq
import httpx
import httpx2
import pytest
from pydantic import BaseModel

from backofthebook.errors import ErrorCode
from backofthebook.llm import (
    CLAUDE_DEFAULT_MODEL,
    FALLBACK_BETA,
    ClaudeProvider,
    ExtractiveProvider,
    GroqProvider,
    LLMError,
    _claude_error,
    _groq_error,
    make_provider,
)


class Answer(BaseModel):
    value: int


# --- Claude -------------------------------------------------------------------------------


class FakeClaudeStream:
    def __init__(self, tokens: list[str], stop_reason: str) -> None:
        self.text_stream = iter(tokens)
        self._final = SimpleNamespace(stop_reason=stop_reason)

    def __enter__(self) -> FakeClaudeStream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get_final_message(self) -> Any:
        return self._final


class FakeClaudeClient:
    def __init__(self, tokens=("Hi", " there"), stop_reason="end_turn", parsed=None, raises=None):
        self.calls: list[dict[str, Any]] = []
        self._tokens, self._stop, self._parsed, self._raises = (
            list(tokens),
            stop_reason,
            parsed,
            raises,
        )
        self.beta = SimpleNamespace(
            messages=SimpleNamespace(stream=self._stream, parse=self._parse)
        )

    def _stream(self, **kwargs: Any) -> FakeClaudeStream:
        self.calls.append(kwargs)
        if self._raises:
            raise self._raises
        return FakeClaudeStream(self._tokens, self._stop)

    def _parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._raises:
            raise self._raises
        return SimpleNamespace(stop_reason=self._stop, parsed_output=self._parsed)


def test_claude_stream_request_uses_effort_and_server_side_fallback():
    client = FakeClaudeClient()
    text = "".join(ClaudeProvider(client=client, effort="high").stream_text("sys", "question"))

    assert text == "Hi there"
    call = client.calls[0]
    assert call["model"] == CLAUDE_DEFAULT_MODEL == "claude-opus-5-5"
    assert call["system"] == "sys"
    assert call["messages"] == [{"role": "user", "content": "question"}]
    assert call["output_config"] == {"effort": "high"}
    assert call["betas"] == [FALLBACK_BETA]
    assert call["fallbacks"] == "default"
    assert "thinking" not in call  # Opus 5.5 always thinks; effort is the control


def test_claude_stream_refusal_raises():
    provider = ClaudeProvider(client=FakeClaudeClient(stop_reason="refusal"))
    with pytest.raises(LLMError, match="declined") as raised:
        list(provider.stream_text("sys", "q"))
    assert raised.value.code is ErrorCode.LLM_REFUSED


def test_claude_generate_uses_structured_output_schema():
    client = FakeClaudeClient(parsed=Answer(value=4))
    result = ClaudeProvider(client=client).generate("sys", "2+2?", Answer)

    assert result == Answer(value=4)
    assert client.calls[0]["output_format"] is Answer
    assert client.calls[0]["fallbacks"] == "default"


@pytest.mark.parametrize(
    ("stop_reason", "parsed", "message"),
    [
        ("refusal", None, "declined"),
        ("max_tokens", None, "incomplete"),
        ("end_turn", None, "incomplete"),
    ],
)
def test_claude_generate_failures(stop_reason: str, parsed: Any, message: str):
    provider = ClaudeProvider(client=FakeClaudeClient(stop_reason=stop_reason, parsed=parsed))
    with pytest.raises(LLMError, match=message):
        provider.generate("sys", "q", Answer)


def _anthropic_status(status: int) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request)
    cls = {
        400: anthropic.BadRequestError,
        401: anthropic.AuthenticationError,
        429: anthropic.RateLimitError,
    }.get(status, anthropic.InternalServerError)
    return cls(f"status {status}", response=response, body=None)


@pytest.mark.parametrize(
    ("error", "code", "retryable"),
    [
        (_anthropic_status(401), ErrorCode.LLM_AUTH_FAILED, False),
        (_anthropic_status(429), ErrorCode.LLM_RATE_LIMITED, True),
        (_anthropic_status(500), ErrorCode.LLM_UNAVAILABLE, True),
        (_anthropic_status(400), ErrorCode.LLM_REQUEST_REJECTED, False),
        (
            anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x")),
            ErrorCode.LLM_UNAVAILABLE,
            True,
        ),
    ],
)
def test_claude_errors_get_codes(error: Exception, code: ErrorCode, retryable: bool):
    mapped = _claude_error(error)
    assert (mapped.code, mapped.retryable) == (code, retryable)
    provider = ClaudeProvider(client=FakeClaudeClient(raises=error))
    with pytest.raises(LLMError) as raised:
        list(provider.stream_text("sys", "q"))
    assert raised.value.code is code


# --- Groq ---------------------------------------------------------------------------------


class FakeGroqClient:
    def __init__(self, replies: list[str] | None = None, deltas=("A", None, "B")) -> None:
        self.calls: list[dict[str, Any]] = []
        self._replies = list(replies or [])
        self._deltas = deltas
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return [
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=d))])
                for d in self._deltas
            ]
        content = self._replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_groq_stream_skips_empty_deltas():
    client = FakeGroqClient()
    assert "".join(GroqProvider(client=client).stream_text("sys", "q")) == "AB"
    assert client.calls[0]["messages"][0] == {"role": "system", "content": "sys"}


def test_groq_generate_parses_fenced_json():
    client = FakeGroqClient(replies=['```json\n{"value": 3}\n```'])
    assert GroqProvider(client=client).generate("sys", "q", Answer) == Answer(value=3)
    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_groq_generate_repairs_invalid_json_once():
    client = FakeGroqClient(replies=['{"value": "three"}', '{"value": 3}'])
    assert GroqProvider(client=client).generate("sys", "q", Answer) == Answer(value=3)
    assert "invalid" in client.calls[1]["messages"][-1]["content"]


def test_groq_generate_gives_up_after_repair_attempt():
    client = FakeGroqClient(replies=["not json", "still not json"])
    with pytest.raises(LLMError) as raised:
        GroqProvider(client=client).generate("sys", "q", Answer)
    assert raised.value.code is ErrorCode.LLM_BAD_RESPONSE
    assert raised.value.retryable


def test_groq_errors_become_readable_messages():
    request = httpx.Request("POST", "https://api.groq.com")
    error = groq.AuthenticationError(
        "bad key", response=httpx.Response(401, request=request), body=None
    )
    assert _groq_error(error).code is ErrorCode.LLM_AUTH_FAILED
    assert "GROQ_API_KEY" in str(_groq_error(error))


# --- Extractive and selection -------------------------------------------------------------


def test_extractive_provider_quotes_sources_with_labels():
    prompt = (
        "Course-material excerpts:\n\n[S1] (a.pdf, p. 2)\nThe median is the middle.\n\n"
        "Student question: q"
    )
    text = "".join(ExtractiveProvider().stream_text("sys", prompt))
    assert "The median is the middle. [S1]" in text


def test_extractive_provider_cannot_make_quizzes():
    with pytest.raises(LLMError, match="needs an LLM") as raised:
        ExtractiveProvider().generate("sys", "q", Answer)
    assert raised.value.code is ErrorCode.NO_LLM_CONFIGURED


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"GROQ_API_KEY": "g", "ANTHROPIC_API_KEY": "a"}, "groq"),
        ({"ANTHROPIC_API_KEY": "a"}, "claude"),
        ({}, "extractive"),
        ({"GROQ_API_KEY": "g", "BACKOFTHEBOOK_LLM": "claude", "ANTHROPIC_API_KEY": "a"}, "claude"),
    ],
)
def test_make_provider_picks_by_available_key(monkeypatch: pytest.MonkeyPatch, env, expected):
    for var in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "BACKOFTHEBOOK_LLM"):
        monkeypatch.delenv(var, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert make_provider().name == expected


def test_make_provider_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown"):
        make_provider("gpt")
