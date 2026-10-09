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
    StreamEvent,
    _claude_error,
    _groq_error,
    make_provider,
)


class Answer(BaseModel):
    value: int


# --- Claude -------------------------------------------------------------------------------


def _delta(kind: str, value: str) -> SimpleNamespace:
    field = "thinking" if kind == "thinking_delta" else "text"
    return SimpleNamespace(
        type="content_block_delta", delta=SimpleNamespace(type=kind, **{field: value})
    )


class FakeClaudeStream:
    def __init__(self, tokens: list[str], stop_reason: str) -> None:
        self._events = [
            SimpleNamespace(type="content_block_start"),
            _delta("thinking_delta", "Let me think."),
            *[_delta("text_delta", t) for t in tokens],
        ]
        self._final = SimpleNamespace(stop_reason=stop_reason)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._events)

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


CONVERSATION = [
    {"role": "user", "content": "first"},
    {"role": "assistant", "content": "reply"},
    {"role": "user", "content": "question"},
]


def test_claude_streams_thinking_then_text_with_effort_and_fallback():
    client = FakeClaudeClient()
    events = list(ClaudeProvider(client=client, effort="high").chat_stream("sys", CONVERSATION))

    assert events[0] == StreamEvent("thinking", "Let me think.")
    assert "".join(e.text for e in events if e.kind == "text") == "Hi there"
    call = client.calls[0]
    assert call["model"] == CLAUDE_DEFAULT_MODEL == "claude-opus-5-5"
    assert call["system"] == "sys"
    assert call["messages"] == CONVERSATION
    assert call["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert call["output_config"] == {"effort": "high"}
    assert call["betas"] == [FALLBACK_BETA]
    assert call["fallbacks"] == "default"
    assert call["max_tokens"] == 4096


def test_claude_stream_refusal_raises():
    provider = ClaudeProvider(client=FakeClaudeClient(stop_reason="refusal"))
    with pytest.raises(LLMError, match="declined") as raised:
        list(provider.chat_stream("sys", CONVERSATION))
    assert raised.value.code is ErrorCode.LLM_REFUSED


def test_claude_generate_uses_structured_output_schema():
    client = FakeClaudeClient(parsed=Answer(value=4))
    result = ClaudeProvider(client=client).generate("sys", "2+2?", Answer)

    assert result == Answer(value=4)
    assert client.calls[0]["output_format"] is Answer
    assert client.calls[0]["max_tokens"] == 8192
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
        list(provider.chat_stream("sys", CONVERSATION))
    assert raised.value.code is code


# --- Groq ---------------------------------------------------------------------------------


class FakeGroqClient:
    def __init__(
        self,
        replies: list[str] | None = None,
        deltas=(("Hmm", None), (None, "A"), (None, None), (None, "B")),
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._replies = list(replies or [])
        self._deltas = deltas
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return [
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(reasoning=r, content=c))]
                )
                for r, c in self._deltas
            ]
        content = self._replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_groq_streams_reasoning_and_text_and_skips_empty_deltas():
    client = FakeGroqClient()
    events = list(GroqProvider(client=client).chat_stream("sys", CONVERSATION))

    assert events == [
        StreamEvent("thinking", "Hmm"),
        StreamEvent("text", "A"),
        StreamEvent("text", "B"),
    ]
    call = client.calls[0]
    assert call["messages"] == [{"role": "system", "content": "sys"}, *CONVERSATION]
    assert call["include_reasoning"] is True
    assert call["max_completion_tokens"] == 2048


def test_groq_only_requests_reasoning_from_models_that_support_it():
    client = FakeGroqClient()
    list(
        GroqProvider(model="llama-3.3-70b-versatile", client=client).chat_stream(
            "sys", CONVERSATION
        )
    )
    assert "include_reasoning" not in client.calls[0]


def test_groq_generate_parses_fenced_json():
    client = FakeGroqClient(replies=['```json\n{"value": 3}\n```'])
    assert GroqProvider(client=client).generate("sys", "q", Answer) == Answer(value=3)
    assert client.calls[0]["response_format"] == {"type": "json_object"}
    assert client.calls[0]["max_completion_tokens"] == 4096


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


def test_extractive_provider_cannot_chat_or_make_quizzes():
    """The answer engine quotes passages itself in no-AI mode; the provider never writes."""
    with pytest.raises(LLMError, match="needs an LLM") as raised:
        ExtractiveProvider().chat_stream("sys", [{"role": "user", "content": "q"}])
    assert raised.value.code is ErrorCode.NO_LLM_CONFIGURED
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


@pytest.mark.parametrize(
    ("env", "name", "expected"),
    [
        ({"ANTHROPIC_API_KEY": "a"}, None, "extractive"),
        ({"GROQ_API_KEY": "g", "ANTHROPIC_API_KEY": "a"}, None, "groq"),
        ({"GROQ_API_KEY": "g", "ANTHROPIC_API_KEY": "a"}, "claude", "groq"),
        ({"ANTHROPIC_API_KEY": "a", "BACKOFTHEBOOK_LLM": "claude"}, None, "extractive"),
    ],
)
def test_make_provider_never_uses_claude_unless_allowed(
    monkeypatch: pytest.MonkeyPatch, env, name, expected
):
    for var in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "BACKOFTHEBOOK_LLM"):
        monkeypatch.delenv(var, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert make_provider(name, allow_claude=False).name == expected


def test_make_provider_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown"):
        make_provider("gpt")
