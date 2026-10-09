"""LLM providers behind one small interface: streamed text and schema-validated JSON.

- `ClaudeProvider` uses the Anthropic SDK (structured outputs + server-side refusal fallback).
- `GroqProvider` uses Groq's free tier (JSON mode + Pydantic validation with one repair retry).
- `ExtractiveProvider` stands for "no AI model": answers quote the retrieved passages
  themselves (see `AnswerEngine.quote_passages`), so the app and retrieval work offline.
- `BudgetedProvider` wraps another provider and spends from a call budget before every
  request; `CallLimiter` is a budget of calls per minute that threads can share.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from backofthebook.errors import BackOfTheBookError, ErrorCode

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)

CLAUDE_DEFAULT_MODEL = "claude-opus-5-5"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
# Server-side fallback: if a request is declined by a safety classifier, the API re-runs it on
# Anthropic's recommended model for that refusal category instead of returning the refusal.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Most tokens one request may write (reasoning included). Answers and quizzes need far less;
# the caps bound what a single request can cost on a public deployment.
CLAUDE_CHAT_MAX_TOKENS = 4096
CLAUDE_QUIZ_MAX_TOKENS = 8192
GROQ_CHAT_MAX_TOKENS = 2048
GROQ_QUIZ_MAX_TOKENS = 4096


class LLMError(BackOfTheBookError):
    """A provider failure, with an error code and a message that is safe to show to the user."""


Message = dict[str, str]  # {"role": "user" | "assistant", "content": "..."}


@dataclass(frozen=True)
class StreamEvent:
    """A piece of a streamed reply: the model's reasoning summary, or the answer itself."""

    kind: Literal["thinking", "text"]
    text: str


class LLMProvider(Protocol):
    name: str

    def chat_stream(self, system: str, messages: list[Message]) -> Iterator[StreamEvent]:
        """Stream a reply to a conversation (oldest message first, ending with the user)."""
        ...

    def generate(self, system: str, user: str, schema: type[T]) -> T: ...


def _status_error(provider: str, status: int) -> LLMError:
    # The provider's own message can include account or request details, so it is only
    # logged (see _claude_error and _groq_error), never shown.
    details = {"provider": provider, "status": status}
    if status >= 500:
        return LLMError(
            ErrorCode.LLM_UNAVAILABLE,
            f"{provider} is having problems right now; try again shortly.",
            details=details,
        )
    return LLMError(
        ErrorCode.LLM_REQUEST_REJECTED,
        f"{provider} couldn't handle this request.",
        details=details,
    )


def _claude_error(e: Exception) -> LLMError:
    """Map Anthropic SDK exceptions (most specific first) to coded errors."""
    import anthropic

    logger.warning("Claude request failed: %r", e)

    if isinstance(e, anthropic.AuthenticationError):
        return LLMError(
            ErrorCode.LLM_AUTH_FAILED, "Claude rejected the API key; check ANTHROPIC_API_KEY."
        )
    if isinstance(e, anthropic.RateLimitError):
        return LLMError(
            ErrorCode.LLM_RATE_LIMITED, "Claude rate limit reached; wait a moment and try again."
        )
    if isinstance(e, anthropic.APIStatusError):
        return _status_error("Claude", e.status_code)
    if isinstance(e, anthropic.APIConnectionError):
        return LLMError(
            ErrorCode.LLM_UNAVAILABLE, "Could not reach the Claude API; check your connection."
        )
    return LLMError(ErrorCode.LLM_UNAVAILABLE, "The request to Claude failed; try again shortly.")


def _groq_error(e: Exception) -> LLMError:
    """Map Groq SDK exceptions (most specific first) to coded errors."""
    import groq

    logger.warning("Groq request failed: %r", e)
    if isinstance(e, groq.AuthenticationError):
        return LLMError(ErrorCode.LLM_AUTH_FAILED, "Groq rejected the API key; check GROQ_API_KEY.")
    if isinstance(e, groq.RateLimitError):
        return LLMError(
            ErrorCode.LLM_RATE_LIMITED, "Groq rate limit reached; wait a moment and try again."
        )
    if isinstance(e, groq.APIStatusError):
        return _status_error("Groq", e.status_code)
    if isinstance(e, groq.APIConnectionError):
        return LLMError(
            ErrorCode.LLM_UNAVAILABLE, "Could not reach the Groq API; check your connection."
        )
    return LLMError(ErrorCode.LLM_UNAVAILABLE, "The request to Groq failed; try again shortly.")


class ClaudeProvider:
    name = "claude"

    def __init__(
        self, model: str = CLAUDE_DEFAULT_MODEL, effort: str = "medium", client: Any = None
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort

    def chat_stream(self, system: str, messages: list[Message]) -> Iterator[StreamEvent]:
        import anthropic

        try:
            with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=CLAUDE_CHAT_MAX_TOKENS,
                system=system,
                messages=messages,
                # Opus 5.5 always thinks; "summarized" returns a readable summary to show users.
                thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": self.effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            ) as stream:
                for event in stream:
                    if event.type != "content_block_delta":
                        continue
                    if event.delta.type == "thinking_delta" and event.delta.thinking:
                        yield StreamEvent("thinking", event.delta.thinking)
                    elif event.delta.type == "text_delta" and event.delta.text:
                        yield StreamEvent("text", event.delta.text)
                final = stream.get_final_message()
        except anthropic.APIError as e:
            raise _claude_error(e) from e
        if final.stop_reason == "refusal":
            raise LLMError(ErrorCode.LLM_REFUSED, "Claude declined to answer this request.")

    def generate(self, system: str, user: str, schema: type[T]) -> T:
        import anthropic

        try:
            response = self.client.beta.messages.parse(
                model=self.model,
                max_tokens=CLAUDE_QUIZ_MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.effort},
                output_format=schema,
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.APIError as e:
            raise _claude_error(e) from e
        if response.stop_reason == "refusal":
            raise LLMError(ErrorCode.LLM_REFUSED, "Claude declined to generate this content.")
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise LLMError(
                ErrorCode.LLM_BAD_RESPONSE,
                "Claude's response was incomplete; try fewer questions.",
                details={"stop_reason": response.stop_reason},
            )
        parsed: T = response.parsed_output
        return parsed


def _extract_json(text: str) -> str:
    """Strip Markdown code fences some models wrap around JSON."""
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1) if match else text


class GroqProvider:
    name = "groq"

    def __init__(self, model: str = GROQ_DEFAULT_MODEL, client: Any = None) -> None:
        if client is None:
            import groq

            client = groq.Groq()
        self.client = client
        self.model = model

    @property
    def _reasons(self) -> bool:
        """gpt-oss models on Groq can return their reasoning alongside the answer."""
        return self.model.startswith("openai/gpt-oss")

    def chat_stream(self, system: str, messages: list[Message]) -> Iterator[StreamEvent]:
        import groq

        extra: dict[str, Any] = (
            {"include_reasoning": True, "reasoning_effort": "medium"} if self._reasons else {}
        )
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, *messages],
                temperature=0.3,
                max_completion_tokens=GROQ_CHAT_MAX_TOKENS,
                stream=True,
                **extra,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if getattr(delta, "reasoning", None):
                    yield StreamEvent("thinking", delta.reasoning)
                if delta.content:
                    yield StreamEvent("text", delta.content)
        except groq.APIError as e:
            raise _groq_error(e) from e

    def generate(self, system: str, user: str, schema: type[T]) -> T:
        schema_hint = "\n\nRespond with only a JSON object matching this JSON Schema:\n" + (
            json.dumps(schema.model_json_schema())
        )
        messages = [
            {"role": "system", "content": system + schema_hint},
            {"role": "user", "content": user},
        ]
        import groq

        problem = ""
        for _ in range(2):  # one repair attempt with the problem as feedback
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    max_completion_tokens=GROQ_QUIZ_MAX_TOKENS,
                    response_format={"type": "json_object"},
                )
            except groq.BadRequestError as e:
                failed = _rejected_json(e)
                if failed is None:
                    raise _groq_error(e) from e
                content, problem = failed, "it didn't parse as JSON"
            except groq.APIError as e:
                raise _groq_error(e) from e
            else:
                content = response.choices[0].message.content or ""
                try:
                    return schema.model_validate_json(_extract_json(content))
                except ValidationError as e:
                    problem = str(e)
            messages = [
                *messages,
                {"role": "assistant", "content": content or "(no output)"},
                {
                    "role": "user",
                    "content": f"That JSON was invalid: {problem}. Return corrected JSON.",
                },
            ]
        raise LLMError(
            ErrorCode.LLM_BAD_RESPONSE,
            "Groq's response wasn't in the expected format, even after a retry. Try again.",
            details={"validation_error": problem},
        )


def _rejected_json(error: Exception) -> str | None:
    """When Groq's JSON mode refused the model's own output (HTTP 400 with the code
    json_validate_failed), the output it refused; otherwise None. That is a bad response from
    the model, not a bad request, so it gets the same repair retry as invalid JSON."""
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        body = body.get("error", body)
    if isinstance(body, dict) and body.get("code") == "json_validate_failed":
        return str(body.get("failed_generation") or "")
    return None


class ExtractiveProvider:
    """No-LLM mode. AnswerEngine answers by quoting passages itself, so this never writes."""

    name = "extractive"

    def chat_stream(self, system: str, messages: list[Message]) -> Iterator[StreamEvent]:
        raise LLMError(
            ErrorCode.NO_LLM_CONFIGURED,
            "Chat needs an LLM. Set GROQ_API_KEY or ANTHROPIC_API_KEY.",
        )

    def generate(self, system: str, user: str, schema: type[T]) -> T:
        raise LLMError(
            ErrorCode.NO_LLM_CONFIGURED,
            "Quiz generation needs an LLM. Set GROQ_API_KEY or ANTHROPIC_API_KEY.",
        )


class CallLimiter:
    """Allows at most `limit` calls in any `window` seconds, counted across all threads."""

    def __init__(
        self, limit: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limit = limit
        self.window = window
        self._clock = clock
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        """Count one call and return True, or return False if the limit is reached."""
        with self._lock:
            now = self._clock()
            while self._calls and now - self._calls[0] >= self.window:
                self._calls.popleft()
            if len(self._calls) >= self.limit:
                return False
            self._calls.append(now)
            return True


class BudgetedProvider:
    """A provider that calls `spend` before every request. `spend` raises (usually an
    LLMError with code USAGE_LIMIT) to stop the request from being made."""

    def __init__(self, inner: LLMProvider, spend: Callable[[], None]) -> None:
        self.inner = inner
        self.name = inner.name
        self._spend = spend

    def chat_stream(self, system: str, messages: list[Message]) -> Iterator[StreamEvent]:
        self._spend()  # now, not when the stream is first read
        return self.inner.chat_stream(system, messages)

    def generate(self, system: str, user: str, schema: type[T]) -> T:
        self._spend()
        return self.inner.generate(system, user, schema)


def make_provider(name: str | None = None, *, allow_claude: bool = True) -> LLMProvider:
    """Pick a provider by name, or by whichever API key is set (Groq first, then Claude).

    With `allow_claude=False` Claude is never used, even if it is named or its key is the only
    one set; the choice falls back to Groq or to no AI.
    """
    name = (name or os.environ.get("BACKOFTHEBOOK_LLM") or "").lower()
    if name == "claude" and not allow_claude:
        name = ""
    if not name:
        if os.environ.get("GROQ_API_KEY"):
            name = "groq"
        elif os.environ.get("ANTHROPIC_API_KEY") and allow_claude:
            name = "claude"
        else:
            name = "extractive"
    if name == "claude":
        return ClaudeProvider(
            model=os.environ.get("BACKOFTHEBOOK_CLAUDE_MODEL", CLAUDE_DEFAULT_MODEL),
            effort=os.environ.get("BACKOFTHEBOOK_CLAUDE_EFFORT", "medium"),
        )
    if name == "groq":
        return GroqProvider(model=os.environ.get("BACKOFTHEBOOK_GROQ_MODEL", GROQ_DEFAULT_MODEL))
    if name == "extractive":
        return ExtractiveProvider()
    raise ValueError(f"unknown LLM provider {name!r} (use claude, groq, or extractive)")
