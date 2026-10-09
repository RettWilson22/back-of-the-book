"""LLM providers behind one small interface: streamed text and schema-validated JSON.

- `ClaudeProvider` uses the Anthropic SDK (structured outputs + server-side refusal fallback).
- `GroqProvider` uses Groq's free tier (JSON mode + Pydantic validation with one repair retry).
- `ExtractiveProvider` stands for "no AI model": answers quote the retrieved passages
  themselves (see `AnswerEngine.quote_passages`), so the app and retrieval work offline.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from backofthebook.errors import BackOfTheBookError, ErrorCode

T = TypeVar("T", bound=BaseModel)

CLAUDE_DEFAULT_MODEL = "claude-opus-5-5"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
# Server-side fallback: if a request is declined by a safety classifier, the API re-runs it on
# Anthropic's recommended model for that refusal category instead of returning the refusal.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


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


def _status_error(provider: str, status: int, message: str) -> LLMError:
    details = {"provider": provider, "status": status}
    if status >= 500:
        return LLMError(
            ErrorCode.LLM_UNAVAILABLE,
            f"{provider} is having problems right now (HTTP {status}); try again shortly.",
            details=details,
        )
    return LLMError(
        ErrorCode.LLM_REQUEST_REJECTED,
        f"{provider} rejected the request (HTTP {status}): {message}",
        details=details,
    )


def _claude_error(e: Exception) -> LLMError:
    """Map Anthropic SDK exceptions (most specific first) to coded errors."""
    import anthropic

    if isinstance(e, anthropic.AuthenticationError):
        return LLMError(
            ErrorCode.LLM_AUTH_FAILED, "Claude rejected the API key; check ANTHROPIC_API_KEY."
        )
    if isinstance(e, anthropic.RateLimitError):
        return LLMError(
            ErrorCode.LLM_RATE_LIMITED, "Claude rate limit reached; wait a moment and try again."
        )
    if isinstance(e, anthropic.APIStatusError):
        return _status_error("Claude", e.status_code, e.message)
    if isinstance(e, anthropic.APIConnectionError):
        return LLMError(
            ErrorCode.LLM_UNAVAILABLE, "Could not reach the Claude API; check your connection."
        )
    return LLMError(ErrorCode.LLM_UNAVAILABLE, f"Claude request failed: {e}")


def _groq_error(e: Exception) -> LLMError:
    """Map Groq SDK exceptions (most specific first) to coded errors."""
    import groq

    if isinstance(e, groq.AuthenticationError):
        return LLMError(ErrorCode.LLM_AUTH_FAILED, "Groq rejected the API key; check GROQ_API_KEY.")
    if isinstance(e, groq.RateLimitError):
        return LLMError(
            ErrorCode.LLM_RATE_LIMITED, "Groq rate limit reached; wait a moment and try again."
        )
    if isinstance(e, groq.APIStatusError):
        return _status_error("Groq", e.status_code, e.message)
    if isinstance(e, groq.APIConnectionError):
        return LLMError(
            ErrorCode.LLM_UNAVAILABLE, "Could not reach the Groq API; check your connection."
        )
    return LLMError(ErrorCode.LLM_UNAVAILABLE, f"Groq request failed: {e}")


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
                max_tokens=64000,
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
                max_tokens=16000,
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
        schema_hint = "\n\nRespond with only a JSON object matching this JSON Schema:\n" + str(
            schema.model_json_schema()
        )
        messages = [
            {"role": "system", "content": system + schema_hint},
            {"role": "user", "content": user},
        ]
        import groq

        last_error: ValidationError | None = None
        for _ in range(2):  # one repair attempt with the validation error as feedback
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )
            except groq.APIError as e:
                raise _groq_error(e) from e
            content = response.choices[0].message.content or ""
            try:
                return schema.model_validate_json(_extract_json(content))
            except ValidationError as e:
                last_error = e
                messages += [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": f"That JSON was invalid: {e}. Return corrected JSON.",
                    },
                ]
        raise LLMError(
            ErrorCode.LLM_BAD_RESPONSE,
            "Groq's response wasn't in the expected format, even after a retry. Try again.",
            details={"validation_error": str(last_error)},
        )


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


def make_provider(name: str | None = None) -> LLMProvider:
    """Pick a provider by name, or by whichever API key is set (Groq first, then Claude)."""
    name = (name or os.environ.get("BACKOFTHEBOOK_LLM") or "").lower()
    if not name:
        if os.environ.get("GROQ_API_KEY"):
            name = "groq"
        elif os.environ.get("ANTHROPIC_API_KEY"):
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
