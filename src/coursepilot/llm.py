"""LLM providers behind one small interface: streamed text and schema-validated JSON.

- `ClaudeProvider` uses the Anthropic SDK (structured outputs + server-side refusal fallback).
- `GroqProvider` uses Groq's free tier (JSON mode + Pydantic validation with one repair retry).
- `ExtractiveProvider` needs no API key: it returns the retrieved passages themselves, so the
  app and retrieval can be tried offline.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

CLAUDE_DEFAULT_MODEL = "claude-opus-5-5"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
# Server-side fallback: if a request is declined by a safety classifier, the API re-runs it on
# Anthropic's recommended model for that refusal category instead of returning the refusal.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """A provider failure with a message that is safe to show to the user."""


class LLMProvider(Protocol):
    name: str

    def stream_text(self, system: str, user: str) -> Iterator[str]: ...

    def generate(self, system: str, user: str, schema: type[T]) -> T: ...


def _claude_error(e: Exception) -> LLMError:
    """Map Anthropic SDK exceptions (most specific first) to a user-facing message."""
    import anthropic

    if isinstance(e, anthropic.AuthenticationError):
        return LLMError("Claude rejected the API key; check ANTHROPIC_API_KEY.")
    if isinstance(e, anthropic.RateLimitError):
        return LLMError("Claude rate limit reached; wait a moment and try again.")
    if isinstance(e, anthropic.APIStatusError):
        return LLMError(f"Claude API error {e.status_code}: {e.message}")
    if isinstance(e, anthropic.APIConnectionError):
        return LLMError("Could not reach the Claude API; check your connection.")
    return LLMError(f"Claude request failed: {e}")


def _groq_error(e: Exception) -> LLMError:
    """Map Groq SDK exceptions (most specific first) to a user-facing message."""
    import groq

    if isinstance(e, groq.AuthenticationError):
        return LLMError("Groq rejected the API key; check GROQ_API_KEY.")
    if isinstance(e, groq.RateLimitError):
        return LLMError("Groq rate limit reached; wait a moment and try again.")
    if isinstance(e, groq.APIStatusError):
        return LLMError(f"Groq API error {e.status_code}: {e.message}")
    if isinstance(e, groq.APIConnectionError):
        return LLMError("Could not reach the Groq API; check your connection.")
    return LLMError(f"Groq request failed: {e}")


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

    def stream_text(self, system: str, user: str) -> Iterator[str]:
        import anthropic

        try:
            with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=64000,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            ) as stream:
                yield from stream.text_stream
                final = stream.get_final_message()
        except anthropic.APIError as e:
            raise _claude_error(e) from e
        if final.stop_reason == "refusal":
            raise LLMError("Claude declined to answer this request.")

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
            raise LLMError("Claude declined to generate this content.")
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise LLMError("Claude's response was incomplete; try a smaller request.")
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

    def stream_text(self, system: str, user: str) -> Iterator[str]:
        import groq

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.2,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
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
        raise LLMError(f"Groq returned JSON that did not match the schema: {last_error}")


class ExtractiveProvider:
    """No-LLM mode: answers by quoting the retrieved sources the prompt already contains."""

    name = "extractive"

    def stream_text(self, system: str, user: str) -> Iterator[str]:
        # Parses the prompt format produced by answer.build_prompt: "[Sn] (citation)\ntext".
        sources = re.findall(
            r"^\[S(\d+)\] \(([^)]*)\)\n(.*?)(?=^\[S\d+\]|^Student question:|\Z)", user, re.M | re.S
        )
        if not sources:
            yield "No matching passages were found in your course materials."
            return
        yield "No LLM is configured, so here are the most relevant passages:\n\n"
        for number, _citation, text in sources[:3]:
            snippet = " ".join(text.split())
            yield f"- {snippet[:400]}{'…' if len(snippet) > 400 else ''} [S{number}]\n"

    def generate(self, system: str, user: str, schema: type[T]) -> T:
        raise LLMError("Quiz generation needs an LLM. Set GROQ_API_KEY or ANTHROPIC_API_KEY.")


def make_provider(name: str | None = None) -> LLMProvider:
    """Pick a provider by name, or by whichever API key is set (Groq first, then Claude)."""
    name = (name or os.environ.get("COURSEPILOT_LLM") or "").lower()
    if not name:
        if os.environ.get("GROQ_API_KEY"):
            name = "groq"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            name = "claude"
        else:
            name = "extractive"
    if name == "claude":
        return ClaudeProvider(
            model=os.environ.get("COURSEPILOT_CLAUDE_MODEL", CLAUDE_DEFAULT_MODEL),
            effort=os.environ.get("COURSEPILOT_CLAUDE_EFFORT", "medium"),
        )
    if name == "groq":
        return GroqProvider(model=os.environ.get("COURSEPILOT_GROQ_MODEL", GROQ_DEFAULT_MODEL))
    if name == "extractive":
        return ExtractiveProvider()
    raise ValueError(f"unknown LLM provider {name!r} (use claude, groq, or extractive)")
