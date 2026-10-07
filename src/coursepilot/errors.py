"""Error codes shared by the library, the CLI, and the web app.

Every failure a user can hit is a `CoursePilotError` with:
- `code`: a stable name (`LLM_RATE_LIMITED`) that callers can branch on and users can report;
- `message`: a sentence that is safe to show to a user;
- `retryable`: whether trying the same request again could succeed;
- `exit_code`: the CLI exit status for the error's category;
- `details`: optional machine-readable context (e.g. the HTTP status from a provider).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    # The request itself is invalid; fix the input.
    EMPTY_TOPIC = "EMPTY_TOPIC"
    TOPIC_TOO_LONG = "TOPIC_TOO_LONG"
    INVALID_QUESTION_COUNT = "INVALID_QUESTION_COUNT"
    INVALID_DIFFICULTY = "INVALID_DIFFICULTY"
    # The loaded course materials can't support the request.
    TOPIC_NOT_COVERED = "TOPIC_NOT_COVERED"
    # The LLM provider failed or isn't set up.
    NO_LLM_CONFIGURED = "NO_LLM_CONFIGURED"
    LLM_AUTH_FAILED = "LLM_AUTH_FAILED"
    LLM_RATE_LIMITED = "LLM_RATE_LIMITED"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    LLM_REQUEST_REJECTED = "LLM_REQUEST_REJECTED"
    LLM_REFUSED = "LLM_REFUSED"
    LLM_BAD_RESPONSE = "LLM_BAD_RESPONSE"
    # The LLM answered, but nothing usable came out of it.
    NO_VALID_QUESTIONS = "NO_VALID_QUESTIONS"
    # A bug: anything not anticipated above.
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True)
class ErrorInfo:
    category: str
    retryable: bool
    exit_code: int


_INPUT = ErrorInfo("invalid request", retryable=False, exit_code=2)

CATALOG: dict[ErrorCode, ErrorInfo] = {
    ErrorCode.EMPTY_TOPIC: _INPUT,
    ErrorCode.TOPIC_TOO_LONG: _INPUT,
    ErrorCode.INVALID_QUESTION_COUNT: _INPUT,
    ErrorCode.INVALID_DIFFICULTY: _INPUT,
    ErrorCode.TOPIC_NOT_COVERED: ErrorInfo("not in materials", retryable=False, exit_code=3),
    ErrorCode.NO_LLM_CONFIGURED: ErrorInfo("configuration", retryable=False, exit_code=4),
    ErrorCode.LLM_AUTH_FAILED: ErrorInfo("configuration", retryable=False, exit_code=4),
    ErrorCode.LLM_RATE_LIMITED: ErrorInfo("LLM service", retryable=True, exit_code=5),
    ErrorCode.LLM_UNAVAILABLE: ErrorInfo("LLM service", retryable=True, exit_code=5),
    ErrorCode.LLM_REQUEST_REJECTED: ErrorInfo("LLM service", retryable=False, exit_code=5),
    ErrorCode.LLM_REFUSED: ErrorInfo("LLM service", retryable=False, exit_code=5),
    ErrorCode.LLM_BAD_RESPONSE: ErrorInfo("LLM service", retryable=True, exit_code=5),
    ErrorCode.NO_VALID_QUESTIONS: ErrorInfo("generation", retryable=True, exit_code=6),
    ErrorCode.INTERNAL_ERROR: ErrorInfo("internal", retryable=False, exit_code=1),
}


class CoursePilotError(Exception):
    def __init__(
        self, code: ErrorCode, message: str, *, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    @property
    def retryable(self) -> bool:
        return CATALOG[self.code].retryable

    @property
    def exit_code(self) -> int:
        return CATALOG[self.code].exit_code

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.code.value}: {self.message!r})"
