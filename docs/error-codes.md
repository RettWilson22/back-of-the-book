# Error codes

Every failure is a `BackOfTheBookError` ([`errors.py`](../src/backofthebook/errors.py)) with a stable `code`, a `message` that's safe to show users, a `retryable` flag, and optional machine-readable `details` (such as the provider's HTTP status). The web app shows the message, plus the code for anything that isn't a simple input mistake. The CLI prints `Error [CODE]: message` and exits with the category's exit code.

| Code | When | Retryable | CLI exit |
|---|---|---|---|
| `EMPTY_TOPIC` | No topic given | no | 2 |
| `TOPIC_TOO_LONG` | Topic over 200 characters | no | 2 |
| `INVALID_QUESTION_COUNT` | Not 1–10 questions | no | 2 |
| `INVALID_DIFFICULTY` | Not easy / medium / hard | no | 2 |
| `QUESTION_TOO_LONG` | Question over 2,000 characters | no | 2 |
| `DUPLICATE_FILE_NAMES` | `ingest` was given two different files with the same name | no | 2 |
| `TOPIC_NOT_COVERED` | Course quiz topic isn't in the loaded materials | no | 3 |
| `SOURCE_NOT_FOUND` | No Wikipedia article matches the Quiz generator topic | no | 3 |
| `SOURCE_UNAVAILABLE` | Wikipedia couldn't be reached | yes | 5 |
| `NO_LLM_CONFIGURED` | No `GROQ_API_KEY` or `ANTHROPIC_API_KEY` | no | 4 |
| `LLM_AUTH_FAILED` | API key rejected | no | 4 |
| `LLM_RATE_LIMITED` | Provider rate limit | yes | 5 |
| `LLM_UNAVAILABLE` | Network error or provider 5xx | yes | 5 |
| `LLM_REQUEST_REJECTED` | Provider 4xx (e.g. invalid model) | no | 5 |
| `LLM_REFUSED` | Model declined the request | no | 5 |
| `LLM_BAD_RESPONSE` | Malformed or truncated response, even after one repair attempt | yes | 5 |
| `USAGE_LIMIT` | The web app's limit on model requests per minute or per visitor was reached | yes | 5 |
| `NO_VALID_QUESTIONS` | Nothing usable survived validation and the source check | yes | 6 |
| `INTERNAL_ERROR` | An unexpected bug (logged with a traceback) | no | 1 |

Invalid input is rejected before any API call. A test checks that every code has a catalog entry.
