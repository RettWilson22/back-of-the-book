"""Limits that the web app's visitors share. They are safe to use from many threads."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable


class CallLimiter:
    """Allows at most `limit` calls in any `window` seconds for each key (such as a visitor),
    counted across all threads. Keys with no calls in the last window are forgotten."""

    def __init__(
        self, limit: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limit = limit
        self.window = window
        self._clock = clock
        self._calls: dict[str, deque[float]] = {}
        self._swept = clock()
        self._lock = threading.Lock()

    def try_acquire(self, key: str = "") -> bool:
        """Count one call for `key` and return True, or return False if its limit is reached."""
        with self._lock:
            now = self._clock()
            if now - self._swept >= self.window:
                self._calls = {
                    k: calls
                    for k, calls in self._calls.items()
                    if calls and now - calls[-1] < self.window
                }
                self._swept = now
            calls = self._calls.setdefault(key, deque())
            while calls and now - calls[0] >= self.window:
                calls.popleft()
            if len(calls) >= self.limit:
                return False
            calls.append(now)
            return True


class Quota:
    """A fixed amount, such as passages held in memory, that callers take from and give back."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.used = 0
        self._lock = threading.Lock()

    def try_take(self, amount: int) -> bool:
        """Take `amount` and return True, or return False if that would go over the total."""
        with self._lock:
            if self.used + amount > self.total:
                return False
            self.used += amount
            return True

    def give_back(self, amount: int) -> None:
        with self._lock:
            self.used -= amount
