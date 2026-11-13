"""A consecutive-failure circuit breaker.

States:

* ``closed``    - traffic flows; consecutive failures are counted.
* ``open``      - the engine is skipped until ``cooldown_s`` has passed.
* ``half_open`` - one probe (a request or a health check) may go through;
  success closes the breaker, failure opens it again.

The breaker is not thread-safe on its own: the dispatcher calls it under its
lock. The clock is injectable so tests do not sleep.
"""

from __future__ import annotations

import time
from typing import Callable

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if cooldown_s < 0:
            raise ValueError("cooldown_s must be >= 0")
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._state = CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._probe_out = False
        self.trips = 0

    @property
    def state(self) -> str:
        if self._state == OPEN and self._clock() - self._opened_at >= self.cooldown_s:
            self._state = HALF_OPEN
            self._probe_out = False
        return self._state

    def allows(self) -> bool:
        """True if a new request may be sent to this engine now."""
        state = self.state
        if state == CLOSED:
            return True
        if state == HALF_OPEN:
            return not self._probe_out
        return False

    def on_dispatch(self) -> None:
        """Mark that a request was sent; in half-open it becomes the probe."""
        if self.state == HALF_OPEN:
            self._probe_out = True

    def record_success(self) -> None:
        state = self.state
        self._failures = 0
        if state == HALF_OPEN:
            self._state = CLOSED
            self._probe_out = False

    def record_failure(self) -> bool:
        """Count a failure. Returns True when this call opened the breaker."""
        state = self.state
        if state == HALF_OPEN:
            self._open()
            return True
        if state == OPEN:
            return False
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._open()
            return True
        return False

    def record_neutral(self) -> None:
        """A request ended without telling us anything (e.g. client hung up)."""
        if self._state == HALF_OPEN:
            self._probe_out = False

    def _open(self) -> None:
        self._state = OPEN
        self._opened_at = self._clock()
        self._failures = 0
        self._probe_out = False
        self.trips += 1
