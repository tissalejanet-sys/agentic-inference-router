"""Admission control: seats, a bounded FIFO queue, and engine choice.

``acquire`` returns an engine with a seat reserved for the caller, or raises:

* :class:`QueueFull`    - every seat is taken and ``queue_limit`` requests
  are already waiting (the HTTP layer answers 429 + Retry-After);
* :class:`QueueTimeout` - the caller waited ``timeout`` seconds without a seat;
* :class:`NoEngine`     - every engine's breaker is open.

Waiters are served strictly in arrival order: a newcomer never takes a seat
while someone is queued ahead of it. ``release`` must be called exactly once
per successful ``acquire``.
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter, deque
from typing import Callable, Iterable, List, Optional, Sequence

from .breaker import OPEN
from .engine import Engine
from .policy import Policy


class AdmissionError(Exception):
    def __init__(self, message: str, retry_after: int = 1) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class QueueFull(AdmissionError):
    pass


class QueueTimeout(AdmissionError):
    pass


class NoEngine(AdmissionError):
    pass


class Dispatcher:
    def __init__(
        self,
        engines: Sequence[Engine],
        policy: Policy,
        queue_limit: int = 64,
        clock: Callable[[], float] = time.monotonic,
        wake_interval_s: float = 0.25,
    ) -> None:
        if not engines:
            raise ValueError("at least one engine is required")
        names = [e.name for e in engines]
        if len(set(names)) != len(names):
            raise ValueError("engine names must be unique")
        if queue_limit < 0:
            raise ValueError("queue_limit must be >= 0")
        self.engines: List[Engine] = list(engines)
        self.policy = policy
        self.queue_limit = queue_limit
        self._clock = clock
        self._wake = wake_interval_s
        self._cond = threading.Condition()
        self._waiting: "deque[object]" = deque()
        self.counters: Counter = Counter()

    # -- admission ---------------------------------------------------------

    def acquire(
        self,
        key: Optional[str] = None,
        timeout: float = 30.0,
        avoid: Iterable[Engine] = (),
    ) -> Engine:
        """Reserve a seat. ``avoid`` is a soft preference used by retries."""
        avoid_set = set(id(e) for e in avoid)
        deadline = self._clock() + timeout
        with self._cond:
            self._check_any_engine()
            if not self._waiting:
                engine = self._try_pick(key, avoid_set)
                if engine is not None:
                    return engine
            if len(self._waiting) >= self.queue_limit:
                self.counters["rejected_queue_full"] += 1
                raise QueueFull("all seats busy and the queue is full", self.retry_after())
            ticket = object()
            self._waiting.append(ticket)
            self.counters["queued"] += 1
            try:
                while True:
                    if self._waiting[0] is ticket:
                        engine = self._try_pick(key, avoid_set)
                        if engine is not None:
                            return engine
                    self._check_any_engine()
                    remaining = deadline - self._clock()
                    if remaining <= 0:
                        self.counters["rejected_timeout"] += 1
                        raise QueueTimeout("timed out waiting for a seat", self.retry_after())
                    self._cond.wait(min(remaining, self._wake))
            finally:
                try:
                    self._waiting.remove(ticket)
                except ValueError:
                    pass
                self._cond.notify_all()

    def release(
        self,
        engine: Engine,
        ok: Optional[bool],
        latency_s: Optional[float] = None,
        outcome: str = "ok",
    ) -> None:
        """Return the seat. ``ok`` True/False feeds the breaker; None is neutral."""
        with self._cond:
            if engine.inflight <= 0:
                raise RuntimeError(f"release without acquire on {engine.name}")
            engine.inflight -= 1
            engine.outcomes[outcome] += 1
            if ok is True:
                engine.breaker.record_success()
                if latency_s is not None:
                    engine.latency.add(latency_s)
            elif ok is False:
                if engine.breaker.record_failure():
                    self.policy.forget(engine.name)
            else:
                engine.breaker.record_neutral()
            self._cond.notify_all()

    def record_health(self, engine: Engine, ok: bool) -> None:
        with self._cond:
            if ok:
                engine.breaker.record_success()
            elif engine.breaker.record_failure():
                self.policy.forget(engine.name)
            self._cond.notify_all()

    def count(self, name: str, n: int = 1) -> None:
        with self._cond:
            self.counters[name] += n

    # -- helpers -----------------------------------------------------------

    def _check_any_engine(self) -> None:
        if all(e.breaker.state == OPEN for e in self.engines):
            self.counters["rejected_no_engine"] += 1
            cooldown = min(e.breaker.cooldown_s for e in self.engines)
            raise NoEngine("no healthy engine", max(1, math.ceil(cooldown)))

    def _try_pick(self, key: Optional[str], avoid: set) -> Optional[Engine]:
        free = [e for e in self.engines if e.has_seat() and e.breaker.allows()]
        if not free:
            return None
        preferred = [e for e in free if id(e) not in avoid] or free
        engine = self.policy.choose(preferred, key)
        engine.inflight += 1
        engine.breaker.on_dispatch()
        return engine

    def retry_after(self) -> int:
        """Seconds until a queued request would likely get a seat (1..60)."""
        means = [e.latency.mean() for e in self.engines if len(e.latency)]
        seats = sum(e.seats for e in self.engines) or 1
        if not means:
            return 1
        mean = sum(means) / len(means)
        estimate = mean * (len(self._waiting) + 1) / seats
        return int(min(60, max(1, math.ceil(estimate))))

    def snapshot(self) -> dict:
        with self._cond:
            return {
                "queue_depth": len(self._waiting),
                "queue_limit": self.queue_limit,
                "policy": self.policy.name,
                "policy_stats": dict(self.policy.stats()),
                "counters": dict(self.counters),
                "engines": [
                    {
                        "name": e.name,
                        "seats": e.seats,
                        "inflight": e.inflight,
                        "state": e.breaker.state,
                        "trips": e.breaker.trips,
                        "p50": e.latency.quantile(0.5),
                        "p95": e.latency.quantile(0.95),
                        "samples": len(e.latency),
                        "outcomes": dict(e.outcomes),
                    }
                    for e in self.engines
                ],
            }
