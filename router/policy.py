"""Engine selection policies.

A policy only *chooses*; it never waits. The dispatcher hands it the engines
that have a free seat and a breaker that lets traffic through, so every
policy below may assume ``candidates`` is non-empty and admissible.

Anything with ``name``, ``seats`` and ``inflight`` attributes is an engine as
far as a policy is concerned, which is what lets ``sim/simulate.py`` replay
a workload against exactly the code the HTTP server runs.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any, Iterable, Optional, Protocol, Sequence


class EngineLike(Protocol):
    name: str
    seats: int
    inflight: int


def load(engine: EngineLike) -> float:
    """Fraction of seats in use."""
    return engine.inflight / engine.seats if engine.seats > 0 else float("inf")


def prefix_key(messages: Sequence[Any], n: int = 2, length: int = 16) -> Optional[str]:
    """Short hash of the first ``n`` chat messages.

    In an agent loop the system prompt and the task statement stay fixed
    while tool results are appended, so the first few messages identify the
    conversation and therefore the KV prefix an engine may still hold.
    The hash is over canonical JSON, so key order inside a message does not
    matter but every field (role, content, tool_calls, ...) does.
    """
    if n <= 0 or not messages:
        return None
    head = list(messages[:n])
    canon = json.dumps(head, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:length]


class Policy:
    name = "base"

    def choose(self, candidates: Sequence[EngineLike], key: Optional[str] = None) -> EngineLike:
        raise NotImplementedError

    def forget(self, engine_name: str) -> None:
        """Called when an engine's breaker opens."""

    def stats(self) -> dict:
        return {}


class LeastLoaded(Policy):
    """Lowest seat utilisation wins; ties go to fewer in-flight, then name."""

    name = "least-loaded"

    def choose(self, candidates, key=None):
        return min(candidates, key=lambda e: (load(e), e.inflight, e.name))


class RoundRobin(Policy):
    """Rotate over the configured engine order, skipping engines without a seat."""

    name = "round-robin"

    def __init__(self, engine_names: Iterable[str]) -> None:
        self._order = list(engine_names)
        if not self._order:
            raise ValueError("round-robin needs at least one engine")
        self._cursor = 0

    def choose(self, candidates, key=None):
        by_name = {e.name: e for e in candidates}
        n = len(self._order)
        for i in range(n):
            idx = (self._cursor + i) % n
            engine = by_name.get(self._order[idx])
            if engine is not None:
                self._cursor = idx + 1
                return engine
        # A candidate that is not in the configured order: still serve it.
        return candidates[0]


class PrefixAffinity(Policy):
    """Send requests with the same prefix key to the same engine.

    * first sight of a key: pick with the fallback policy and pin the key there;
    * pinned engine has a seat: use it (a *hit*);
    * pinned engine is busy: use the fallback for this request only, keep the
      pin (an *overflow*), because the pinned engine still holds the prefix;
    * pinned engine's breaker opens: :meth:`forget` drops its pins.

    The table is an LRU bounded by ``max_keys``.
    """

    name = "prefix-affinity"

    def __init__(self, fallback: Optional[Policy] = None, max_keys: int = 10_000) -> None:
        self.fallback = fallback or LeastLoaded()
        self.max_keys = max_keys
        self._table: "OrderedDict[str, str]" = OrderedDict()
        self.hits = 0
        self.overflows = 0
        self.new_keys = 0

    def pinned(self, key: str) -> Optional[str]:
        return self._table.get(key)

    def choose(self, candidates, key=None):
        if key is None:
            return self.fallback.choose(candidates)
        pinned = self._table.get(key)
        if pinned is not None:
            self._table.move_to_end(key)
            for engine in candidates:
                if engine.name == pinned:
                    self.hits += 1
                    return engine
            self.overflows += 1
            return self.fallback.choose(candidates)
        engine = self.fallback.choose(candidates)
        self.new_keys += 1
        self._table[key] = engine.name
        while len(self._table) > self.max_keys:
            self._table.popitem(last=False)
        return engine

    def forget(self, engine_name: str) -> None:
        for key in [k for k, v in self._table.items() if v == engine_name]:
            del self._table[key]

    def stats(self) -> dict:
        return {
            "affinity_hits": self.hits,
            "affinity_overflows": self.overflows,
            "affinity_new_keys": self.new_keys,
            "affinity_keys": len(self._table),
        }


POLICIES = ("prefix-affinity", "least-loaded", "round-robin")


def make_policy(name: str, engine_names: Iterable[str], max_keys: int = 10_000) -> Policy:
    if name == "least-loaded":
        return LeastLoaded()
    if name == "round-robin":
        return RoundRobin(engine_names)
    if name == "prefix-affinity":
        return PrefixAffinity(LeastLoaded(), max_keys=max_keys)
    raise ValueError(f"unknown policy {name!r}; choose from {', '.join(POLICIES)}")
