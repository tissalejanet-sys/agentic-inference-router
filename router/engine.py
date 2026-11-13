"""Per-engine state the router keeps."""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlsplit

from .breaker import CircuitBreaker


class LatencyWindow:
    """The last ``size`` latencies, for p50/p95 without unbounded memory."""

    def __init__(self, size: int = 512) -> None:
        self._values: "deque[float]" = deque(maxlen=size)

    def add(self, seconds: float) -> None:
        self._values.append(seconds)

    def __len__(self) -> int:
        return len(self._values)

    def quantile(self, q: float) -> Optional[float]:
        """Nearest-rank quantile, ``None`` when empty."""
        if not self._values:
            return None
        values = sorted(self._values)
        idx = max(0, math.ceil(q * len(values)) - 1)
        return values[min(idx, len(values) - 1)]

    def mean(self) -> Optional[float]:
        if not self._values:
            return None
        return sum(self._values) / len(self._values)


@dataclass(eq=False)
class Engine:
    name: str
    url: str
    seats: int
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    api_key: Optional[str] = None
    inflight: int = 0
    latency: LatencyWindow = field(default_factory=LatencyWindow)
    outcomes: Counter = field(default_factory=Counter)

    def __post_init__(self) -> None:
        if self.seats < 1:
            raise ValueError(f"engine {self.name}: seats must be >= 1")
        parts = urlsplit(self.url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"engine {self.name}: url must be http(s)://host[:port][/prefix]")
        self.scheme = parts.scheme
        self.host = parts.hostname
        self.port = parts.port or (443 if parts.scheme == "https" else 80)
        self.base_path = parts.path.rstrip("/")

    def has_seat(self) -> bool:
        return self.inflight < self.seats
