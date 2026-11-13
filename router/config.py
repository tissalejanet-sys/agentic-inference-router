"""Router configuration (JSON file or dict)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from typing import List, Optional


@dataclass
class EngineConfig:
    name: str
    url: str
    seats: int = 4
    api_key: Optional[str] = None


@dataclass
class RouterConfig:
    engines: List[EngineConfig] = field(default_factory=list)
    policy: str = "prefix-affinity"
    prefix_messages: int = 2
    queue_limit: int = 64
    queue_timeout_s: float = 30.0
    max_retries: int = 2
    connect_timeout_s: float = 2.0
    read_timeout_s: float = 600.0
    health_interval_s: float = 5.0
    health_path: str = "/v1/models"
    health_timeout_s: float = 2.0
    breaker_failures: int = 3
    breaker_cooldown_s: float = 10.0
    affinity_max_keys: int = 10_000

    @classmethod
    def from_dict(cls, data: dict) -> "RouterConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {', '.join(sorted(unknown))}")
        values = dict(data)
        engines = [EngineConfig(**e) for e in values.pop("engines", [])]
        if not engines:
            raise ValueError("config needs at least one engine")
        return cls(engines=engines, **values)

    @classmethod
    def load(cls, path: str) -> "RouterConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
