"""agentic-inference-router: an OpenAI-compatible front for several engines."""

from .breaker import CircuitBreaker
from .dispatcher import Dispatcher, NoEngine, QueueFull, QueueTimeout
from .engine import Engine
from .policy import LeastLoaded, PrefixAffinity, RoundRobin, make_policy, prefix_key

__all__ = [
    "CircuitBreaker",
    "Dispatcher",
    "Engine",
    "LeastLoaded",
    "NoEngine",
    "PrefixAffinity",
    "QueueFull",
    "QueueTimeout",
    "RoundRobin",
    "make_policy",
    "prefix_key",
]

__version__ = "0.1.0"
