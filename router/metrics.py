"""Prometheus text exposition of a dispatcher snapshot."""

from __future__ import annotations

from typing import List

_STATE_VALUE = {"closed": 1, "half_open": 0.5, "open": 0}


def _label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render(snapshot: dict) -> str:
    out: List[str] = []

    def metric(name: str, kind: str, help_text: str) -> None:
        out.append(f"# HELP {name} {help_text}")
        out.append(f"# TYPE {name} {kind}")

    metric("router_queue_depth", "gauge", "Requests waiting for a seat.")
    out.append(f"router_queue_depth {snapshot['queue_depth']}")
    metric("router_queue_limit", "gauge", "Maximum queued requests before 429.")
    out.append(f"router_queue_limit {snapshot['queue_limit']}")

    engines = snapshot["engines"]
    total_seats = sum(e["seats"] for e in engines)
    total_used = sum(e["inflight"] for e in engines)
    metric("router_seats_total", "gauge", "Seats across all engines.")
    out.append(f"router_seats_total {total_seats}")
    metric("router_seats_in_use", "gauge", "Seats currently serving a request.")
    out.append(f"router_seats_in_use {total_used}")

    metric("router_engine_seats", "gauge", "Configured seats per engine.")
    for e in engines:
        out.append(f'router_engine_seats{{engine="{_label(e["name"])}"}} {e["seats"]}')
    metric("router_engine_inflight", "gauge", "Requests in flight per engine.")
    for e in engines:
        out.append(f'router_engine_inflight{{engine="{_label(e["name"])}"}} {e["inflight"]}')
    metric("router_engine_up", "gauge", "Breaker state: 1 closed, 0.5 half-open, 0 open.")
    for e in engines:
        out.append(f'router_engine_up{{engine="{_label(e["name"])}"}} {_STATE_VALUE[e["state"]]}')
    metric("router_engine_breaker_trips_total", "counter", "Times the breaker opened.")
    for e in engines:
        out.append(f'router_engine_breaker_trips_total{{engine="{_label(e["name"])}"}} {e["trips"]}')
    metric("router_engine_latency_seconds", "summary", "Request latency over the recent window.")
    for e in engines:
        name = _label(e["name"])
        for q, key in (("0.5", "p50"), ("0.95", "p95")):
            if e[key] is not None:
                out.append(
                    f'router_engine_latency_seconds{{engine="{name}",quantile="{q}"}} {e[key]:.6f}'
                )
        out.append(f'router_engine_latency_seconds_count{{engine="{name}"}} {e["samples"]}')
    metric("router_engine_requests_total", "counter", "Finished requests by outcome.")
    for e in engines:
        for outcome, n in sorted(e["outcomes"].items()):
            out.append(
                f'router_engine_requests_total{{engine="{_label(e["name"])}",outcome="{_label(outcome)}"}} {n}'
            )

    counters = snapshot["counters"]
    metric("router_events_total", "counter", "Admission and retry events.")
    for event in ("queued", "rejected_queue_full", "rejected_timeout", "rejected_no_engine", "retries"):
        out.append(f'router_events_total{{event="{event}"}} {counters.get(event, 0)}')

    stats = snapshot.get("policy_stats") or {}
    if stats:
        metric("router_policy_stat", "gauge", f"Policy counters ({snapshot['policy']}).")
        for k, v in sorted(stats.items()):
            out.append(f'router_policy_stat{{name="{_label(k)}"}} {v}')
    return "\n".join(out) + "\n"
