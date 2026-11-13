#!/usr/bin/env python3
"""Replay a synthetic agentic workload against the router's policies.

A discrete-event simulation. The policies are the ones in ``router.policy``,
imported unchanged; only the engines and the clock are simulated.

Workload (one *task* = one agent run):
  * tasks arrive as a Poisson process;
  * a task runs 6-14 turns; every turn's prompt is the whole history so far
    (system prompt + task + all earlier tool results and answers), so turns
    of one task share a growing prefix;
  * after a turn the agent "runs tools" for an exponential think time;
  * 30 % of turns are parallel tool-call bursts: 3-6 requests that share the
    same prefix are sent at once, and the next turn waits for all of them.

Engine model (deliberately simple, see the README for what it leaves out):
  * ``seats`` concurrent requests per engine;
  * prefill is compute-bound and shared: an engine prefills one request at a
    time at ``prefill_tps`` tokens/s, and only the *uncached* part of the
    prompt is prefilled;
  * decode runs in parallel on every seat at
    ``decode_tps / (1 + decode_slowdown * (busy_seats - 1))`` tokens/s;
  * each engine keeps an LRU prefix cache of ``kv_tokens`` tokens; a request
    reuses the longest prefix of its own task that the engine still holds
    (plus the common system prompt once the engine has seen it).

Every policy replays the identical task list for each seed.

    python3 sim/simulate.py                 # the README table
    python3 sim/simulate.py --rate 0.12 --seeds 5 --json
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import random
import sys
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.policy import POLICIES, make_policy, prefix_key  # noqa: E402

SYSTEM_PROMPT = "You are a coding agent. Use the tools to finish the task."


@dataclass
class EngineParams:
    engines: int = 4
    seats: int = 8
    prefill_tps: float = 5000.0
    decode_tps: float = 45.0
    decode_slowdown: float = 0.05
    kv_tokens: int = 150_000


@dataclass
class WorkloadParams:
    tasks: int = 300
    rate: float = 0.12  # tasks per second
    sys_tokens: int = 2000
    turns: Tuple[int, int] = (6, 14)
    in_tokens: Tuple[int, int] = (200, 1500)
    out_tokens: Tuple[int, int] = (100, 400)
    think_mean_s: float = 2.0
    burst_prob: float = 0.3
    burst_width: Tuple[int, int] = (3, 6)


@dataclass
class Task:
    tid: int
    arrival: float
    key: str
    turns: List[List[Tuple[int, int]]]  # per turn: [(new_in, out), ...]
    thinks: List[float]
    context: int = 0
    turn: int = 0
    pending: int = 0
    grow: int = 0
    done_at: Optional[float] = None


@dataclass
class Request:
    task: Task
    prompt: int
    new_in: int
    out: int
    submit: float
    burst: bool
    start: float = 0.0
    cached: int = 0


class SimEngine:
    def __init__(self, name: str, p: EngineParams) -> None:
        self.name = name
        self.seats = p.seats
        self.inflight = 0
        self.p = p
        self.prefill_free_at = 0.0
        self.cache: "OrderedDict[str, int]" = OrderedDict()
        self.cache_used = 0
        self.warm = False  # has the shared system prompt

    def cached_for(self, key: str, prompt: int, sys_tokens: int) -> int:
        hit = self.cache.get(key)
        if hit is not None:
            self.cache.move_to_end(key)
            return min(hit, prompt)
        return min(sys_tokens, prompt) if self.warm else 0

    def remember(self, key: str, tokens: int) -> None:
        old = self.cache.pop(key, 0)
        self.cache_used -= old
        tokens = max(tokens, old)
        self.cache[key] = tokens
        self.cache_used += tokens
        while self.cache_used > self.p.kv_tokens and len(self.cache) > 1:
            _, evicted = self.cache.popitem(last=False)
            self.cache_used -= evicted
        self.warm = True


def make_tasks(w: WorkloadParams, seed: int) -> List[Task]:
    rng = random.Random(seed)
    t = 0.0
    tasks = []
    for tid in range(w.tasks):
        t += rng.expovariate(w.rate)
        turns, thinks = [], []
        for _ in range(rng.randint(*w.turns)):
            width = rng.randint(*w.burst_width) if rng.random() < w.burst_prob else 1
            turns.append([(rng.randint(*w.in_tokens), rng.randint(*w.out_tokens)) for _ in range(width)])
            thinks.append(rng.expovariate(1.0 / w.think_mean_s))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"task {seed}-{tid}: fix the failing test"},
        ]
        tasks.append(Task(tid, t, prefix_key(messages, 2), turns, thinks))
    return tasks


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[max(0, math.ceil(q * len(values)) - 1)]


def simulate(policy_name: str, tasks: List[Task], ep: EngineParams, w: WorkloadParams) -> Dict[str, float]:
    engines = [SimEngine(f"e{i}", ep) for i in range(ep.engines)]
    policy = make_policy(policy_name, [e.name for e in engines])
    events: List[Tuple[float, int, str, object]] = []
    seq = 0

    def push(at: float, kind: str, obj: object) -> None:
        nonlocal seq
        seq += 1
        heapq.heappush(events, (at, seq, kind, obj))

    queue: "deque[Request]" = deque()
    done: List[Request] = []
    max_queue = 0

    for task in tasks:
        task.context, task.turn, task.pending, task.grow, task.done_at = w.sys_tokens, 0, 0, 0, None
        push(task.arrival, "turn", task)

    def dispatch(now: float) -> None:
        while queue:
            free = [e for e in engines if e.inflight < e.seats]
            if not free:
                return
            req = queue.popleft()
            engine = policy.choose(free, req.task.key)
            req.start = now
            req.cached = engine.cached_for(req.task.key, req.prompt, w.sys_tokens)
            engine.inflight += 1
            prefill_start = max(now, engine.prefill_free_at)
            prefill_end = prefill_start + (req.prompt - req.cached) / ep.prefill_tps
            engine.prefill_free_at = prefill_end
            rate = ep.decode_tps / (1.0 + ep.decode_slowdown * (engine.inflight - 1))
            engine.remember(req.task.key, req.prompt)  # KV is allocated while the request runs
            push(prefill_end + req.out / rate, "done", (engine, req))

    while events:
        now, _, kind, obj = heapq.heappop(events)
        if kind == "turn":
            task = obj  # type: ignore[assignment]
            calls = task.turns[task.turn]
            task.pending = len(calls)
            task.grow = 0
            for new_in, out in calls:
                queue.append(Request(task, task.context + new_in, new_in, out, now, len(calls) > 1))
            max_queue = max(max_queue, len(queue))
        else:
            engine, req = obj  # type: ignore[misc]
            engine.inflight -= 1
            # Single calls leave prompt+answer reusable; burst siblings only share the old context.
            engine.remember(req.task.key, req.task.context if req.burst else req.prompt + req.out)
            done.append(req)
            task = req.task
            task.grow += req.new_in + req.out
            task.pending -= 1
            if task.pending == 0:
                task.context += task.grow
                think = task.thinks[task.turn]
                task.turn += 1
                if task.turn < len(task.turns):
                    push(now + think, "turn", task)
                else:
                    task.done_at = now
        dispatch(now)

    first = min(t.arrival for t in tasks)
    end = max(t.done_at for t in tasks if t.done_at is not None)
    span = end - first
    waits = [r.start - r.submit for r in done]
    prompt_tokens = sum(r.prompt for r in done)
    return {
        "requests": len(done),
        "req_per_s": len(done) / span,
        "out_tok_per_s": sum(r.out for r in done) / span,
        "wait_p50_s": percentile(waits, 0.5),
        "wait_p95_s": percentile(waits, 0.95),
        "task_p95_s": percentile([t.done_at - t.arrival for t in tasks], 0.95),
        "cache_hit": sum(r.cached for r in done) / prompt_tokens if prompt_tokens else 0.0,
        "max_queue": max_queue,
        "makespan_s": span,
    }


def run(policies, ep: EngineParams, w: WorkloadParams, seeds: List[int]) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for name in policies:
        rows = [simulate(name, make_tasks(w, seed), ep, w) for seed in seeds]
        results[name] = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    return results


def table(results: Dict[str, Dict[str, float]]) -> str:
    head = "| policy | req/s | out tok/s | p50 wait s | p95 wait s | p95 task s | prefix cache hit |"
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    lines = [head, sep]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['req_per_s']:.2f} | {r['out_tok_per_s']:.0f} | {r['wait_p50_s']:.2f} | "
            f"{r['wait_p95_s']:.2f} | {r['task_p95_s']:.0f} | {100 * r['cache_hit']:.1f} % |"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    ep, w = EngineParams(), WorkloadParams()
    ap = argparse.ArgumentParser(description="Compare routing policies on a synthetic agentic workload.")
    ap.add_argument("--rate", type=float, nargs="+", default=[0.08, 0.12, 0.15], help="task arrival rate(s), tasks/s")
    ap.add_argument("--tasks", type=int, default=w.tasks)
    ap.add_argument("--seeds", type=int, default=5, help="average over seeds 1..N")
    ap.add_argument("--engines", type=int, default=ep.engines)
    ap.add_argument("--seats", type=int, default=ep.seats)
    ap.add_argument("--prefill-tps", type=float, default=ep.prefill_tps)
    ap.add_argument("--decode-tps", type=float, default=ep.decode_tps)
    ap.add_argument("--kv-tokens", type=int, default=ep.kv_tokens)
    ap.add_argument("--policies", nargs="+", default=["round-robin", "least-loaded", "prefix-affinity"], choices=POLICIES)
    ap.add_argument("--json", action="store_true", help="print JSON instead of tables")
    args = ap.parse_args(argv)

    ep = EngineParams(args.engines, args.seats, args.prefill_tps, args.decode_tps, ep.decode_slowdown, args.kv_tokens)
    seeds = list(range(1, args.seeds + 1))
    report = {}
    for rate in args.rate:
        w = WorkloadParams(tasks=args.tasks, rate=rate)
        report[str(rate)] = run(args.policies, ep, w, seeds)
    if args.json:
        print(json.dumps({"engine": ep.__dict__, "seeds": seeds, "results": report}, indent=2))
        return
    print(
        f"engines={ep.engines} seats={ep.seats} prefill={ep.prefill_tps:.0f} tok/s decode={ep.decode_tps:.0f} tok/s/seat "
        f"kv={ep.kv_tokens} tokens/engine tasks={args.tasks} seeds={seeds}"
    )
    for rate, results in report.items():
        print(f"\narrival rate {rate} tasks/s")
        print(table(results))


if __name__ == "__main__":
    main()
