# agentic-inference-router

An OpenAI-compatible HTTP front for a pool of inference engines, written for
**agentic** traffic: long conversations whose prompts share a growing prefix,
bursts of parallel tool calls, and engines that each serve a fixed number of
concurrent sequences.

Pure Python 3.11+ standard library. No dependencies.

## The problem

Put several inference servers (anything that speaks `/v1/chat/completions`)
behind one address and the obvious load balancer gets three things wrong for
agent workloads:

1. **It ignores the KV cache.** Turn 12 of an agent run re-sends turns 1-11.
   An engine that served turn 11 can skip re-reading (prefilling) most of that
   prompt; any other engine has to start from scratch. Round-robin spreads one
   conversation over every engine and pays prefill again and again, and each
   engine ends up holding a copy of the same prefix.
2. **It ignores seats.** An engine runs well up to N concurrent sequences and
   then thrashes or rejects. Without admission control the overflow becomes
   timeouts at the engine instead of an orderly queue at the front.
3. **It retries the wrong things.** Re-sending a chat completion after the
   engine has started generating doubles the work at exactly the moment the
   fleet is overloaded.

## What it does

| Feature | Behaviour |
|---|---|
| Seats | Each engine has `seats`; the router never sends more concurrent requests. |
| Admission control | When every seat is taken a request waits in one FIFO queue (`queue_limit`). Queue full: `429` + `Retry-After`. Waited `queue_timeout_s`: `503` + `Retry-After`. |
| Least-loaded | Pick the engine with the lowest `inflight / seats`. |
| Prefix affinity | Hash the first `prefix_messages` messages (system prompt + task) into a 16-hex key. First sight: pin the key to the least-loaded engine. Later: go to the pinned engine if it has a seat, else overflow to least-loaded *for that request only* and keep the pin. |
| Health + breaker | A background thread probes `GET /v1/models`. 3 consecutive failures (traffic or probes) open the breaker; after `breaker_cooldown_s` one probe is let through (half-open); success closes it. Opening drops the engine's affinity pins. |
| Safe retries | Retried on another engine only if the client has received nothing **and** the engine did no work: connect/upload failed, or the engine answered `429`/`502`/`503`. A timeout after the request was sent, a `500`, or a broken stream is never retried. |
| Streaming | Responses are relayed as they arrive (SSE works); unknown-length bodies are re-framed as chunked. |
| `/metrics` | Prometheus text: queue depth and limit, seats total/in use, per-engine seats, in-flight, breaker state, p50/p95 latency, outcomes, admission and retry counters, affinity hits/overflows. |

```
                       clients (agents, tools, IDEs)
                                   |
                         POST /v1/chat/completions
                                   v
  +----------------------------------------------------------------+
  | router                                                         |
  |   parse -> prefix key = sha256(messages[:N])[:16]              |
  |      |                                                         |
  |      v                                                         |
  |   Dispatcher ---- free seat? ----------- yes ----+             |
  |      | no                                        |             |
  |      v                                           v             |
  |   FIFO queue (bounded) -- full --> 429     Policy.choose()     |
  |      | seat freed                           | affinity pin     |
  |      +------------------------------------->| least-loaded     |
  |                                             | round-robin      |
  |   HealthChecker --> CircuitBreaker per engine (closed/open/half)|
  +---------------------------------------------|------------------+
                     +--------------------------+------------+
                     v                          v            v
               engine a (4 seats)        engine b (4)   engine c (8)
               KV prefix cache           KV cache       KV cache
```

## Quick start

```bash
# two fake engines (echo the last user message after 0.1 s)
python3 -m router.mock_engine --port 9001 --name a &
python3 -m router.mock_engine --port 9002 --name b &

# the router
python3 -m router --config config.example.json --port 8080 &

curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"system","content":"agent"},{"role":"user","content":"hi"}]}'
curl -sN localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"m","stream":true,"messages":[{"role":"user","content":"stream me"}]}'
curl -s localhost:8080/metrics
```

Point `url` in the config at real servers (vLLM, SGLang, llama.cpp server,
TGI, ... anything OpenAI-compatible) and set `seats` to the concurrency each
one is tuned for. Every response carries `X-Router-Engine` so you can see
where it went.

Config keys (all optional except `engines`): `policy`
(`prefix-affinity` | `least-loaded` | `round-robin`), `prefix_messages`,
`queue_limit`, `queue_timeout_s`, `max_retries`, `connect_timeout_s`,
`read_timeout_s`, `health_interval_s`, `health_path`, `health_timeout_s`,
`breaker_failures`, `breaker_cooldown_s`, `affinity_max_keys`. Per engine:
`name`, `url`, `seats`, `api_key`.

## Tests

```bash
python3 -m unittest -v
```

51 tests: policies, breaker state machine (fake clock), dispatcher
(FIFO, 429, timeouts, soft retry preference), and end-to-end runs against
mock engines on `127.0.0.1` (retry on refused connection and on 503, no retry
on 500, 429 + `Retry-After`, seat cap under a burst, affinity stickiness,
metrics text, health checker). No network beyond loopback.

## Simulation: does prefix affinity pay off?

`sim/simulate.py` is a discrete-event simulator that replays a synthetic
agent workload against **the same policy objects the server uses** (only the
engines and the clock are simulated).

Workload: 300 agent tasks arriving as a Poisson process; 6-14 turns each; each
turn re-sends the whole history plus 200-1500 new tokens (tool output) and
gets 100-400 tokens back; exponential think time (mean 2 s) between turns;
30 % of turns are bursts of 3-6 parallel calls sharing the same prefix.

Engines: 4 engines x 8 seats; prefill 5000 tok/s per engine, one request at a
time, only for the uncached part of the prompt; decode 45 tok/s per seat,
5 % slower per extra busy seat; 150k-token LRU prefix cache per engine.
Each row is the mean of seeds 1-5; every policy replays the same tasks.

**Stated assumption (the whole benefit of affinity rests on it):** a prompt
token that the chosen engine still holds in its prefix cache costs zero
prefill time; every other token costs `1 / prefill_tps` seconds. Real engines
gain less than this (block granularity, cache lookups, partial evictions), so
the affinity gains below are an upper bound for this workload.

```bash
python3 sim/simulate.py            # reproduces the tables below (~1.5 s)
python3 sim/simulate.py --help     # change engines, seats, cache, rates
```

**0.08 tasks/s** (light load)

| policy | req/s | out tok/s | p50 wait s | p95 wait s | p95 task s | prefix cache hit |
|---|---:|---:|---:|---:|---:|---:|
| round-robin | 1.60 | 400 | 0.00 | 0.83 | 162 | 68.8 % |
| least-loaded | 1.60 | 400 | 0.00 | 1.17 | 162 | 70.8 % |
| prefix-affinity | 1.60 | 400 | 0.00 | 0.25 | 142 | 88.1 % |

**0.12 tasks/s** (near saturation)

| policy | req/s | out tok/s | p50 wait s | p95 wait s | p95 task s | prefix cache hit |
|---|---:|---:|---:|---:|---:|---:|
| round-robin | 2.21 | 555 | 7.70 | 20.03 | 392 | 52.4 % |
| least-loaded | 2.23 | 558 | 7.34 | 18.47 | 372 | 54.3 % |
| prefix-affinity | 2.27 | 568 | 6.45 | 13.73 | 312 | 69.9 % |

**0.15 tasks/s** (overloaded: the queue keeps growing while tasks arrive)

| policy | req/s | out tok/s | p50 wait s | p95 wait s | p95 task s | prefix cache hit |
|---|---:|---:|---:|---:|---:|---:|
| round-robin | 2.13 | 533 | 44.29 | 89.66 | 1143 | 39.4 % |
| least-loaded | 2.14 | 535 | 41.68 | 89.95 | 1126 | 40.2 % |
| prefix-affinity | 2.21 | 554 | 32.05 | 79.26 | 1022 | 45.7 % |

How to read it:

* `req/s` and `out tok/s` are completions over the whole run (first arrival
  to last completion). Below saturation every policy keeps up, so throughput
  equals the offered load and the policies differ only in waiting and task
  time. Throughput separates once the fleet is saturated.
* Prefix affinity keeps one copy of each conversation's prefix instead of
  one per engine, so the cache holds more distinct conversations and the
  engines spend less time on prefill: +17-19 points of cache hit at light load,
  p95 queue wait down 26-31 % and p95 task time down 16-20 % near saturation.
* Under overload the advantage shrinks: pinned engines are full, requests
  overflow to whichever engine frees first, and every policy's cache thrashes.
  Affinity is not a substitute for capacity - see the companion
  [llm-scaling-planner](https://github.com/tissalejanet-sys/llm-scaling-planner) for sizing.

What the model leaves out (so treat the numbers as relative, not absolute):
prefill and decode do not interfere with each other, decode speed is fixed at
admission, the cache is a token-count LRU (no block granularity, no sharing of
the system prompt across conversations beyond a warm flag), and the network
and the router itself cost nothing.

## Layout

```
router/
  dispatcher.py   seats, bounded FIFO queue, Retry-After estimate
  policy.py       prefix_key, LeastLoaded, RoundRobin, PrefixAffinity
  breaker.py      closed / open / half-open circuit breaker
  health.py       background /v1/models probes
  server.py       HTTP front, retry rules, streaming relay
  metrics.py      Prometheus text
  mock_engine.py  fake OpenAI-compatible engine for demos and tests
sim/simulate.py   discrete-event comparison of the policies
tests/            unittest suite
```

## Limits and next steps

* One process, threads per connection: fine for tens of engines and hundreds
  of concurrent streams, not for tens of thousands.
* Only `/v1/chat/completions` is routed (plus `/v1/models` passthrough).
* No per-tenant fairness yet: the queue is a single FIFO.
* Possible next step: bounded "affinity wait" (let a request wait a few hundred
  ms for its pinned engine before overflowing), which the simulator can
  evaluate before it goes into the dispatcher.

## License

MIT, see [LICENSE](LICENSE).
