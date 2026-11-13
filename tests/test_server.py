"""End-to-end tests: the router in front of mock engines, all on 127.0.0.1."""

import http.client
import json
import socket
import threading
import time
import unittest

from router.config import RouterConfig
from router.mock_engine import MockEngine
from router.server import build_server


def dead_url() -> str:
    """A loopback port with nothing listening (connection refused)."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


def chat(messages, stream=False):
    return {"model": "any", "messages": messages, "stream": stream}


SYS = {"role": "system", "content": "You are an agent."}


class RouterHarness(unittest.TestCase):
    engines = ()

    def start(self, engine_specs, **overrides):
        """engine_specs: list of (name, url, seats)."""
        cfg = {
            "engines": [{"name": n, "url": u, "seats": s} for n, u, s in engine_specs],
            "queue_timeout_s": 5,
            "connect_timeout_s": 1,
            "read_timeout_s": 10,
        }
        cfg.update(overrides)
        self.router = build_server(RouterConfig.from_dict(cfg), "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.router.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop_router)
        return self.router

    def _stop_router(self):
        self.router.shutdown()
        self.router.server_close()

    def mock(self, name, **kw):
        m = MockEngine(name, **kw).start()
        self.addCleanup(m.stop)
        return m

    def request(self, method, path, body=None):
        host, port = self.router.server_address[:2]
        conn = http.client.HTTPConnection(host, port, timeout=10)
        try:
            data = json.dumps(body).encode() if body is not None else None
            headers = {"Content-Type": "application/json"} if data else {}
            conn.request(method, path, body=data, headers=headers)
            resp = conn.getresponse()
            return resp.status, dict(resp.getheaders()), resp.read()
        finally:
            conn.close()


class BasicRoutingTest(RouterHarness):
    def test_completion_is_proxied(self):
        a = self.mock("a")
        self.start([("a", a.url, 2)])
        status, headers, body = self.request("POST", "/v1/chat/completions", chat([SYS, {"role": "user", "content": "hello"}]))
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-Router-Engine"], "a")
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "[a] hello")

    def test_stream_is_relayed_chunked(self):
        a = self.mock("a")
        self.start([("a", a.url, 2)])
        status, headers, body = self.request("POST", "/v1/chat/completions", chat([SYS, {"role": "user", "content": "one two three"}], stream=True))
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers["Content-Type"])
        self.assertEqual(headers.get("Transfer-Encoding"), "chunked")
        text = body.decode()
        self.assertTrue(text.rstrip().endswith("data: [DONE]"))
        self.assertEqual(text.count("data: "), 5)  # "[a]", "one", "two", "three", [DONE]

    def test_bad_requests(self):
        a = self.mock("a")
        self.start([("a", a.url, 1)])
        self.assertEqual(self.request("POST", "/v1/chat/completions", {"model": "x"})[0], 400)
        self.assertEqual(self.request("POST", "/v1/other", chat([]))[0], 404)
        self.assertEqual(self.request("GET", "/nope")[0], 404)
        self.assertEqual(a.calls, 0)

    def test_models_and_healthz(self):
        a = self.mock("a", model="m1")
        self.start([("a", a.url, 1)])
        status, _, body = self.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["data"][0]["id"], "m1")
        status, _, body = self.request("GET", "/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["engines_up"], ["a"])


class RetryTest(RouterHarness):
    def test_connection_refused_is_retried_elsewhere(self):
        b = self.mock("b")
        self.start([("dead", dead_url(), 2), ("b", b.url, 2)], policy="round-robin")
        status, headers, _ = self.request("POST", "/v1/chat/completions", chat([SYS, {"role": "user", "content": "x"}]))
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-Router-Engine"], "b")
        snap = self.router.dispatcher.snapshot()
        self.assertEqual(snap["counters"]["retries"], 1)
        dead = next(e for e in snap["engines"] if e["name"] == "dead")
        self.assertEqual(dead["outcomes"], {"connect_error": 1})

    def test_503_before_body_is_retried(self):
        a = self.mock("a", status=503)
        b = self.mock("b")
        self.start([("a", a.url, 2), ("b", b.url, 2)], policy="round-robin")
        status, headers, _ = self.request("POST", "/v1/chat/completions", chat([SYS, {"role": "user", "content": "x"}]))
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-Router-Engine"], "b")
        self.assertEqual((a.calls, b.calls), (1, 1))

    def test_500_is_not_retried(self):
        a = self.mock("a", status=500)
        b = self.mock("b")
        self.start([("a", a.url, 2), ("b", b.url, 2)], policy="round-robin")
        status, _, _ = self.request("POST", "/v1/chat/completions", chat([SYS, {"role": "user", "content": "x"}]))
        self.assertEqual(status, 500)
        self.assertEqual((a.calls, b.calls), (1, 0))

    def test_last_attempt_passes_the_engine_status_through(self):
        a = self.mock("a", status=503)
        self.start([("a", a.url, 2)], max_retries=1)
        status, _, _ = self.request("POST", "/v1/chat/completions", chat([SYS, {"role": "user", "content": "x"}]))
        self.assertEqual(status, 503)
        self.assertEqual(a.calls, 2)

    def test_breaker_opens_on_a_dead_engine(self):
        b = self.mock("b")
        self.start([("dead", dead_url(), 4), ("b", b.url, 4)], policy="round-robin", breaker_failures=2, breaker_cooldown_s=60)
        for _ in range(4):
            self.assertEqual(self.request("POST", "/v1/chat/completions", chat([SYS, {"role": "user", "content": "x"}]))[0], 200)
        _, _, body = self.request("GET", "/metrics")
        self.assertIn('router_engine_up{engine="dead"} 0', body.decode())
        self.assertIn('router_engine_up{engine="b"} 1', body.decode())


class AdmissionTest(RouterHarness):
    def test_429_with_retry_after_when_queue_is_full(self):
        a = self.mock("a", delay=0.5)
        self.start([("a", a.url, 1)], queue_limit=0)
        first = {}
        t = threading.Thread(target=lambda: first.update(r=self.request("POST", "/v1/chat/completions", chat([SYS, {"role": "user", "content": "1"}]))))
        t.start()
        deadline = time.monotonic() + 5
        while self.router.dispatcher.engines[0].inflight == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        status, headers, body = self.request("POST", "/v1/chat/completions", chat([SYS, {"role": "user", "content": "2"}]))
        t.join(5)
        self.assertEqual(status, 429)
        self.assertGreaterEqual(int(headers["Retry-After"]), 1)
        self.assertEqual(json.loads(body)["error"]["code"], 429)
        self.assertEqual(first["r"][0], 200)

    def test_seat_limit_is_respected_under_burst(self):
        a = self.mock("a", delay=0.1)
        self.start([("a", a.url, 2)], queue_limit=16)
        results = []
        lock = threading.Lock()

        def go(i):
            r = self.request("POST", "/v1/chat/completions", chat([SYS, {"role": "user", "content": str(i)}]))
            with lock:
                results.append(r[0])

        threads = [threading.Thread(target=go, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        self.assertEqual(results, [200] * 8)
        self.assertLessEqual(a.peak_active, 2)


class AffinityTest(RouterHarness):
    def test_turns_of_one_conversation_stay_on_one_engine(self):
        a = self.mock("a", delay=0.3)
        b = self.mock("b", delay=0.3)
        self.start([("a", a.url, 4), ("b", b.url, 4)], policy="prefix-affinity")
        task_x = [SYS, {"role": "user", "content": "task x"}]
        task_y = [SYS, {"role": "user", "content": "task y"}]
        out = {}
        t = threading.Thread(target=lambda: out.update(x=self.request("POST", "/v1/chat/completions", chat(task_x))))
        t.start()
        while self.router.dispatcher.engines[0].inflight == 0:
            time.sleep(0.005)
        # While x occupies a, y is pinned to the less loaded b ...
        y1 = self.request("POST", "/v1/chat/completions", chat(task_y))[1]["X-Router-Engine"]
        t.join(5)
        # ... and y's next turn goes back to b even though both are idle now.
        y2 = self.request("POST", "/v1/chat/completions", chat(task_y + [{"role": "assistant", "content": "ok"}, {"role": "tool", "content": "result"}]))[1]["X-Router-Engine"]
        self.assertEqual(out["x"][1]["X-Router-Engine"], "a")
        self.assertEqual((y1, y2), ("b", "b"))
        _, _, metrics = self.request("GET", "/metrics")
        self.assertIn('router_policy_stat{name="affinity_hits"} 1', metrics.decode())


class MetricsTest(RouterHarness):
    def test_metrics_text(self):
        a = self.mock("a")
        self.start([("a", a.url, 3)])
        self.request("POST", "/v1/chat/completions", chat([SYS, {"role": "user", "content": "x"}]))
        status, headers, body = self.request("GET", "/metrics")
        text = body.decode()
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("text/plain"))
        for needle in (
            "router_queue_depth 0",
            "router_seats_total 3",
            "router_seats_in_use 0",
            'router_engine_seats{engine="a"} 3',
            'router_engine_latency_seconds{engine="a",quantile="0.95"}',
            'router_engine_requests_total{engine="a",outcome="ok"} 1',
        ):
            self.assertIn(needle, text)


class HealthCheckTest(RouterHarness):
    def test_health_checker_trips_and_recovers(self):
        a = self.mock("a")
        self.start([("a", a.url, 1)], breaker_failures=1, breaker_cooldown_s=0)
        from router.health import HealthChecker

        hc = HealthChecker(self.router.dispatcher, interval_s=60)
        a.health_status = 500
        self.assertEqual(hc.check_all(), {"a": False})
        self.assertEqual(self.router.dispatcher.engines[0].breaker.trips, 1)
        a.health_status = 200
        self.assertEqual(hc.check_all(), {"a": True})
        self.assertEqual(self.router.dispatcher.engines[0].breaker.state, "closed")


if __name__ == "__main__":
    unittest.main()
