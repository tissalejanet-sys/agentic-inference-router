"""The OpenAI-compatible HTTP front.

Endpoints:

* ``POST /v1/chat/completions`` - routed to an engine, streamed back as it
  arrives (SSE included);
* ``GET  /v1/models``           - answered by the first healthy engine;
* ``GET  /metrics``             - Prometheus text;
* ``GET  /healthz``             - 200 while at least one engine is usable.

Retry rule: a request is retried on another engine only when the client has
not received a byte *and* the engine provably did no work - the TCP connect
or the request upload failed, or the engine answered 429/502/503 before any
body. A timeout or disconnect after the request was sent is not retried:
the engine may already be generating, and a second copy would double the
load at the worst possible moment.
"""

from __future__ import annotations

import http.client
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple

from .breaker import CircuitBreaker, OPEN
from .config import RouterConfig
from .dispatcher import Dispatcher, NoEngine, QueueFull, QueueTimeout
from .engine import Engine
from .health import HealthChecker, connection_for
from .metrics import render
from .policy import make_policy, prefix_key

RETRY_STATUSES = frozenset({429, 502, 503})
CHAT_PATH = "/v1/chat/completions"
MAX_BODY = 64 * 1024 * 1024


class RouterServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: Tuple[str, int], config: RouterConfig, dispatcher: Dispatcher, verbose: bool = False):
        self.config = config
        self.dispatcher = dispatcher
        self.verbose = verbose
        self.health: Optional[HealthChecker] = None
        super().__init__(address, RouterHandler)

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"

    def start_health_checks(self) -> HealthChecker:
        self.health = HealthChecker(
            self.dispatcher,
            path=self.config.health_path,
            interval_s=self.config.health_interval_s,
            timeout_s=self.config.health_timeout_s,
        )
        self.health.start()
        return self.health

    def server_close(self) -> None:
        if self.health is not None:
            self.health.stop()
        super().server_close()


def build_server(config: RouterConfig, host: str = "127.0.0.1", port: int = 8080, verbose: bool = False) -> RouterServer:
    engines = [
        Engine(
            name=e.name,
            url=e.url,
            seats=e.seats,
            api_key=e.api_key,
            breaker=CircuitBreaker(config.breaker_failures, config.breaker_cooldown_s),
        )
        for e in config.engines
    ]
    policy = make_policy(config.policy, [e.name for e in engines], config.affinity_max_keys)
    dispatcher = Dispatcher(engines, policy, queue_limit=config.queue_limit)
    return RouterServer((host, port), config, dispatcher, verbose=verbose)


class _UpstreamBroke(Exception):
    """The engine failed while we were relaying its body."""


class RouterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "agentic-inference-router/0.1"
    server: RouterServer

    def log_message(self, fmt, *args):  # noqa: D401 - quiet unless verbose
        if self.server.verbose:
            super().log_message(fmt, *args)

    # -- small response helpers -------------------------------------------

    def _send(self, status: int, body: bytes, ctype: str, extra: Optional[dict] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, str(v))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj: dict, extra: Optional[dict] = None) -> None:
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json", extra)

    def _error(self, status: int, message: str, retry_after: Optional[int] = None) -> None:
        extra = {"Retry-After": retry_after} if retry_after is not None else None
        self._json(status, {"error": {"message": message, "type": "router_error", "code": status}}, extra)

    # -- GET ----------------------------------------------------------------

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/metrics":
            body = render(self.server.dispatcher.snapshot()).encode("utf-8")
            self._send(200, body, "text/plain; version=0.0.4; charset=utf-8")
        elif path == "/healthz":
            snap = self.server.dispatcher.snapshot()
            up = [e["name"] for e in snap["engines"] if e["state"] != OPEN]
            self._json(200 if up else 503, {"engines_up": up, "queue_depth": snap["queue_depth"]})
        elif path == "/v1/models":
            self._models()
        else:
            self._error(404, f"no route for GET {path}")

    def _models(self) -> None:
        cfg = self.server.config
        for engine in self.server.dispatcher.engines:
            if engine.breaker.state == OPEN:
                continue
            conn = connection_for(engine, cfg.health_timeout_s)
            try:
                conn.request("GET", engine.base_path + "/v1/models", headers=self._upstream_headers(engine))
                resp = conn.getresponse()
                body = resp.read()
                if resp.status == 200:
                    self._send(200, body, resp.getheader("Content-Type", "application/json"))
                    return
            except (OSError, http.client.HTTPException):
                continue
            finally:
                conn.close()
        self._error(503, "no engine answered /v1/models")

    # -- POST ---------------------------------------------------------------

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path != CHAT_PATH:
            self._error(404, f"no route for POST {path}")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY:
            self._error(400, "missing or invalid Content-Length")
            return
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw or b"null")
        except ValueError:
            self._error(400, "body is not valid JSON")
            return
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            self._error(400, "body must be a JSON object with a 'messages' list")
            return
        key = prefix_key(body["messages"], self.server.config.prefix_messages)
        self._route(raw, key)

    def _upstream_headers(self, engine: Engine) -> dict:
        headers = {"Content-Type": "application/json", "Accept": self.headers.get("Accept", "*/*")}
        if engine.api_key:
            headers["Authorization"] = f"Bearer {engine.api_key}"
        elif self.headers.get("Authorization"):
            headers["Authorization"] = self.headers["Authorization"]
        return headers

    def _route(self, raw: bytes, key: Optional[str]) -> None:
        cfg = self.server.config
        dispatcher = self.server.dispatcher
        tried = []
        for attempt in range(cfg.max_retries + 1):
            try:
                engine = dispatcher.acquire(key, timeout=cfg.queue_timeout_s, avoid=tried)
            except QueueFull as exc:
                self._error(429, str(exc), retry_after=exc.retry_after)
                return
            except (QueueTimeout, NoEngine) as exc:
                self._error(503, str(exc), retry_after=exc.retry_after)
                return
            can_retry = attempt < cfg.max_retries
            if not self._try_engine(engine, raw, can_retry):
                return
            tried.append(engine)
            dispatcher.count("retries")
        self._error(502, "every attempt failed before the engine accepted the request")

    def _try_engine(self, engine: Engine, raw: bytes, can_retry: bool) -> bool:
        """Serve the request from ``engine``. Returns True if the caller should retry."""
        cfg = self.server.config
        dispatcher = self.server.dispatcher
        started = time.monotonic()
        conn = connection_for(engine, cfg.connect_timeout_s)
        # Phase 1: connect and upload. Failing here means the engine did no work.
        try:
            conn.connect()
            conn.sock.settimeout(cfg.read_timeout_s)
            conn.request("POST", engine.base_path + CHAT_PATH, body=raw, headers=self._upstream_headers(engine))
        except OSError:
            conn.close()
            dispatcher.release(engine, ok=False, outcome="connect_error")
            if can_retry:
                return True
            self._error(502, f"engine {engine.name} is unreachable")
            return False
        # Phase 2: wait for the status line. The engine may be working: no retry.
        try:
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException):
            conn.close()
            dispatcher.release(engine, ok=False, outcome="no_response")
            self._error(502, f"engine {engine.name} failed before responding")
            return False
        if resp.status in RETRY_STATUSES and can_retry:
            try:
                resp.read()
            except (OSError, http.client.HTTPException):
                pass
            conn.close()
            # 429 means "busy", not "broken": do not count it against the breaker.
            dispatcher.release(engine, ok=None if resp.status == 429 else False, outcome=f"http_{resp.status}")
            return True
        # Phase 3: relay. From the first byte on, there is no way back.
        ok: Optional[bool] = resp.status < 500
        outcome = "ok" if resp.status < 400 else f"http_{resp.status}"
        try:
            self._relay(engine, resp)
        except _UpstreamBroke:
            ok, outcome = False, "stream_broken"
            self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            ok, outcome = None, "client_gone"
            self.close_connection = True
        finally:
            conn.close()
            dispatcher.release(engine, ok=ok, latency_s=time.monotonic() - started, outcome=outcome)
        return False

    def _relay(self, engine: Engine, resp: http.client.HTTPResponse) -> None:
        ctype = resp.getheader("Content-Type", "application/json")
        length = resp.getheader("Content-Length")
        chunked = length is None and self.request_version == "HTTP/1.1"
        self.send_response(resp.status)
        self.send_header("Content-Type", ctype)
        self.send_header("X-Router-Engine", engine.name)
        if length is not None:
            self.send_header("Content-Length", length)
        elif chunked:
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.close_connection = True
        if "text/event-stream" in ctype:
            self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.flush()
        while True:
            try:
                data = resp.read1(65536)
            except (OSError, http.client.HTTPException) as exc:
                raise _UpstreamBroke() from exc
            if not data:
                break
            if chunked:
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            else:
                self.wfile.write(data)
            self.wfile.flush()
        if chunked:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
