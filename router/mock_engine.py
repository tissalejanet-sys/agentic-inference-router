"""A tiny OpenAI-compatible engine for demos and tests.

It does not run a model: it echoes the last user message after ``delay``
seconds, optionally as an SSE stream. ``status`` forces an error status so
retries and the breaker can be exercised.

    python -m router.mock_engine --port 9001 --name a --delay 0.2
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # close-delimited bodies, like many engines' SSE
    server: "MockEngine"

    def log_message(self, fmt, *args):
        pass

    def _json(self, status: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/v1/models"):
            if self.server.health_status != 200:
                self._json(self.server.health_status, {"error": "unhealthy"})
                return
            self._json(200, {"object": "list", "data": [{"id": self.server.model, "object": "model"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(length) or b"{}")
        with self.server.lock:
            self.server.calls += 1
            self.server.active += 1
            self.server.peak_active = max(self.server.peak_active, self.server.active)
        try:
            if self.server.status != 200:
                self._json(self.server.status, {"error": {"message": "forced failure"}})
                return
            time.sleep(self.server.delay)
            last = next((m.get("content") for m in reversed(req.get("messages", [])) if m.get("role") == "user"), "")
            text = f"[{self.server.name}] {last}"
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for word in text.split(" "):
                    chunk = {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": word + " "}}]}
                    self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                return
            self._json(
                200,
                {
                    "id": f"mock-{self.server.calls}",
                    "object": "chat.completion",
                    "model": self.server.model,
                    "engine": self.server.name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": len(text.split()), "total_tokens": len(text.split())},
                },
            )
        finally:
            with self.server.lock:
                self.server.active -= 1


class MockEngine(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, name: str = "mock", host: str = "127.0.0.1", port: int = 0, delay: float = 0.0,
                 status: int = 200, model: str = "mock-model") -> None:
        self.name = name
        self.delay = delay
        self.status = status
        self.health_status = 200
        self.model = model
        self.calls = 0
        self.active = 0
        self.peak_active = 0
        self.lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        super().__init__((host, port), _Handler)

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "MockEngine":
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.shutdown()
        self.server_close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9001)
    ap.add_argument("--name", default="mock")
    ap.add_argument("--delay", type=float, default=0.1)
    ap.add_argument("--status", type=int, default=200)
    args = ap.parse_args()
    engine = MockEngine(args.name, args.host, args.port, args.delay, args.status)
    print(f"mock engine {args.name} on {engine.url}", flush=True)
    try:
        engine.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
