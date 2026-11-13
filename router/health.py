"""Background health checks that feed the circuit breakers."""

from __future__ import annotations

import http.client
import threading

from .dispatcher import Dispatcher
from .engine import Engine


def connection_for(engine: Engine, timeout: float) -> http.client.HTTPConnection:
    if engine.scheme == "https":
        return http.client.HTTPSConnection(engine.host, engine.port, timeout=timeout)
    return http.client.HTTPConnection(engine.host, engine.port, timeout=timeout)


def probe(engine: Engine, path: str = "/v1/models", timeout: float = 2.0) -> bool:
    """GET ``path``; any status below 500 counts as alive."""
    conn = connection_for(engine, timeout)
    try:
        headers = {"Authorization": f"Bearer {engine.api_key}"} if engine.api_key else {}
        conn.request("GET", engine.base_path + path, headers=headers)
        resp = conn.getresponse()
        resp.read()
        return resp.status < 500
    except (OSError, http.client.HTTPException):
        return False
    finally:
        conn.close()


class HealthChecker(threading.Thread):
    def __init__(
        self,
        dispatcher: Dispatcher,
        path: str = "/v1/models",
        interval_s: float = 5.0,
        timeout_s: float = 2.0,
    ) -> None:
        super().__init__(name="health-checker", daemon=True)
        self.dispatcher = dispatcher
        self.path = path
        self.interval_s = interval_s
        self.timeout_s = timeout_s
        self._stop_event = threading.Event()

    def check_all(self) -> dict:
        results = {}
        for engine in self.dispatcher.engines:
            ok = probe(engine, self.path, self.timeout_s)
            self.dispatcher.record_health(engine, ok)
            results[engine.name] = ok
        return results

    def run(self) -> None:
        self.check_all()
        while not self._stop_event.wait(self.interval_s):
            self.check_all()

    def stop(self) -> None:
        self._stop_event.set()
