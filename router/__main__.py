"""Run the router: ``python -m router --config config.example.json``."""

from __future__ import annotations

import argparse

from .config import RouterConfig
from .policy import POLICIES
from .server import build_server


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m router", description="OpenAI-compatible router for inference engines")
    ap.add_argument("--config", required=True, help="JSON config file")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--policy", choices=POLICIES, help="override the policy in the config")
    ap.add_argument("-v", "--verbose", action="store_true", help="log every request")
    args = ap.parse_args()

    config = RouterConfig.load(args.config)
    if args.policy:
        config.policy = args.policy
    server = build_server(config, args.host, args.port, verbose=args.verbose)
    server.start_health_checks()
    names = ", ".join(f"{e.name}({e.seats})" for e in config.engines)
    print(f"router on {server.url} policy={config.policy} engines: {names}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
