"""Local HTTP server for the Stonkfly Orchard demonstration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Callable

from .brain import MaleCNSOrchardBrain
from .core import OrchardSimulation, ToyOrchardBrain

STATIC = Path(__file__).with_name("static")
ROOT = Path(__file__).resolve().parents[2]


class OrchardRuntime:
    def __init__(
        self,
        simulation: OrchardSimulation,
        *,
        log_path: Path | None = None,
    ) -> None:
        self.simulation = simulation
        self.log_path = log_path
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self._logged_events = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self._flush_events()

    def start(self) -> None:
        if not self.thread.is_alive():
            self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def _loop(self) -> None:
        dt = self.simulation.config.world_dt_seconds
        deadline = time.monotonic()
        while not self.stop_event.is_set():
            with self.lock:
                running = self.simulation.running
                if running:
                    self.simulation.advance()
                    self._flush_events()
            if running:
                deadline += dt
                self.stop_event.wait(max(0.0, deadline - time.monotonic()))
            else:
                deadline = time.monotonic()
                self.stop_event.wait(0.05)

    def _flush_events(self) -> None:
        if self.log_path is None:
            self._logged_events = len(self.simulation.events)
            return
        pending = self.simulation.events[self._logged_events :]
        if not pending:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            for event in pending:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        self._logged_events += len(pending)

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return self.simulation.snapshot()

    def command(self, action: str) -> dict[str, object]:
        with self.lock:
            result = self.simulation.command(action)
            self._flush_events()
            return result

    def events_after(self, sequence: int) -> list[dict[str, object]]:
        with self.lock:
            return [
                event
                for event in self.simulation.events
                if int(event["sequence"]) > sequence
            ][-80:]


def handler_factory(runtime: OrchardRuntime) -> type[BaseHTTPRequestHandler]:
    class OrchardHandler(BaseHTTPRequestHandler):
        def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _static(self, name: str, content_type: str) -> None:
            path = STATIC / name
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path, _, query = self.path.partition("?")
            if path == "/":
                self._static("index.html", "text/html; charset=utf-8")
            elif path == "/app.css":
                self._static("app.css", "text/css; charset=utf-8")
            elif path == "/app.js":
                self._static("app.js", "text/javascript; charset=utf-8")
            elif path == "/api/state":
                self._json(runtime.snapshot())
            elif path == "/api/events":
                try:
                    after = int(query.removeprefix("after=")) if query else -1
                except ValueError:
                    self._json({"error": "invalid after sequence"}, HTTPStatus.BAD_REQUEST)
                    return
                self._json({"events": runtime.events_after(after)})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/control":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 2 or length > 1024:
                    raise ValueError("invalid body size")
                payload = json.loads(self.rfile.read(length))
                self._json(runtime.command(str(payload["action"])))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: object) -> None:
            return

    return OrchardHandler


def make_runtime(
    *, seed: int, toy: bool, log_root: Path | None = None
) -> OrchardRuntime:
    if toy:
        factory: Callable[[], object] = ToyOrchardBrain
    else:
        brain = MaleCNSOrchardBrain(seed=seed)
        factory = lambda: brain
    simulation = OrchardSimulation(factory, seed=seed)
    log_path = None
    if log_root is not None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        session = log_root / f"orchard-{stamp}-seed-{seed}"
        session.mkdir(parents=True, exist_ok=False)
        (session / "session.json").write_text(
            json.dumps(
                {
                    "schema": "stonkfly-orchard-session-v1",
                    "seed": seed,
                    "brain": "toy" if toy else "MaleCNS-E1G-M2",
                    "market_or_trading_used": False,
                },
                indent=2,
            )
            + "\n"
        )
        log_path = session / "events.jsonl"
    return OrchardRuntime(simulation, log_path=log_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Use the tiny deterministic fixture instead of local MaleCNS data",
    )
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.port < 65536 or args.seed < 0:
        parser.error("port and seed are outside the supported range")
    runtime = make_runtime(
        seed=args.seed,
        toy=args.toy,
        log_root=None if args.no_log else ROOT / "runs/orchard",
    )
    server = ThreadingHTTPServer((args.host, args.port), handler_factory(runtime))
    runtime.start()
    print(
        f"Stonkfly Orchard ready at http://{args.host}:{args.port} "
        f"(seed={args.seed}, brain={'toy' if args.toy else 'MaleCNS E1-G + M2'})",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
