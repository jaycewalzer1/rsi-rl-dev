"""The demo page: the swarm, the anneal and the Qwen before/after, from runs/<run>/events.jsonl.

    rsi demo --run NAME [--port 8791]          live: the page polls the log every 2 s
    rsi demo --run NAME --export page.html     one self-contained file with the log baked in (replay)
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .store import read_jsonl

PAGE = Path(__file__).resolve().parent.parent / "demo" / "index.html"


def state(run_dir: Path) -> dict:
    return {"run": run_dir.name, "events": read_jsonl(run_dir / "events.jsonl")}


def export(run_dir: Path, out: Path) -> Path:
    html = PAGE.read_text().replace("window.__STATE__ = null;", f"window.__STATE__ = {json.dumps(state(run_dir))};")
    out.write_text(html)
    return out


def serve(run_dir: Path, port: int) -> None:
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/api/state"):
                body, ctype = json.dumps(state(run_dir)).encode(), "application/json"
            else:
                body, ctype = PAGE.read_bytes(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    print(f"demo: http://127.0.0.1:{port}  (run {run_dir.name})")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
