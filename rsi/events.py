"""The swarm's event log: every agent says when it starts, what it is doing, and how it ended.

One JSON line per event in runs/<run>/events.jsonl. The demo page (`rsi demo`) tails it,
so a live run and a replay look the same. Stages that run outside a swarm call emit()
with no log open, which does nothing.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")

_lock = threading.Lock()
_path: Path | None = None
_t0 = time.time()


def open_log(run_dir: Path, fresh: bool = False) -> Path:
    global _path, _t0
    _path = run_dir / "events.jsonl"
    if fresh and _path.exists():
        _path.unlink()
    _t0 = time.time()
    return _path


def emit(kind: str, *, stage: str, agent: str | None = None, role: str | None = None, msg: str = "", **data) -> None:
    """kind: stage_start | stage_end | spawn | work | done | fail | metric | artifact."""
    if _path is None:
        return
    row = {"t": round(time.time(), 2), "dt": round(time.time() - _t0, 2), "kind": kind, "stage": stage,
           "agent": agent, "role": role, "msg": msg, **data}
    with _lock, _path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def swarm(stage: str, role: str, items: Iterable[T], fn: Callable[[T, str], R], *,
          workers: int = 6, label: Callable[[T], str] = str) -> list[R | None]:
    """Run fn(item, agent_id) for every item on a pool of agents. A failed agent returns None."""
    items = list(items)

    def one(pair: tuple[int, T]) -> R | None:
        i, item = pair
        agent = f"{role}-{i + 1:02d}"
        emit("spawn", stage=stage, agent=agent, role=role, msg=label(item))
        try:
            out = fn(item, agent)
        except Exception as e:  # one agent failing must not sink the swarm
            emit("fail", stage=stage, agent=agent, role=role, msg=f"{type(e).__name__}: {e}"[:300])
            return None
        emit("done", stage=stage, agent=agent, role=role, msg=_brief(out))
        return out

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(one, enumerate(items)))


def _brief(out) -> str:
    if isinstance(out, list):
        return f"{len(out)} items"
    if isinstance(out, dict):
        return out.get("_brief") or ", ".join(list(out)[:4])
    return str(out)[:120] if out is not None else ""
