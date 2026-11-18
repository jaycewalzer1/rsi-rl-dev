"""The pool of calibrated tasks: what is in it, whether a new design is new, and when a task
has stopped teaching the current solver.

    runs/archive.jsonl   one row per change of state; the last row per task is its state

    novelty    a new design is compared with every task in the pool and with the worked
               examples: cosine similarity of word 1- and 2-grams, half on the ticket, half
               on the criteria and codebase. At or above RSI_NOVELTY_MAX it is a near copy and
               is rejected before anything is built
    admit      a task that calibrated into the band joins the pool
    recheck    a pass rate is a fact about one solver. `rsi recheck` measures the active tasks
               again with today's RSI_SOLVER_MODELS and writes the result to history, so the
               design policy learns what the new solver handles
    retire     a task that re-measures too easy has stopped teaching. It leaves the active
               pool and is shown to the author as a saturated design, not to be rebuilt
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path

from .config import CONFIG
from .store import read_jsonl

ARCHIVE = Path("runs/archive.jsonl")


# ---------------------------------------------------------------- novelty

def vector(text: str) -> Counter:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return Counter(words) + Counter(zip(words, words[1:]))


def cosine(a: Counter, b: Counter) -> float:
    dot = sum(v * b[k] for k, v in a.items() if k in b)
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


def design_text(d: dict) -> tuple[str, str]:
    """(what the solver reads, what the task is built from) for a design or an archive row."""
    ticket = d.get("ticket") or " ".join([d.get("summary", ""), *d.get("ticket_rules", [])])
    body = " ".join([d.get("body", ""), d.get("codebase", ""), d.get("summary", "")]
                    + [str(c.get("text", "")) + " " + " ".join(c.get("surfaces", [])) for c in d.get("criteria", []) if isinstance(c, dict)])
    return ticket, body


def similarity(a: dict, b: dict) -> float:
    (ta, ba), (tb, bb) = design_text(a), design_text(b)
    return 0.5 * cosine(vector(ta), vector(tb)) + 0.5 * cosine(vector(ba), vector(bb))


def nearest(design: dict, others: list[dict] | None = None) -> tuple[float, str | None]:
    """The most similar task among the pool (active and retired) and `others` (worked examples)."""
    best = (0.0, None)
    for o in pool() + (others or []):
        name = o.get("task") or o.get("slug")
        if name != design.get("slug"):
            best = max(best, (similarity(design, o), name), key=lambda x: x[0])
    return best


# ---------------------------------------------------------------- the pool

def pool(status: str | None = None) -> list[dict]:
    last = {}
    for row in read_jsonl(ARCHIVE):
        last[row["task"]] = row
    return [r for r in last.values() if status is None or r["status"] == status]


def _append(row: dict) -> dict:
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with ARCHIVE.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def _rates(diff: dict) -> dict:
    return {model: [m["solved"], m["graded"]] for model, m in diff["models"].items()}


def admit(task_dir: Path, diff: dict) -> dict:
    """A task that calibrated into the band joins the active pool."""
    design = json.loads((task_dir / "authoring" / "design.json").read_text())
    ticket, body = design_text({**design, "ticket": (task_dir / "instruction.md").read_text()})
    return _append({"task": task_dir.name, "status": "active", "date": time.strftime("%Y-%m-%d"), "path": str(task_dir),
                    "modes": design.get("modes", []), "summary": design.get("summary", ""), "ticket": ticket, "body": body,
                    "measured": _rates(diff)})


def recheck(names: list[str] | None = None, attempts: int | None = None) -> list[dict]:
    """Measure active tasks again with today's solvers. Too easy now: retired. Every result goes
    to history as a `recheck` row, which is how the design policy learns about a new solver."""
    from . import calibrate, gate  # these import this module

    out = []
    for row in pool("active"):
        task_dir = Path(row["path"])
        if names and row["task"] not in names:
            continue
        if not (task_dir / "task.toml").exists():
            print(f"recheck: {row['task']} is no longer on disk at {task_dir}, skipped")
            continue
        if not gate.run(task_dir, gate.PRE_MEASURE):
            print(f"recheck: {row['task']} no longer passes the gate, skipped")
            continue
        diff = gate.stage_difficulty(task_dir, attempts=attempts, round_no=900)
        calibrate.record(task_dir, diff, rounds=0, kind="recheck")
        verdicts = {m["verdict"] for m in diff["models"].values()}
        state = {**row, "date": time.strftime("%Y-%m-%d"), "measured": _rates(diff)}
        if "too_easy" in verdicts:
            state.update(status="retired", why="re-measured too easy: " + diff["note"])
        out.append(_append(state))
        print(f"recheck: {row['task']} {state['status']} ({diff['note']})")
    return out


def saturated_examples() -> str:
    """Retired tasks as lines for the author's prompt: designs today's solvers have outgrown."""
    lines = []
    for r in pool("retired"):
        rates = ", ".join(f"{m.split('/')[-1]} {k}/{n}" for m, (k, n) in r.get("measured", {}).items())
        lines.append(f"- RETIRED {r['task']} [{', '.join(r.get('modes', []))}]: {r.get('summary', '')} Now solved {rates}. Outgrown: do not rebuild it.")
    return "\n".join(lines)


def listing() -> str:
    rows = pool()
    if not rows:
        return f"pool: empty ({ARCHIVE} has no rows yet)"
    return "\n".join(f"{r['status']:<8} {r['date']}  {r['task']}  [{', '.join(r.get('modes', []))}]  "
                     + ", ".join(f"{m.split('/')[-1]} {k}/{n}" for m, (k, n) in r.get("measured", {}).items()) for r in rows)
