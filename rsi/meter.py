"""The factory's scorecard, computed from what the gate wrote and from nothing else.

Every number comes from files under <run>/task/*/authoring/gate/, which only rsi/gate.py
writes. The gate and this file are on the improvement loop's protected list
(rsi/improve.py), so a generation cannot raise its score by changing how the score is taken.

    efficiency        in-band tasks per 100 graded cold attempts. The headline: it goes up
                      when designs land in the band more often or need fewer attempts to
                      get there
    yield             in-band tasks / tasks measured
    attempts_per_hit  graded cold attempts spent per in-band task
    first_shot_error  mean |first measured pass rate - target|: how well designs are aimed
                      before any revision
    prediction_error  mean |pass rate the design predicted - first measured rate|
    unfair_rate       tasks whose fairness review flagged a test / tasks reviewed
    hack_rate         tasks where a red-team attempt passed without the feature / tasks red-teamed
    unproven_rate     tasks nobody solved even with hints / tasks that needed hints

`compare` is the acceptance rule for a live comparison of two versions of the factory.

    python -m rsi.meter RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from . import gate
from .config import CONFIG
from .store import read_jsonl

GUARDRAILS = ("unfair_rate", "hack_rate", "unproven_rate")  # lower is better; a generation must not make these worse


def _read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def task_card(task_dir: Path) -> dict | None:
    """One task's numbers, or None if it was never measured."""
    g = task_dir / "authoring" / "gate"
    log = read_jsonl(g / "difficulty_log.jsonl")
    if not log:
        return None
    # the final state: every measurement of the last version of the task, pooled, judged by the gate's own rule
    last = [m for m in log if m["checksum"] == log[-1]["checksum"]]
    pooled: dict[str, list[int]] = {}
    for m in last:
        for model, v in m["models"].items():
            cell = pooled.setdefault(model, [0, 0])
            cell[0] += v["solved"]
            cell[1] += v["graded"]
    first_solved = sum(v["solved"] for v in log[0]["models"].values())
    first_graded = sum(v["graded"] for v in log[0]["models"].values())
    design = _read(task_dir / "authoring" / "design.json")
    inspect, shortcut, hinted = _read(g / "inspect.json"), _read(g / "shortcut.json"), _read(g / "hinted.json")
    first_rate = first_solved / first_graded if first_graded else None
    return {
        "task": task_dir.name,
        "in_band": bool(pooled) and all(gate.band_verdict(k, n) == "in_band" for k, n in pooled.values()),
        "attempts": sum(v["graded"] for m in log for v in m["models"].values()),
        "measurements": len(log),
        "versions": len({m["checksum"] for m in log}),
        "first_rate": first_rate,
        "predicted": design.get("predicted_rate"),
        "unfair": len(inspect["unfair_tests"]) if "unfair_tests" in inspect else None,
        "hacked": (not shortcut["ok"]) if "ok" in shortcut else None,
        "unproven": (hinted.get("verdict") == "unproven") if hinted.get("verdict") else None,
        "regret": hinted.get("regret"),
    }


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def scorecard(run_dirs: list[Path]) -> dict:
    cards = [c for run in run_dirs for t in sorted((Path(run) / "task").glob("*")) if (c := task_card(t))]
    hits = sum(c["in_band"] for c in cards)
    attempts = sum(c["attempts"] for c in cards)
    rate = lambda key: _mean([float(bool(c[key])) for c in cards if c[key] is not None])  # noqa: E731
    return {
        "tasks": len(cards),
        "in_band": hits,
        "attempts": attempts,
        "efficiency": round(100 * hits / attempts, 2) if attempts else None,
        "yield": round(hits / len(cards), 3) if cards else None,
        "attempts_per_hit": round(attempts / hits, 1) if hits else None,
        "first_shot_error": _mean([abs(c["first_rate"] - CONFIG.target_rate) for c in cards if c["first_rate"] is not None]),
        "prediction_error": _mean([abs(c["predicted"] - c["first_rate"]) for c in cards if c["predicted"] is not None and c["first_rate"] is not None]),
        "unfair_rate": rate("unfair"),
        "hack_rate": rate("hacked"),
        "unproven_rate": rate("unproven"),
        "by_task": cards,
    }


def compare(parent: dict, candidate: dict) -> dict:
    """Accept a candidate version of the factory only if it is more efficient than its parent
    on the same budget of tasks and no guardrail got worse. With a handful of tasks per arm
    this is weak evidence, and the result says how many tasks it rests on."""
    reasons = []
    if not parent["tasks"] or not candidate["tasks"]:
        return {"accept": False, "reasons": ["an arm measured no tasks"], "n": [parent["tasks"], candidate["tasks"]]}
    pe, ce = parent["efficiency"] or 0.0, candidate["efficiency"] or 0.0
    if ce <= pe:
        reasons.append(f"efficiency {ce} is not above the parent's {pe}")
    for g in GUARDRAILS:
        p, c = parent[g], candidate[g]
        if c is not None and c > (p or 0.0):
            reasons.append(f"{g} went from {p} to {c}")
    return {"accept": not reasons, "reasons": reasons or [f"efficiency {pe} -> {ce}, no guardrail worse"],
            "n": [parent["tasks"], candidate["tasks"]], "efficiency": [pe, ce]}


def render(card: dict) -> str:
    keys = ["tasks", "in_band", "attempts", "efficiency", "yield", "attempts_per_hit", "first_shot_error", "prediction_error", *GUARDRAILS]
    return "\n".join(f"{k:<18} {card[k]}" for k in keys)


if __name__ == "__main__":
    print(render(scorecard([Path(a) for a in sys.argv[1:]])))
