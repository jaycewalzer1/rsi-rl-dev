"""Stage 6: hold a task at the target pass rate.

    measure (difficulty + inspect)
      unfair failures   -> revise: restate or drop those cases, re-measure
      need_more         -> more attempts on the same checksum, counts pooled
      too_easy          -> revise: add a criterion from a fresh failure mode
      too_hard          -> hinted attempts (told every criterion and trap) decide which:
                             feasible  -> revise: drop the case family most failed attempts miss
                             unproven  -> revise: fix the tests even hinted attempts all fail
      in_band           -> done
    after every revision: shape, audit, oracle, nop and mutations again before re-measuring

Every finished task appends to runs/history.jsonl which modes made solvers slip.
combine.py reads that file, so the next design leans on modes that work. That
feedback is the self-improving part of the loop.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from . import build, gate
from .combine import HISTORY, pick_modes
from .config import CONFIG
from .store import read_json


def _difficulty(task_dir: Path) -> dict:
    return json.loads((task_dir / "authoring" / "gate" / "difficulty.json").read_text())


def _pooled(prev: dict | None, new: dict, task_dir: Path) -> dict:
    """Pool attempt counts with the previous measurement when the task did not change."""
    if not prev or prev["checksum"] != new["checksum"]:
        return new
    for model, cur in new["models"].items():
        old = prev["models"].get(model)
        if not old:
            continue
        cur["graded"] += old["graded"]
        cur["solved"] += old["solved"]
        cur["attempts"] += old["attempts"]
        cur["voided"] += old["voided"]
        cur["failed_tests"] = {**old["failed_tests"], **cur["failed_tests"]}
        cur["unreproduced"] = old.get("unreproduced", []) + cur.get("unreproduced", [])
        cur["rate"] = round(cur["solved"] / cur["graded"], 3) if cur["graded"] else None
        cur["wilson95"] = [round(x, 3) for x in gate.wilson(cur["solved"], cur["graded"])]
        cur["verdict"] = gate.band_verdict(cur["solved"], cur["graded"])
    new["ok"] = all(m["verdict"] == "in_band" for m in new["models"].values())
    (task_dir / "authoring" / "gate" / "difficulty.json").write_text(json.dumps(new, indent=2))
    return new


def mode_slips(task_dir: Path, diff: dict) -> dict[str, int]:
    """Per failure mode: how many graded attempts failed at least one of its tests."""
    criteria = json.loads((task_dir / "authoring" / "criteria.json").read_text())
    test_mode = {t: c.get("mode", c["id"]) for c in criteria for t in c.get("tests", [])}
    slips: Counter = Counter()
    for m in diff["models"].values():
        for failed in m["failed_tests"].values():
            for mode in {test_mode.get(t) for t in failed} - {None}:
                slips[mode] += 1
    return dict(slips)


def hardest_family(task_dir: Path, diff: dict) -> str:
    criteria = json.loads((task_dir / "authoring" / "criteria.json").read_text())
    test_ac = {t: c["id"] for c in criteria for t in c.get("tests", [])}
    hits: Counter = Counter()
    for m in diff["models"].values():
        for failed in m["failed_tests"].values():
            hits.update({test_ac.get(t, "?") for t in failed})
    return hits.most_common(1)[0][0] if hits else "?"


def record(task_dir: Path, diff: dict, rounds: int) -> None:
    design = json.loads((task_dir / "authoring" / "design.json").read_text())
    graded = sum(m["graded"] for m in diff["models"].values())
    row = {
        "task": task_dir.name,
        "modes": design.get("modes", []),
        "attempts": graded,
        "solved": sum(m["solved"] for m in diff["models"].values()),
        "models": {k: {"agent": v["agent"], "rate": v["rate"]} for k, v in diff["models"].items()},
        "mode_slips": mode_slips(task_dir, diff),
        "rounds": rounds,
        "in_band": diff["ok"],
    }
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a") as f:
        f.write(json.dumps(row) + "\n")


def run(task_dir: Path, modes_path: Path | None = None) -> bool:
    prev = None
    for rnd in range(CONFIG.max_calibration_rounds):
        print(f"calibrate: round {rnd + 1}/{CONFIG.max_calibration_rounds} on {task_dir.name}")
        if not gate.run(task_dir, gate.PRE_MEASURE):
            build.revise(task_dir, gate_failures(task_dir), "Fix the failing gate stage(s) without changing the ticket.")
            continue
        gate.stage_difficulty(task_dir, round_no=rnd)
        diff = _pooled(prev, _difficulty(task_dir), task_dir)
        prev = diff
        hinted = gate.stage_hinted(task_dir)  # runs only when nobody solved it cold; else a stamp
        insp = gate.stage_inspect(task_dir)
        if insp["unfair_tests"]:
            build.revise(task_dir, insp, "These failing tests are unfair (no plain ticket sentence states them). Restate the ticket sentence or drop the tests.")
            prev = None
            continue

        verdicts = {m["verdict"] for m in diff["models"].values()}
        if verdicts == {"in_band"}:
            record(task_dir, diff, rnd + 1)
            print(f"calibrate: IN BAND\n{gate.summary(task_dir)}")
            return True
        if "too_easy" in verdicts:
            extra = fresh_mode(task_dir, modes_path)
            build.revise(task_dir, diff, f"Too easy. Add ONE new acceptance criterion built on this failure mode, in a different part of the code, with its own hidden cases and mutation: {extra}")
            prev = None
        elif "too_hard" in verdicts and hinted.get("verdict") == "unproven":
            build.revise(task_dir, hinted, "Nobody solved this, not even attempts that were told every criterion and where builds slip, "
                         f"so the task is unproven, not hard. These tests failed in every hinted attempt: {hinted['always_failed'][:40]}. "
                         "For each, either the ticket does not determine the behaviour (restate its sentence so it does) or the case is "
                         "wrong (fix or drop it). Do not delete cases just to reach a pass rate.")
            prev = None
        elif "too_hard" in verdicts:
            fam = hardest_family(task_dir, diff)
            proof = f" Hinted attempts solved it {hinted['solved']}/{hinted['graded']}, so it is possible and the ticket is enough." if hinted.get("solved") else ""
            build.revise(task_dir, diff, f"Too hard (no or almost no full solves).{proof} Remove the hidden cases of criterion {fam} that most failed attempts miss, keep its ticket sentence, keep one case so the mutation is still caught.")
            prev = None
        # need_more: loop again with the same checksum, counts pool
    diff = _difficulty(task_dir)
    record(task_dir, diff, CONFIG.max_calibration_rounds)
    print(f"calibrate: gave up after {CONFIG.max_calibration_rounds} rounds\n{gate.summary(task_dir)}")
    return False


def gate_failures(task_dir: Path) -> dict:
    out = {}
    for p in (task_dir / "authoring" / "gate").glob("*.json"):
        r = json.loads(p.read_text())
        if not r["ok"]:
            out[r["stage"]] = r
    return out


def fresh_mode(task_dir: Path, modes_path: Path | None) -> str:
    design = json.loads((task_dir / "authoring" / "design.json").read_text())
    pool_path = modes_path or task_dir.parent.parent / "failure_modes.json"
    pool = [m for m in read_json(pool_path) if m["id"] not in design.get("modes", [])]
    if not pool:
        return "any documented failure mode not yet in this task"
    m = pick_modes(pool, 1, random.Random())[0]
    design.setdefault("modes", []).append(m["id"])
    (task_dir / "authoring" / "design.json").write_text(json.dumps(design, indent=2))
    return f"{m['id']}: {m['mode']} (textbook build: {m.get('textbook_build')}; hidden case: {m.get('hidden_case')})"
