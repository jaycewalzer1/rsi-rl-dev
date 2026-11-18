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

Every finished task appends a row to runs/history.jsonl: per solver model, how many attempts
slipped on each mode and which modes slipped together in one attempt, plus the pass rate the
design predicted and the one its first measurement gave. combine.py fits its stacking model
to those rows, so the next design is aimed with what the last ones measured.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path

from . import archive, build, combine, gate
from .config import CONFIG
from .store import read_json, read_jsonl


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
        cur["failure_messages"] = {**old.get("failure_messages", {}), **cur.get("failure_messages", {})}
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


def slip_patterns(task_dir: Path, model_result: dict) -> dict:
    """Per graded attempt of one solver: which modes it slipped on. {"solved": k, "slips": [[mode, ...], ...]},
    one list per attempt that did not pass (empty when its failing tests map to no mode)."""
    criteria = json.loads((task_dir / "authoring" / "criteria.json").read_text())
    test_mode = {t: c.get("mode", c["id"]) for c in criteria for t in c.get("tests", [])}
    slips = [sorted({test_mode.get(t) for t in failed} - {None}) for failed in model_result["failed_tests"].values()]
    slips += [[] for _ in model_result.get("unreproduced", [])]
    return {"solved": model_result["solved"], "slips": slips}


def _read_gate(task_dir: Path, name: str) -> dict:
    p = task_dir / "authoring" / "gate" / name
    return json.loads(p.read_text()) if p.exists() else {}


def record(task_dir: Path, diff: dict, rounds: int, kind: str = "calibrate") -> dict:
    design = json.loads((task_dir / "authoring" / "design.json").read_text())
    graded = sum(m["graded"] for m in diff["models"].values())
    log = read_jsonl(task_dir / "authoring" / "gate" / "difficulty_log.jsonl")
    first = log[0]["models"] if log else {}
    first_graded = sum(v["graded"] for v in first.values())
    hinted = _read_gate(task_dir, "hinted.json")
    by_model = {}
    for model, m in diff["models"].items():
        patterns = slip_patterns(task_dir, m)
        by_model[model] = {"agent": m["agent"], "attempts": m["graded"], "solved": m["solved"], "patterns": patterns,
                           "mode_slips": dict(Counter(mode for slip in patterns["slips"] for mode in slip))}
    row = {
        "task": task_dir.name,
        "date": time.strftime("%Y-%m-%d"),
        "kind": kind,
        "modes": design.get("modes", []),
        "attempts": graded,
        "solved": sum(m["solved"] for m in diff["models"].values()),
        "models": {k: {"agent": v["agent"], "rate": v["rate"]} for k, v in diff["models"].items()},
        "by_model": by_model,
        "mode_slips": mode_slips(task_dir, diff),
        "predicted": design.get("predicted_rate"),
        "first_rate": round(sum(v["solved"] for v in first.values()) / first_graded, 3) if first_graded else None,
        "regret": hinted.get("regret"),
        "design_score": hinted.get("score"),
        "rounds": rounds,
        "in_band": diff["ok"],
    }
    combine.HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with combine.HISTORY.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


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
            archive.admit(task_dir, diff)
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
    """The mode to add to a task that measured too easy: the one whose handle rate would bring
    the measured pass rate nearest the target (combine.pick_extra)."""
    design = json.loads((task_dir / "authoring" / "design.json").read_text())
    pool_path = modes_path or task_dir.parent.parent / "failure_modes.json"
    pool = [m for m in read_json(pool_path) if m["id"] not in design.get("modes", [])]
    if not pool:
        return "any documented failure mode not yet in this task"
    models = _read_gate(task_dir, "difficulty.json").get("models", {})
    graded = sum(m["graded"] for m in models.values())
    rate = sum(m["solved"] for m in models.values()) / graded if graded else 1.0
    m = combine.pick_extra(pool, rate, random.Random())
    design.setdefault("modes", []).append(m["id"])
    (task_dir / "authoring" / "design.json").write_text(json.dumps(design, indent=2))
    return f"{m['id']}: {m['mode']} (textbook build: {m.get('textbook_build')}; hidden case: {m.get('hidden_case')})"
