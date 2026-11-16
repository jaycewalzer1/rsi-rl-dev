"""Stage 5: the testing suite. Harbor-native port of the Frist hackathon's ws_check.

Stages, in order, stopping at the first hard failure:
  shape       files, task.toml, ticket length, criteria <-> tests <-> mutations mapping
  oracle      reference solution scores 1, three runs, same result each time
  nop         untouched base scores 0
  mutations   every authoring/mutations/*.patch on top of the reference scores 0
  shortcut    an agent told to game the grader without building the feature scores 0
  difficulty  K cold attempts per solver model; pass rate inside the band, at least one solve
  inspect     every failed attempt's failing tests map to a plain ticket sentence (fair)

Each stage writes authoring/gate/<stage>.json stamped with the task checksum, so a
result from before the task last changed shows as STALE in the summary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
import tomllib
from pathlib import Path

from . import harbor, llm
from .config import CONFIG

STAGES = ["shape", "oracle", "nop", "mutations", "shortcut", "difficulty", "inspect"]
MODEL_STAGES = {"shortcut", "difficulty", "inspect"}


def checksum(task_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(task_dir.rglob("*")):
        rel = p.relative_to(task_dir)
        if p.is_file() and rel.parts[:2] != ("authoring", "gate"):
            h.update(str(rel).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _save(task_dir: Path, stage: str, ok: bool, detail: dict) -> dict:
    out = {"stage": stage, "ok": ok, "checksum": checksum(task_dir), **detail}
    path = task_dir / "authoring" / "gate" / f"{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"GATE {stage}: {'PASS' if ok else 'FAIL'} {detail.get('note', '')}")
    return out


def _jobs(task_dir: Path) -> Path:
    return task_dir.parent.parent / "jobs" / task_dir.name


# ---------------------------------------------------------------- shape

def stage_shape(task_dir: Path) -> dict:
    problems = []
    for rel in [
        "task.toml", "instruction.md", "environment/Dockerfile", "tests/test.sh",
        "solution/solve.sh", "solution/reference.patch", "authoring/criteria.json",
    ]:
        if not (task_dir / rel).exists():
            problems.append(f"missing {rel}")
    if problems:
        return _save(task_dir, "shape", False, {"note": "; ".join(problems)})

    tomllib.loads((task_dir / "task.toml").read_text())
    words = len((task_dir / "instruction.md").read_text().split())
    if not 150 <= words <= 300:
        problems.append(f"ticket is {words} words (want 150-250, hard cap 300)")

    criteria = json.loads((task_dir / "authoring" / "criteria.json").read_text())
    mutations = sorted((task_dir / "authoring" / "mutations").glob("*.patch"))
    covered = {m.name.split("-")[1] for m in mutations if "-" in m.name}
    for c in criteria:
        if not c.get("tests"):
            problems.append(f"{c['id']} has no hidden tests")
        if c["id"] not in covered:
            problems.append(f"{c['id']} has no mutation (authoring/mutations/NN-{c['id']}-*.patch)")
    n_tests = sum(len(c.get("tests", [])) for c in criteria)
    if n_tests < 30:
        problems.append(f"only {n_tests} hidden tests mapped to criteria (want 60+)")
    return _save(
        task_dir, "shape", not problems,
        {"note": "; ".join(problems) or f"{words} words, {len(criteria)} criteria, {n_tests} tests, {len(mutations)} mutations"},
    )


# ---------------------------------------------------------------- oracle / nop

def stage_oracle(task_dir: Path, runs: int = 3) -> dict:
    trials = harbor.run_job(task_dir, "oracle", attempts=runs, jobs_dir=_jobs(task_dir), job_name="gate-oracle")
    rewards = [t.reward for t in trials]
    ok = len(trials) == runs and all(t.solved for t in trials)
    return _save(task_dir, "oracle", ok, {"rewards": rewards, "note": f"rewards {rewards}"})


def stage_nop(task_dir: Path) -> dict:
    trials = harbor.run_job(task_dir, "nop", jobs_dir=_jobs(task_dir), job_name="gate-nop")
    ok = all(t.reward == 0 for t in trials)
    return _save(task_dir, "nop", ok, {"rewards": [t.reward for t in trials], "note": f"base reward {trials[0].reward}"})


# ---------------------------------------------------------------- mutations

def mutant_task(task_dir: Path, patch: Path, into: Path) -> Path:
    """Copy of the task whose oracle applies the reference and then the mutation."""
    dst = into / f"{task_dir.name}--{patch.stem}"
    shutil.copytree(task_dir, dst, ignore=shutil.ignore_patterns("gate"))
    shutil.copy(patch, dst / "solution" / "mutation.patch")
    with (dst / "solution" / "solve.sh").open("a") as f:
        f.write("\n# gate: apply one mutation on top of the reference\n")
        f.write("git -C /app apply --whitespace=nowarn /solution/mutation.patch\n")
    return dst


def stage_mutations(task_dir: Path) -> dict:
    patches = sorted((task_dir / "authoring" / "mutations").glob("*.patch"))
    survivors, results = [], {}
    with tempfile.TemporaryDirectory(prefix="rsi-mut-") as tmp:
        for patch in patches:
            mt = mutant_task(task_dir, patch, Path(tmp))
            trial = harbor.run_job(mt, "oracle", jobs_dir=_jobs(task_dir), job_name=f"gate-mut-{patch.stem}")[0]
            caught = trial.reward == 0 and trial.exception is None and bool(harbor.failed_tests(trial))
            results[patch.name] = {"reward": trial.reward, "failed_tests": harbor.failed_tests(trial)}
            if not caught:
                survivors.append(patch.name)
    return _save(
        task_dir, "mutations", not survivors and bool(patches),
        {"results": results, "survivors": survivors,
         "note": f"{len(patches) - len(survivors)}/{len(patches)} caught by assertion"},
    )


# ---------------------------------------------------------------- shortcut

SHORTCUT = """\n\n---\nRED-TEAM MODE: do not implement the feature. Your only goal is to make the
hidden grader award full reward with the least real work: find and exploit any weakness
(test discovery, environment, files the grader reads, hard-coded expectations)."""


def stage_shortcut(task_dir: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="rsi-short-") as tmp:
        dst = Path(tmp) / f"{task_dir.name}--shortcut"
        shutil.copytree(task_dir, dst, ignore=shutil.ignore_patterns("gate"))
        with (dst / "instruction.md").open("a") as f:
            f.write(SHORTCUT)
        model = CONFIG.solver_models[0]
        trial = harbor.run_job(dst, CONFIG.solver_agent, model=model, jobs_dir=_jobs(task_dir), job_name="gate-shortcut")[0]
    return _save(task_dir, "shortcut", not trial.solved, {"model": model, "reward": trial.reward, "note": f"red-team reward {trial.reward} ({model})"})


# ---------------------------------------------------------------- difficulty

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def band_verdict(solved: int, graded: int) -> str:
    """'in_band', 'too_easy', 'too_hard' or 'need_more'."""
    if graded == 0:
        return "need_more"
    rate = solved / graded
    lo, hi = wilson(solved, graded)
    if solved == 0:
        return "too_hard" if graded >= 6 else "need_more"
    if rate > CONFIG.band_high and lo > CONFIG.target_rate:
        return "too_easy"
    if rate < CONFIG.band_low and hi < CONFIG.target_rate:
        return "too_hard"
    if CONFIG.band_low <= rate <= CONFIG.band_high:
        return "in_band"
    return "need_more"


def stage_difficulty(task_dir: Path, attempts: int | None = None, round_no: int = 0) -> dict:
    attempts = attempts or CONFIG.attempts
    per_model = {}
    for model in CONFIG.solver_models:
        tag = model.split("/")[-1]
        trials = harbor.run_job(
            task_dir, CONFIG.solver_agent, model=model, attempts=attempts,
            jobs_dir=_jobs(task_dir), job_name=f"gate-difficulty-r{round_no}-{tag}",
        )
        graded = [t for t in trials if t.reward is not None and t.exception is None]
        solved = sum(t.solved for t in graded)
        per_model[model] = {
            "agent": CONFIG.solver_agent,
            "attempts": len(trials),
            "graded": len(graded),
            "voided": len(trials) - len(graded),
            "solved": solved,
            "rate": round(solved / len(graded), 3) if graded else None,
            "wilson95": [round(x, 3) for x in wilson(solved, len(graded))],
            "verdict": band_verdict(solved, len(graded)),
            "failed_tests": {t.name: harbor.failed_tests(t) for t in graded if not t.solved},
            "job": f"gate-difficulty-r{round_no}-{tag}",
        }
    ok = all(m["verdict"] == "in_band" for m in per_model.values())
    note = "; ".join(f"{m}: {v['solved']}/{v['graded']} {v['verdict']}" for m, v in per_model.items())
    return _save(task_dir, "difficulty", ok, {"models": per_model, "band": [CONFIG.band_low, CONFIG.band_high], "note": note})


# ---------------------------------------------------------------- inspect

INSPECT = """Ticket given to the agent:
<<<
{ticket}
>>>

Hidden test -> criterion map:
{criteria}

A cold attempt failed these hidden tests: {failed}

For each failing test, quote the ONE ticket sentence it enforces, word for word. If no
sentence plainly states the behaviour (it needs a second reading or an unstated
assumption), mark it unfair.

JSON: {{"tests": [{{"test": "...", "sentence": "... or null", "fair": true|false}}]}}"""


def stage_inspect(task_dir: Path) -> dict:
    diff = json.loads((task_dir / "authoring" / "gate" / "difficulty.json").read_text())
    ticket = (task_dir / "instruction.md").read_text()
    criteria = (task_dir / "authoring" / "criteria.json").read_text()
    flat = " ".join(ticket.split())
    unfair, checked = set(), 0
    for model in diff["models"].values():
        for trial, failed in model["failed_tests"].items():
            if not failed:
                continue
            verdict = llm.ask_json(
                INSPECT.format(ticket=ticket, criteria=criteria, failed=failed), CONFIG.judge_model
            )
            checked += 1
            for t in verdict["tests"]:
                if not t.get("fair") or not t.get("sentence") or " ".join(t["sentence"].split()) not in flat:
                    unfair.add(t["test"])
    return _save(
        task_dir, "inspect", not unfair,
        {"judge_model": CONFIG.judge_model, "unfair_tests": sorted(unfair),
         "note": f"{checked} failed attempts read, {len(unfair)} unfair tests"},
    )


# ---------------------------------------------------------------- driver

RUNNERS = {
    "shape": stage_shape, "oracle": stage_oracle, "nop": stage_nop,
    "mutations": stage_mutations, "shortcut": stage_shortcut,
    "difficulty": stage_difficulty, "inspect": stage_inspect,
}


def run(task_dir: Path, stages: list[str] | None = None) -> bool:
    for stage in stages or STAGES:
        if not RUNNERS[stage](task_dir)["ok"]:
            print(f"GATE stopped at {stage}")
            return False
    return True


def summary(task_dir: Path) -> str:
    now = checksum(task_dir)
    lines, all_pass = [], True
    for stage in STAGES:
        path = task_dir / "authoring" / "gate" / f"{stage}.json"
        if not path.exists():
            lines.append(f"{stage:<11} NOT RUN")
            all_pass = False
            continue
        r = json.loads(path.read_text())
        state = "PASS" if r["ok"] else "FAIL"
        if r["checksum"] != now:
            state = "STALE"
        all_pass &= state == "PASS"
        lines.append(f"{stage:<11} {state:<5} {r.get('note', '')}")
    lines.append("GATE: ALL PASS" if all_pass else "GATE: NOT READY")
    return "\n".join(lines)


def stage_list(spec: str | None) -> list[str] | None:
    if not spec:
        return None
    names = re.split(r"[ ,]+", spec.strip())
    bad = [n for n in names if n not in RUNNERS]
    if bad:
        raise SystemExit(f"unknown stages: {bad}; known: {STAGES}")
    return names
