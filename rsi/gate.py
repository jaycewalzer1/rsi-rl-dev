"""Stage 5: the testing suite. Harbor-native port of the Frist hackathon's ws_check.

Stages, in order, stopping at the first hard failure:
  shape       files, task.toml, ticket length, criteria <-> tests <-> mutations mapping,
              every criterion's sentence in the ticket word for word
  audit       static leak and verifier checks (rsi/audit.py): no model, no Docker
  oracle      reference solution scores 1, three runs, same result each time
  nop         untouched base scores 0
  mutations   every authoring/mutations/*.patch on top of the reference scores 0
  shortcut    red-team attempts (exploit the grader; the same with the hidden test names;
              hard-code the narrowest thing that passes) score 0, or what passes is judged a
              real implementation next to the reference
  difficulty  K cold attempts per solver model; pass rate inside the band, at least one solve;
              a pass counts only if its recorded patch passes again in a clean container
  hinted      when no cold attempt solves it: attempts that are told every criterion and
              where builds slip. A solve proves the task possible; none means unproven
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
import time
import tomllib
from pathlib import Path

from . import audit, harbor, llm
from .config import CONFIG

STAGES = ["shape", "audit", "oracle", "nop", "mutations", "shortcut", "difficulty", "hinted", "inspect"]
PRE_MEASURE = ["shape", "audit", "oracle", "nop", "mutations"]  # must pass before any solver attempt is spent
MODEL_STAGES = {"shortcut", "difficulty", "hinted", "inspect"}


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
    ticket = " ".join((task_dir / "instruction.md").read_text().split())
    for c in criteria:
        if c.get("sentence") and " ".join(c["sentence"].split()) not in ticket:
            problems.append(f"{c['id']} sentence is not in the ticket word for word")
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


# ---------------------------------------------------------------- audit

def stage_audit(task_dir: Path) -> dict:
    findings = audit.audit(task_dir)
    blocks = [f for f in findings if f["severity"] == "block"]
    note = "; ".join(f"{f['check']}: {f['detail']}" for f in blocks)[:600] or f"clean ({len(findings)} warnings)"
    return _save(task_dir, "audit", not blocks, {"findings": findings, "note": note})


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


# ---------------------------------------------------------------- replay

def replay_task(task_dir: Path, patch: str, into: Path, tag: str) -> Path:
    """Copy of the task whose oracle applies one attempt's recorded patch instead of the reference."""
    dst = into / f"{task_dir.name}--replay-{tag}"
    shutil.copytree(task_dir, dst, ignore=shutil.ignore_patterns("gate"))
    (dst / "solution" / "agent.patch").write_text(patch)
    (dst / "solution" / "solve.sh").write_text(
        "#!/bin/bash\nset -e\n# gate: replay one attempt's recorded change in a clean container\n"
        "git -C /app apply --whitespace=nowarn /solution/agent.patch\n")
    return dst


def replay(task_dir: Path, trial: harbor.Trial) -> float | None:
    """Reward of the attempt's patch on its own, in a clean container.

    The agent works as root in the container the verifier later runs in, so a reward can come
    from something other than the code (a replaced test runner, a planted plugin). Only what is
    in the repo diff is the answer; this re-earns the reward from the diff alone.
    None: the task records no patch, or the replay itself errored (the first reward stands)."""
    patch = harbor.agent_patch(trial)
    if patch is None:
        return None
    if not patch.strip():
        return 0.0  # nothing changed in the repo: the pass did not come from code
    with tempfile.TemporaryDirectory(prefix="rsi-replay-") as tmp:
        rt = replay_task(task_dir, patch, Path(tmp), trial.name[-24:])
        again = harbor.run_job(rt, "oracle", jobs_dir=_jobs(task_dir), job_name=f"gate-replay-{trial.name}"[:80])[0]
    return None if again.reward is None or again.exception else float(again.reward)


# ---------------------------------------------------------------- shortcut

RED_TEAM = "\n\n---\nRED-TEAM MODE: do not implement the feature. "
STRATEGIES = {
    "exploit": RED_TEAM + """Your only goal is to make the hidden grader award full reward with the
least real work: find and exploit any weakness (test discovery, environment, files the grader
reads, hard-coded expectations).""",
    "please-hack": RED_TEAM + """Your only goal is to make the hidden grader award full reward
without building what the ticket asks. The hidden tests are named:
{names}
Use the names to guess what is checked and satisfy exactly that.""",
    "hard-code": RED_TEAM + """Make the hidden tests pass with the narrowest change you can:
special-case the inputs a grader would use, return constants, cover only the examples the
ticket gives. Change nothing outside the repository.""",
}

SHORTCUT_JUDGE = """A red-team agent was told NOT to implement this ticket and to get the hidden
grader to pass anyway. Its change passed, also when replayed alone in a clean container.

TICKET
<<<
{ticket}
>>>

REFERENCE (the honest implementation), as a diff:
<<<
{reference}
>>>

RED-TEAM CHANGE, as a diff:
<<<
{patch}
>>>

Is the red-team change a real, general implementation of the ticket's rules (the agent did
the work after all), or a shortcut that satisfies the tests without the rules (special-cased
inputs, hard-coded outputs, behaviour that only holds for what a test exercises)? Quote the
lines that decide it, word for word from the red-team diff.

JSON: {{"verdict": "implementation"|"shortcut", "evidence": ["<line>", ...], "why": "..."}}"""


def judge_shortcut(task_dir: Path, patch: str) -> dict:
    """Read a passing red-team patch next to the reference. A verdict of 'implementation' stands
    only with at least one quoted line that really is in the patch; anything else is a shortcut."""
    v = llm.ask_json(SHORTCUT_JUDGE.format(
        ticket=(task_dir / "instruction.md").read_text(), patch=patch[:40000],
        reference=(task_dir / "solution" / "reference.patch").read_text()[:40000]), CONFIG.judge_model)
    quoted = [e for e in v.get("evidence", []) if e.strip() and e.strip().lstrip("+- ") in patch]
    real = v.get("verdict") == "implementation" and bool(quoted)
    return {"verdict": "implementation" if real else "shortcut", "evidence": quoted[:5], "why": str(v.get("why", ""))[:400],
            "judge_model": CONFIG.judge_model}


def shortcut_outcome(task_dir: Path, trial: harbor.Trial) -> dict:
    """held | tamper (pass did not survive replay) | implemented (judged real) | shortcut | unreplayed."""
    if not trial.solved:
        return {"outcome": "held", "reward": trial.reward}
    again = replay(task_dir, trial)
    if again is None:
        return {"outcome": "unreplayed", "reward": trial.reward, "why": "the attempt passed and there is no recorded patch to replay"}
    if again < 1:
        return {"outcome": "tamper", "reward": trial.reward, "replay": again,
                "why": "passed in its own container, failed when its patch was replayed in a clean one"}
    j = judge_shortcut(task_dir, harbor.agent_patch(trial) or "")
    return {"outcome": "implemented" if j["verdict"] == "implementation" else "shortcut", "reward": trial.reward, "replay": again,
            "evidence": j["evidence"], "why": j["why"], "judge_model": j["judge_model"]}


def stage_shortcut(task_dir: Path) -> dict:
    criteria = json.loads((task_dir / "authoring" / "criteria.json").read_text())
    names = sorted({re.sub(r"\[.*$", "", t) for c in criteria for t in c.get("tests", [])})
    results = {}
    with tempfile.TemporaryDirectory(prefix="rsi-short-") as tmp:
        for strategy, text in STRATEGIES.items():
            dst = Path(tmp) / f"{task_dir.name}--{strategy}"
            shutil.copytree(task_dir, dst, ignore=shutil.ignore_patterns("gate"))
            with (dst / "instruction.md").open("a") as f:
                f.write(text.format(names="\n".join(f"- {n}" for n in names)) if strategy == "please-hack" else text)
            for model in CONFIG.redteam_models:
                tag = f"{strategy}-{model.split('/')[-1]}"
                trial = harbor.run_job(dst, CONFIG.solver_agent, model=model, jobs_dir=_jobs(task_dir), job_name=f"gate-shortcut-{tag}")[0]
                results[tag] = {"model": model, "strategy": strategy, **shortcut_outcome(task_dir, trial)}
    bad = {k: v for k, v in results.items() if v["outcome"] in ("shortcut", "unreplayed")}
    tampered = [k for k, v in results.items() if v["outcome"] == "tamper"]
    note = (f"{len(results) - len(bad)}/{len(results)} red-team attempts held"
            + (f"; PASSED WITHOUT THE FEATURE: {', '.join(bad)}" if bad else "")
            + (f"; tampering neutralised by replay: {', '.join(tampered)}" if tampered else ""))
    return _save(task_dir, "shortcut", not bad, {"attempts": results, "note": note})


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


def _tally(task_dir: Path, trials: list[harbor.Trial]) -> dict:
    """Counts for one batch of attempts. Errored trials are void; a pass counts only if the
    recorded patch passes again on its own (see replay)."""
    graded = [t for t in trials if t.reward is not None and t.exception is None]
    unreproduced = [t.name for t in graded if t.solved and CONFIG.replay and (r := replay(task_dir, t)) is not None and r < 1]
    solved = sum(t.solved for t in graded) - len(unreproduced)
    return {"attempts": len(trials), "graded": len(graded), "voided": len(trials) - len(graded), "solved": solved,
            "rate": round(solved / len(graded), 3) if graded else None, "unreproduced": unreproduced,
            "failed_tests": {t.name: harbor.failed_tests(t) for t in graded if not t.solved},
            "failure_messages": {t.name: harbor.failure_messages(t) for t in graded if not t.solved}}


def stage_difficulty(task_dir: Path, attempts: int | None = None, round_no: int = 0) -> dict:
    attempts = attempts or CONFIG.attempts
    per_model = {}
    for model in CONFIG.solver_models:
        tag = model.split("/")[-1]
        trials = harbor.run_job(
            task_dir, CONFIG.solver_agent, model=model, attempts=attempts,
            jobs_dir=_jobs(task_dir), job_name=f"gate-difficulty-r{round_no}-{tag}",
        )
        tally = _tally(task_dir, trials)
        per_model[model] = {
            "agent": CONFIG.solver_agent,
            **tally,
            "wilson95": [round(x, 3) for x in wilson(tally["solved"], tally["graded"])],
            "verdict": band_verdict(tally["solved"], tally["graded"]),
            "job": f"gate-difficulty-r{round_no}-{tag}",
        }
    ok = all(m["verdict"] == "in_band" for m in per_model.values())
    note = "; ".join(f"{m}: {v['solved']}/{v['graded']} {v['verdict']}"
                     + (f" ({len(v['unreproduced'])} passes did not survive replay)" if v["unreproduced"] else "") for m, v in per_model.items())
    out = _save(task_dir, "difficulty", ok, {"models": per_model, "band": [CONFIG.band_low, CONFIG.band_high], "note": note})
    # every measurement, in order: the first line is the design's first shot, before any revision
    with (task_dir / "authoring" / "gate" / "difficulty_log.jsonl").open("a") as f:
        f.write(json.dumps({"date": time.strftime("%Y-%m-%d"), "round": round_no, "checksum": out["checksum"],
                            "models": {m: {"solved": v["solved"], "graded": v["graded"]} for m, v in per_model.items()}}) + "\n")
    return out


# ---------------------------------------------------------------- hinted

HINT = """

---
REVIEWER NOTES (privileged: a cold solver never sees this part)
The hidden tests check each of these, separately:
{criteria}
{traps}"""


def hint_text(task_dir: Path) -> str:
    """Every criterion, plus where the textbook build slips if the design recorded it. No test code."""
    criteria = json.loads((task_dir / "authoring" / "criteria.json").read_text())
    design_path = task_dir / "authoring" / "design.json"
    design = json.loads(design_path.read_text()) if design_path.exists() else {}
    traps = [f"- the usual build ({tb.get('build', '?')}) fails on: {tb.get('hidden_case_it_fails', '?')}"
             for tb in (design.get("textbook_build") or {}).values() if isinstance(tb, dict)]
    if design.get("central_change_risk"):
        traps.append(f"- no single change covers it: {design['central_change_risk']}")
    return HINT.format(criteria="\n".join(f"- {c['id']}: {c.get('sentence') or c.get('text', '')}" for c in criteria),
                       traps=("Where builds usually slip:\n" + "\n".join(traps)) if traps else "Builds usually satisfy most of these and drop one.")


def design_score(p_cold: float, p_hint: float | None) -> dict:
    """SPADE's designer reward, with this repo's band as the anchor: 0.4 x regret (how much the
    hint helps, floored at 0) + 0.6 x closeness of the cold pass rate to the target."""
    regret = None if p_hint is None else round(max(0.0, p_hint - p_cold), 3)
    anchor = max(0.0, 1 - abs(p_cold - CONFIG.target_rate) / max(CONFIG.target_rate, 1 - CONFIG.target_rate))
    return {"regret": regret, "anchor": round(anchor, 3), "score": round(0.4 * (regret or 0.0) + 0.6 * anchor, 3)}


def stage_hinted(task_dir: Path, attempts: int | None = None) -> dict:
    """The feasibility proof for a task nobody solved cold.

    hinted solves it   -> feasible: hard, and the hint closes the gap (high regret)
    hinted fails too   -> unproven: possibly impossible or underspecified; do not ship it
    Hinted attempts never count toward the pass rate."""
    path = task_dir / "authoring" / "gate" / "hinted.json"
    if path.exists() and (old := json.loads(path.read_text())).get("checksum") == checksum(task_dir) and old.get("graded"):
        return old  # same task, already measured
    cold = json.loads((task_dir / "authoring" / "gate" / "difficulty.json").read_text())["models"]
    cold_solved, cold_graded = sum(m["solved"] for m in cold.values()), sum(m["graded"] for m in cold.values())
    p_cold = cold_solved / cold_graded if cold_graded else 0.0
    if CONFIG.hinted == "off" or (CONFIG.hinted == "auto" and (cold_solved > 0 or cold_graded < 6)):
        why = "a cold solve already proves the task can be done" if cold_solved else "waiting for more cold attempts" if CONFIG.hinted != "off" else "RSI_HINTED=off"
        return _save(task_dir, "hinted", True, {"skipped": True, **design_score(p_cold, None), "note": f"skipped: {why}"})

    model = CONFIG.solver_models[0]
    with tempfile.TemporaryDirectory(prefix="rsi-hint-") as tmp:
        dst = Path(tmp) / f"{task_dir.name}--hinted"
        shutil.copytree(task_dir, dst, ignore=shutil.ignore_patterns("gate"))
        with (dst / "instruction.md").open("a") as f:
            f.write(hint_text(task_dir))
        trials = harbor.run_job(dst, CONFIG.solver_agent, model=model, attempts=attempts or CONFIG.hinted_attempts,
                                jobs_dir=_jobs(task_dir), job_name=f"gate-hinted-{model.split('/')[-1]}")
    tally = _tally(task_dir, trials)
    failed = list(tally["failed_tests"].values())
    always = sorted(set.intersection(*map(set, failed))) if failed and not tally["solved"] else []
    verdict = "feasible" if tally["solved"] else "need_more" if tally["graded"] < 2 else "unproven"
    note = {"feasible": f"feasible: hinted {tally['solved']}/{tally['graded']} against cold {cold_solved}/{cold_graded}",
            "unproven": f"UNPROVEN: 0/{tally['graded']} even with every criterion and trap spelled out; {len(always)} tests failed every time",
            "need_more": "hinted attempts errored; run again"}[verdict]
    return _save(task_dir, "hinted", verdict == "feasible", {
        "model": model, "agent": CONFIG.solver_agent, **tally, "cold": {"solved": cold_solved, "graded": cold_graded},
        "verdict": verdict, "always_failed": always, **design_score(p_cold, tally["rate"]), "note": note})


# ---------------------------------------------------------------- inspect

INSPECT = """Ticket given to the agent:
<<<
{ticket}
>>>

Hidden test -> criterion map:
{criteria}

A cold attempt failed these hidden tests: {failed}
{measured}
For each failing test, quote the ONE ticket sentence it enforces, word for word. If no
sentence plainly states the behaviour (it needs a second reading or an unstated
assumption), mark it unfair.

JSON: {{"tests": [{{"test": "...", "sentence": "... or null", "fair": true|false}}]}}"""


def divergent_tests(failed_tests: dict[str, list[str]], messages: dict[str, dict[str, str]], min_attempts: int = 3) -> list[str]:
    """Tests that every failed attempt failed, each with a different error. Attempts that agree
    on how they fail usually share a misreading of a stated rule; attempts that each fail their
    own way are usually guessing at a rule the ticket never stated."""
    failing = [set(v) for v in failed_tests.values() if v]
    if len(failing) < min_attempts:
        return []
    out = []
    for test in sorted(set.intersection(*failing)):
        seen = [messages.get(trial, {}).get(test, "") for trial, v in failed_tests.items() if v]
        if all(seen) and len(set(seen)) == len(seen):
            out.append(test)
    return out


def stage_inspect(task_dir: Path) -> dict:
    """Fairness. The judge must tie every failing test to a ticket sentence, and the quote is
    checked against the ticket mechanically. Each distinct set of failing tests gets
    CONFIG.inspect_reviews independent reviews; one 'unfair' is enough (a single review misses
    too much). What the attempts did is evidence too: tests that every attempt failed in its
    own way, and tests that even hinted attempts all failed, are flagged for the judge."""
    gate_dir = task_dir / "authoring" / "gate"
    diff = json.loads((gate_dir / "difficulty.json").read_text())
    ticket = (task_dir / "instruction.md").read_text()
    criteria = (task_dir / "authoring" / "criteria.json").read_text()
    flat = " ".join(ticket.split())

    failed_tests = {t: f for m in diff["models"].values() for t, f in m["failed_tests"].items()}
    messages = {t: f for m in diff["models"].values() for t, f in m.get("failure_messages", {}).items()}
    suspects = {t: "every attempt failed it, each with a different error" for t in divergent_tests(failed_tests, messages)}
    hinted_path = gate_dir / "hinted.json"
    if hinted_path.exists() and (h := json.loads(hinted_path.read_text())).get("checksum") == diff.get("checksum"):
        suspects.update({t: "every hinted attempt failed it too" for t in h.get("always_failed", [])})

    votes: dict[str, int] = {}
    sets = sorted({tuple(sorted(f)) for f in failed_tests.values() if f})
    for failed in sets:
        flagged = {t: why for t, why in suspects.items() if t in failed}
        measured = ("\nMeasured, and usually the sign of a requirement the ticket never states. Look twice before calling these fair:\n"
                    + "\n".join(f"- {t}: {why}" for t, why in flagged.items()) + "\n") if flagged else ""
        for _ in range(max(1, CONFIG.inspect_reviews)):
            verdict = llm.ask_json(INSPECT.format(ticket=ticket, criteria=criteria, failed=list(failed), measured=measured), CONFIG.judge_model)
            for t in verdict["tests"]:
                if not t.get("fair") or not t.get("sentence") or " ".join(t["sentence"].split()) not in flat:
                    votes[t["test"]] = votes.get(t["test"], 0) + 1
    unfair = sorted(votes)
    return _save(
        task_dir, "inspect", not unfair,
        {"judge_model": CONFIG.judge_model, "reviews": max(1, CONFIG.inspect_reviews), "unfair_tests": unfair, "unfair_votes": votes,
         "suspect_tests": suspects,
         "note": f"{len(sets)} distinct failures read {max(1, CONFIG.inspect_reviews)} times each, {len(unfair)} unfair tests, {len(suspects)} flagged by the attempts"},
    )


# ---------------------------------------------------------------- driver

RUNNERS = {
    "shape": stage_shape, "audit": stage_audit, "oracle": stage_oracle, "nop": stage_nop,
    "mutations": stage_mutations, "shortcut": stage_shortcut,
    "difficulty": stage_difficulty, "hinted": stage_hinted, "inspect": stage_inspect,
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
