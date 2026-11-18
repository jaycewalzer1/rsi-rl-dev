"""Stage 7: read what the solvers built when they failed, and feed it back as failure modes.

The modes the factory stacks came from complaints on X and Hacker News, checked against
papers. That is hearsay next to what the loop measures itself: for every failed attempt, the
patch the solver wrote (agent.patch, docs/TASK_CONTRACT.md) and the hidden tests that caught it.

    read      one reading per failed attempt (judge model): given the ticket, the reference,
              the attempt's patch and its failing tests by criterion, say what the attempt
              built for each failed criterion, and whether the slip is the failure mode the
              criterion was built on or a different one. Every reading must quote the
              attempt's patch; one whose quotes are not in the patch is thrown away
    file      runs/mined.jsonl, one row per (attempt, criterion)
    promote   slips of a different kind, seen in at least two attempts, are grouped into
              candidate modes (runs/mined_modes.json, origin "rollout"). papers.merge_seeds
              adds them to the pool the next run draws from

    rsi mine TASK_DIR        read one calibrated task's failed attempts
    rsi mine --promote       group what has been read into new modes
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import gate, harbor, llm
from .config import CONFIG
from .store import read_json, read_jsonl, short_hash, write_json

MINED = Path("runs/mined.jsonl")
MODES = Path("runs/mined_modes.json")

READ = """A coding agent attempted this ticket cold and failed hidden tests. Work out what it built
and why that failed.

TICKET
<<<
{ticket}
>>>

REFERENCE SOLUTION, as a diff:
<<<
{reference}
>>>

THE ATTEMPT: what the agent changed, as a diff:
<<<
{patch}
>>>

FAILED CRITERIA (the failure mode each was built on, its ticket sentence, the failing tests and how they failed):
{failed}

For each failed criterion:
1. quote one or two lines from THE ATTEMPT that show what it built for this rule, word for
   word. If it built nothing for the rule, quote the nearest line it did write and say so.
2. say in one sentence what the attempt did and what the hidden case needed.
3. is that the failure mode the criterion was built on ("documented") or a different kind of
   slip ("other")? For "other", name it: a short kebab-case id, one sentence defining the mode
   in general terms (no names from this ticket), the textbook build that produces it and the
   hidden case that catches it.

JSON: [{{"criterion": "AC1", "quotes": ["..."], "slip": "...", "kind": "documented"|"other",
"other": {{"id": "...", "mode": "...", "textbook_build": "...", "hidden_case": "..."}}}}]"""

GROUP = """Below are slips coding agents made on hidden cases, each judged NOT to be the failure
mode its criterion was built on. Group the ones that are the same kind of slip. Only a slip that
would recur on other tickets is a failure mode.

KNOWN MODES (do not re-invent these; a slip that is really one of them goes under its id):
{known}

SLIPS:
{slips}

JSON: [{{"id": "kebab-case: a known id, or a new one", "mode": "one general sentence",
"textbook_build": "...", "hidden_case": "...", "members": [<slip numbers>]}}]"""


def attempts(task_dir: Path) -> list[dict]:
    """Failed graded attempts of the last difficulty measurement that recorded a patch."""
    diff_path = task_dir / "authoring" / "gate" / "difficulty.json"
    if not diff_path.exists():
        return []
    trials = {t.name: t for job in sorted(gate._jobs(task_dir).glob("gate-difficulty-*")) if job.is_dir() for t in harbor.read_trials(job)}
    out = []
    for model, m in json.loads(diff_path.read_text())["models"].items():
        for name, failed in m["failed_tests"].items():
            patch = harbor.agent_patch(trials[name]) if name in trials else None
            if failed and patch and patch.strip():
                out.append({"trial": name, "model": model, "failed": failed, "patch": patch,
                            "messages": m.get("failure_messages", {}).get(name, {})})
    return out


def read_attempt(task_dir: Path, attempt: dict) -> list[dict]:
    """One judge reading of one failed attempt; only readings that quote the patch survive."""
    criteria = json.loads((task_dir / "authoring" / "criteria.json").read_text())
    failed = set(attempt["failed"])
    slipped = gate.slipped_criteria(attempt["failed"], criteria)
    blocks = []
    for c in criteria:
        hit = [t for t in c.get("tests", []) if t in failed]
        if hit and c["id"] in slipped:
            how = "; ".join(f"{t}: {attempt['messages'].get(t, '')}".rstrip(": ") for t in hit[:6])
            blocks.append(f"- {c['id']} (mode {c.get('mode', '?')}): \"{c.get('sentence') or c.get('text', '')}\"\n  failing: {how}")
    if not blocks:
        return []
    readings = llm.ask_json(READ.format(ticket=(task_dir / "instruction.md").read_text(), patch=attempt["patch"][:40000],
                                        reference=(task_dir / "solution" / "reference.patch").read_text()[:40000],
                                        failed="\n".join(blocks)), CONFIG.judge_model)
    mode_of = {c["id"]: c.get("mode", c["id"]) for c in criteria}
    rows = []
    for r in readings if isinstance(readings, list) else []:
        quotes = [q for q in r.get("quotes", []) if len(core := q.strip().lstrip("+- ").strip()) >= 8 and core in attempt["patch"]]
        if not quotes or r.get("criterion") not in mode_of:
            continue  # a reading that cannot point at the patch is an opinion, not evidence
        other = r.get("other") if r.get("kind") == "other" and isinstance(r.get("other"), dict) and r["other"].get("id") else None
        rows.append({"date": time.strftime("%Y-%m-%d"), "task": task_dir.name, "trial": attempt["trial"], "model": attempt["model"],
                     "criterion": r["criterion"], "mode": mode_of[r["criterion"]], "kind": "other" if other else "documented",
                     "slip": str(r.get("slip", ""))[:400], "quotes": quotes[:2], "other": other, "judge_model": CONFIG.judge_model})
    return rows


def run(task_dir: Path) -> list[dict]:
    """Read every failed attempt of a task and file the readings. Two attempts that wrote the same
    patch and failed the same tests share one reading (one judge call), filed under each: they
    are still two attempts making that slip."""
    filed = read_jsonl(MINED)
    done = {(r["task"], r["trial"]) for r in filed}
    by_key: dict[str, list[dict]] = {}
    for r in filed:
        if r.get("key") and r["trial"] == by_key.get(r["key"], [r])[0]["trial"]:
            by_key.setdefault(r["key"], []).append(r)
    rows = []
    for a in attempts(task_dir):
        if (task_dir.name, a["trial"]) in done:
            continue
        key = short_hash(task_dir.name + a["patch"] + "|".join(sorted(a["failed"])))
        if key not in by_key:
            by_key[key] = [{**r, "key": key} for r in read_attempt(task_dir, a)]
            rows += by_key[key]
        else:
            rows += [{**r, "trial": a["trial"], "model": a["model"]} for r in by_key[key]]
    if rows:
        MINED.parent.mkdir(parents=True, exist_ok=True)
        with MINED.open("a") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    other = sum(r["kind"] == "other" for r in rows)
    print(f"mine: {task_dir.name}: {len(rows)} readings, {other} not the documented mode -> {MINED}")
    return rows


def promote(known: list[dict], min_support: int = 2) -> list[dict]:
    """Group the undocumented slips into candidate modes. A group becomes a mode only if it is
    new and at least `min_support` different attempts show it."""
    slips = [r for r in read_jsonl(MINED) if r["kind"] == "other"]
    if len({(r["task"], r["trial"]) for r in slips}) < min_support:
        return []
    listing = "\n".join(f"[{i}] {r['other']['id']}: {r['other'].get('mode', '')} (what happened: {r['slip']})" for i, r in enumerate(slips))
    groups = llm.ask_json(GROUP.format(known="\n".join(f"- {m['id']}: {m['mode'][:200]}" for m in known), slips=listing), CONFIG.judge_model)
    known_ids = {m["id"] for m in known}
    modes = []
    for g in groups if isinstance(groups, list) else []:
        members = [slips[i] for i in g.get("members", []) if isinstance(i, int) and 0 <= i < len(slips)]
        support = {(r["task"], r["trial"]) for r in members}
        if not g.get("id") or g["id"] in known_ids or len(support) < min_support:
            continue
        modes.append({"id": g["id"], "mode": g.get("mode", ""), "textbook_build": g.get("textbook_build", ""), "hidden_case": g.get("hidden_case", ""),
                      "papers": [], "complaints": [], "origin": "rollout",
                      "evidence": f"read from {len(support)} failed attempts on {len({r['task'] for r in members})} task(s) in this factory",
                      "rollouts": [{"task": r["task"], "trial": r["trial"], "slip": r["slip"], "quote": r["quotes"][0]} for r in members]})
    write_json(MODES, modes)
    print(f"mine: {len(modes)} candidate modes from {len(slips)} undocumented slips -> {MODES}")
    return modes


def modes() -> list[dict]:
    """Modes promoted so far (empty until `rsi mine --promote` has found one)."""
    return read_json(MODES) if MODES.exists() else []
