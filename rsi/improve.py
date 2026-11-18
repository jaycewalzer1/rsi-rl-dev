"""The improvement loop: Claude Code changes the factory, and a fixed shell decides what stays.

    rsi improve [--live N] [--adopt] [--parent REF] [--budget USD] [--force]
    rsi improve --log

One generation:
  1 evidence   everything the runs have measured, condensed (rsi/proposer.py)
  2 propose    a headless Claude Code session, in its own git worktree on branch rsi/gen-N,
               makes ONE change to the factory and writes docs/improvements/gen-N.md:
               evidence, change, prediction, how it could be wrong
  3 check      mechanical, cheapest first; the first failure discards the generation:
                 protected   no protected path was edited, deleted or renamed
                 tamper      no added line reaches around the protected code at run time
                 note        the note exists and has its four parts
                 tests       the whole test suite passes in the worktree
                 backtest    if rsi/combine.py changed: its leave-one-out prediction error on
                             the recorded history is not worse than the parent's
  4 commit     on rsi/gen-N. Nothing is merged, nothing is pushed
  5 live       (--live N) the parent and the candidate each run N tasks through the whole
               pipeline from the same history and seed; rsi/meter.py scores both from gate
               output, and meter.compare accepts or rejects
  6 archive    runs/improve/archive.jsonl: every generation, kept or not, with its numbers.
               The next generation is shown it

The shell (this file), the meter, the gate, the graders and the existing tests are
protected: a generation can change how tasks are made, never how its own work is judged.
What it is shown and how it is asked (rsi/proposer.py) is not protected.
docs/RSI_LOOP.md explains why the loop is built this way.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from . import archive, combine, meter, mine, proposer
from .config import CONFIG
from .store import read_jsonl, write_json

ROOT = Path(__file__).resolve().parent.parent
IMPROVE = CONFIG.runs_dir / "improve"

# How a generation's work is judged. `*` stays inside one folder, `**` crosses folders.
PROTECTED = [
    "rsi/improve.py", "rsi/meter.py", "rsi/gate.py", "rsi/harbor.py",          # the shell, the score, the measurements
    "rsi/config.py", "rsi/llm.py", "rsi/store.py", "rsi/__init__.py",          # targets, model calls and their log
    "rsi/family.py", "rsi/family_check.py", "rsi/families/**", "rsi/qwen.py",  # graders, the control family, the Qwen benchmark
    "tests/*.py",                                                              # existing tests (new ones go in tests/gen/)
    "examples/**",                                                             # measured history and fixtures
    "docs/TASK_CONTRACT.md", "docs/RSI_LOOP.md", "docs/WHY_NOT_TRAIN_THE_AUTHOR.md",
    "pyproject.toml", "uv.lock", ".gitignore",
]

# Added lines in rsi/ that would change protected behaviour without editing a protected file.
_GUARDED = r"(gate|meter|harbor|improve|llm|config|family|family_check|qwen|CONFIG)"
TAMPER = [
    (re.compile(rf"\b{_GUARDED}\.[A-Za-z_]\w*\s*=(?!=)"), "assigns to an attribute of protected code"),
    (re.compile(rf"setattr\(\s*{_GUARDED}\b|object\.__setattr__\("), "sets an attribute of protected code"),
    (re.compile(r"os\.environ\[\s*['\"]RSI_|os\.environ\.(setdefault|update)\(|os\.putenv\("), "sets environment variables in code"),
    (re.compile(r"importlib\.reload|sys\.modules\["), "swaps modules at run time"),
    (re.compile(r"['\"]gate['\"].*(write_text|write_json|\.write\(|open\()|difficulty_log"), "writes into a task's gate folder"),
]


def _glob_re(pattern: str) -> re.Pattern:
    return re.compile("^" + re.escape(pattern).replace(r"\*\*", ".*").replace(r"\*", "[^/]*") + "$")


def is_protected(path: str) -> bool:
    return any(_glob_re(p).match(path) for p in PROTECTED)


def protected_violations(paths: list[str]) -> list[str]:
    return sorted(p for p in paths if is_protected(p))


def tamper_lint(diff: str) -> list[str]:
    """Reasons to distrust a diff: added lines under rsi/ that match a TAMPER pattern."""
    hits, path = [], ""
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else ""
        elif line.startswith("+") and path.startswith("rsi/") and path.endswith(".py"):
            hits += [f"{path}: {why}: {line[1:].strip()[:120]}" for pattern, why in TAMPER if pattern.search(line[1:])]
    return hits


def git(*args: str, cwd: Path | None = None, check: bool = True) -> str:
    p = subprocess.run(["git", *args], cwd=cwd or ROOT, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr.strip()[-500:]}")
    return p.stdout.strip()


def lineage() -> list[dict]:
    return read_jsonl(IMPROVE / "archive.jsonl")


def note_problems(text: str) -> list[str]:
    problems = [] if text.lstrip().startswith("# ") else ["no '# title' line"]
    return problems + [f"missing '{h}'" for h in proposer.NOTE_HEADINGS if h not in text]


def validate(tree: Path, parent: str, n: int, test_cmd: list[str] | None = None) -> dict:
    """The mechanical checks, cheapest first. Stops at the first failure."""
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        checks.append({"check": name, "ok": bool(ok), "detail": detail[:600]})
        return bool(ok)

    def done(status: str) -> dict:
        return {"ok": status == "ok", "status": status, "checks": checks, "changed": changed}

    git("add", "-A", cwd=tree)
    changed = [p for p in git("diff", "--cached", "--no-renames", "--name-only", parent, cwd=tree).splitlines() if p]
    code_changed = [p for p in changed if not p.startswith("docs/improvements/")]
    if not changed:
        return done("no-change")
    bad = protected_violations(changed)
    if not check("protected", not bad, f"touches protected paths: {bad}" if bad else f"{len(changed)} files, none protected"):
        return done("rejected: protected")
    hits = tamper_lint(git("diff", "--cached", "--no-renames", parent, cwd=tree))
    if not check("tamper", not hits, "; ".join(hits) or "no line reaches around protected code"):
        return done("rejected: tamper")
    note = tree / "docs" / "improvements" / f"gen-{n}.md"
    problems = note_problems(note.read_text()) if note.exists() else [f"{note.relative_to(tree)} was not written"]
    if not check("note", not problems, "; ".join(problems) or "evidence, change, prediction, falsifier"):
        return done("rejected: note")
    if not code_changed:
        return done("no-change")  # a note that says the evidence is not there yet: a fine outcome, nothing to adopt
    t = subprocess.run(test_cmd or ["uv", "run", "--project", str(tree), "pytest", "-q"], cwd=tree, capture_output=True, text=True, timeout=3600)
    if not check("tests", t.returncode == 0, (t.stdout + t.stderr).strip()[-600:]):
        return done("rejected: tests")
    if "rsi/combine.py" in changed:
        rows = combine.history_rows()
        ours, theirs = combine.backtest(rows)["mae"], candidate_backtest(tree, rows)
        if not check("backtest", theirs is not None and theirs <= ours + 0.02, f"prediction error {ours} -> {theirs} on {len(rows)} recorded tasks"):
            return done("rejected: backtest")
    return done("ok")


def candidate_backtest(tree: Path, rows: list[dict]) -> float | None:
    """The candidate's own combine.backtest on the recorded history (run with the tree's code)."""
    script = "import json, sys; from rsi import combine; print(json.dumps(combine.backtest(json.load(sys.stdin))['mae']))"
    c = subprocess.run(["uv", "run", "--project", str(tree), "python", "-c", script], cwd=tree, input=json.dumps(rows),
                       capture_output=True, text=True, timeout=600)
    return json.loads(c.stdout.strip().splitlines()[-1]) if c.returncode == 0 and c.stdout.strip() else None


def live(gen_dir: Path, parent_tree: Path, tree: Path, tasks: int, seed: int) -> dict:
    """Parent and candidate each make `tasks` tasks from the same history and seed, in separate
    folders. The scorecards come from this process's meter, not from either tree."""
    cards = {}
    for arm, code in (("parent", parent_tree), ("candidate", tree)):
        cwd = gen_dir / f"live-{arm}"
        (cwd / "runs").mkdir(parents=True, exist_ok=True)
        for src in (combine.HISTORY, archive.ARCHIVE, mine.MINED, mine.MODES):  # both arms start from what is known today
            if src.exists():
                shutil.copy(src, cwd / "runs" / src.name)
        env = {**os.environ, "RSI_RUNS_DIR": "runs", "RSI_LLM_LOG": "runs/llm_calls.jsonl"}
        cmd = ["uv", "run", "--project", str(code), "rsi", "loop", "--run", "live", "--tasks", str(tasks), "--seed", str(seed), "--skip-scan"]
        print(f"improve: live arm {arm}: {' '.join(cmd)}  (in {cwd})", flush=True)
        p = subprocess.run(cmd, cwd=cwd, env=env)
        cards[arm] = {**meter.scorecard([cwd / "runs" / "live"]), "exit": p.returncode}
    return {"tasks_per_arm": tasks, "seed": seed, "parent": cards["parent"], "candidate": cards["candidate"],
            "verdict": meter.compare(cards["parent"], cards["candidate"])}


def run(parent: str = "HEAD", live_tasks: int = 0, adopt: bool = False, budget_usd: float | None = None, force: bool = False,
        min_tasks: int = 3, seed: int = 0, test_cmd: list[str] | None = None) -> dict:
    card = meter.scorecard(proposer.run_dirs(CONFIG.runs_dir))
    if card["tasks"] < min_tasks and not force:
        print(f"improve: {card['tasks']} measured tasks under {CONFIG.runs_dir}/ and a generation needs at least {min_tasks} to reason from.\n"
              f"Run `rsi loop --tasks {min_tasks}` first (or pass --force to propose from the code alone).")
        return {"status": "no-evidence"}

    n = 1 + max((g["gen"] for g in lineage()), default=0)
    gen_dir = (IMPROVE / f"gen-{n}").resolve()
    tree, parent_tree, branch = gen_dir / "tree", gen_dir / "parent", f"rsi/gen-{n}"
    parent_sha = git("rev-parse", parent)
    gen_dir.mkdir(parents=True, exist_ok=True)
    git("worktree", "add", "-q", "-b", branch, str(tree), parent_sha)
    row = {"gen": n, "date": time.strftime("%Y-%m-%d %H:%M"), "parent": parent_sha, "branch": branch, "model": CONFIG.author_model,
           "scorecard_before": {k: v for k, v in card.items() if k != "by_task"}}
    try:
        ev = proposer.evidence(CONFIG.runs_dir)
        write_json(gen_dir / "evidence.json", ev)
        try:
            row["said"] = proposer.propose(tree, n, ev, lineage(), PROTECTED, budget_usd)[-1500:]
            result = validate(tree, parent_sha, n, test_cmd)
        except Exception as e:  # noqa: BLE001  the session died: record it, keep the loop usable
            result = {"ok": False, "status": f"rejected: proposer failed ({type(e).__name__}: {str(e)[:200]})", "checks": [], "changed": []}
        (gen_dir / "change.patch").write_text(git("diff", "--cached", "--no-renames", parent_sha, cwd=tree, check=False) + "\n")
        note = tree / "docs" / "improvements" / f"gen-{n}.md"
        row.update(status=result["status"], checks=result["checks"], changed=result["changed"],
                   title=note.read_text().lstrip().splitlines()[0].lstrip("# ").strip() if note.exists() and note.read_text().strip() else "")
        if result["ok"]:
            git("commit", "-q", "-m", f"rsi improve gen-{n}: {row['title']}\n\nOne generation of the improvement loop (rsi/improve.py), from parent "
                f"{parent_sha[:10]}.\nEvidence, prediction and falsifier: docs/improvements/gen-{n}.md\nChecks passed: "
                f"{', '.join(c['check'] for c in result['checks'])}.", cwd=tree)
            row.update(status="validated-offline", commit=git("rev-parse", "HEAD", cwd=tree))
            if live_tasks:
                git("worktree", "add", "-q", "--detach", str(parent_tree), parent_sha)
                row["live"] = live(gen_dir, parent_tree, tree, live_tasks, seed)
                row["status"] = "accepted-live" if row["live"]["verdict"]["accept"] else "rejected-live"
    finally:
        for t in (tree, parent_tree):
            if t.exists():
                git("worktree", "remove", "--force", str(t), check=False)
        if "commit" not in row:
            git("branch", "-D", branch, check=False)  # nothing was committed: the patch file is the record
            row["branch"] = None
        IMPROVE.mkdir(parents=True, exist_ok=True)
        with (IMPROVE / "archive.jsonl").open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    print(report(row, gen_dir))
    if adopt and row["status"] == "accepted-live":
        if git("rev-parse", "HEAD") == parent_sha and not git("status", "--porcelain", "--untracked-files=no"):
            git("merge", "--ff-only", branch)
            print(f"improve: adopted: the current branch now includes {branch}")
        else:
            print(f"improve: not adopted automatically (HEAD moved or the tree has changes). To adopt: git merge {branch}")
    return row


def report(row: dict, gen_dir: Path | None = None) -> str:
    lines = [f"improve: gen-{row['gen']} {row['status']}" + (f": {row['title']}" if row.get("title") else "")]
    lines += [f"  {'pass' if c['ok'] else 'FAIL'}  {c['check']:<10} {c['detail'][:160]}" for c in row.get("checks", [])]
    if row.get("live"):
        v = row["live"]["verdict"]
        lines.append(f"  live ({row['live']['tasks_per_arm']} tasks per arm, weak evidence at this size): "
                     f"{'ACCEPT' if v['accept'] else 'REJECT'}: {'; '.join(v['reasons'])}")
    if row.get("commit"):
        lines.append(f"  review: git diff {row['parent'][:10]}..{row['branch']}      adopt: git merge --ff-only {row['branch']}")
    elif gen_dir:
        lines.append(f"  what it tried: {gen_dir / 'change.patch'}")
    return "\n".join(lines)


def log() -> str:
    rows = lineage()
    return "\n".join(f"gen-{r['gen']:<3} {r['date']}  {r['status']:<28} {r.get('title', '')}" for r in rows) or "improve: no generations yet"


if __name__ == "__main__":
    run()
