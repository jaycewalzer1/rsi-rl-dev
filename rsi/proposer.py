"""How one generation of the improvement loop looks at the factory and what it is asked to do.

The loop (rsi/improve.py) is a fixed shell: it makes the worktree, enforces the protected
list, runs the tests and the comparison, and keeps the archive. This file is the part inside
the shell: what evidence a generation is shown and how the request is worded. It is NOT
protected. A generation may improve how later generations gather evidence and reason, the
same way it may improve any other part of the factory. What it cannot touch is how its work
is judged.

    evidence(runs_dir)   everything measured so far, condensed: the scorecard, the worst
                         tasks, which gate stages fail and why, unfair and suspect tests,
                         slips read from failed attempts, how well designs were aimed
    propose(...)         one headless Claude Code session in the generation's worktree
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from . import combine, llm, meter, mine
from .config import CONFIG
from .store import read_jsonl

TOOLS = ["Read", "Edit", "Write", "Glob", "Grep", "Bash"]
# the session may read anything, edit files, and run the tests and read-only git; nothing else runs without a person
ALLOWED = ["Read", "Edit", "Write", "Glob", "Grep", "Bash(uv run pytest *)", "Bash(git diff *)", "Bash(git status *)", "Bash(git log *)"]
NOTE_HEADINGS = ("## Evidence", "## Change", "## Prediction", "## How this could be wrong")

PROMPT = """You are one generation of this repository's improvement loop (read docs/RSI_LOOP.md first).
The repository is a factory: code and prompts that design, build, gate and calibrate RL coding
tasks. Your job is to change the factory so that its next tasks come out better.

THE SCORECARD (rsi/meter.py computes it from gate output; you cannot change how):
{card}

EVIDENCE FROM THE RUNS SO FAR:
{evidence}

EARLIER GENERATIONS (what was tried and what happened):
{lineage}

WHAT TO DO
1. Find the weakest number on the scorecard and the cause the evidence points to. Read the
   code and prompts involved before deciding anything.
2. Make ONE focused change that addresses that cause. The smallest change that could work.
   No refactors, no restyling, no change you cannot tie to a line of the evidence.
3. Put tests for what you changed under tests/gen/ and run `uv run pytest -q`.
4. Write docs/improvements/gen-{n}.md with exactly these parts:
     # <one-line title>
     ## Evidence                 the numbers and files that made you choose this
     ## Change                   what you changed and why it addresses the cause
     ## Prediction               which scorecard number moves, which way, roughly how far
     ## How this could be wrong  the result of a live comparison that would falsify it
If the evidence does not support a change, change no code and write the note saying what
evidence is missing. "Not enough data yet" is a good outcome; a change made to look busy is not.

RULES THE LOOP ENFORCES (a generation that breaks one is discarded)
- These paths are protected. Do not edit, delete or rename them:
{protected}
- Do not reach around them: no assigning to their attributes at run time, no setting RSI_*
  environment variables in code, no writing into a task's authoring/gate folder.
- The whole test suite must pass. If rsi/combine.py changes, its leave-one-out prediction
  error on the recorded history must not get worse.
"""


def run_dirs(runs_dir: Path) -> list[Path]:
    """Run folders that hold tasks."""
    return sorted(d for d in Path(runs_dir).glob("*") if (d / "task").is_dir())


def evidence(runs_dir: Path) -> dict:
    dirs = run_dirs(runs_dir)
    card = meter.scorecard(dirs)
    fails, notes, unfair = Counter(), defaultdict(list), []
    for run in dirs:
        for task in sorted((run / "task").glob("*")):
            for p in sorted((task / "authoring" / "gate").glob("*.json")):
                r = json.loads(p.read_text())
                if not r.get("ok", True):
                    fails[r["stage"]] += 1
                    notes[r["stage"]].append(f"{task.name}: {str(r.get('note', ''))[:240]}")
                if r.get("stage") == "inspect" and (r.get("unfair_tests") or r.get("suspect_tests")):
                    unfair.append({"task": task.name, "unfair": r.get("unfair_tests", [])[:6], "suspect": r.get("suspect_tests", {})})
    mined = read_jsonl(mine.MINED)
    history = [{k: r.get(k) for k in ("task", "kind", "modes", "attempts", "solved", "predicted", "first_rate", "rounds", "in_band", "regret")}
               for r in read_jsonl(combine.HISTORY)[-15:]]
    seconds = Counter()
    for c in read_jsonl(llm.LOG):
        seconds[c.get("model", "?")] += c.get("seconds", 0)
    worst = sorted(card["by_task"], key=lambda c: (c["in_band"], -c["attempts"]))[:8]
    return {
        "scorecard": {k: v for k, v in card.items() if k != "by_task"},
        "worst_tasks": worst,
        "gate_failures": {stage: {"tasks": n, "examples": notes[stage][:4]} for stage, n in fails.most_common()},
        "unfair_or_suspect_tests": unfair[:6],
        "slips_read_from_failed_attempts": {"readings": len(mined), "not_the_documented_mode": Counter(
            r["other"]["id"] for r in mined if r.get("kind") == "other" and r.get("other")).most_common(10)},
        "recent_history": history,
        "stack_prediction_backtest": {k: v for k, v in combine.backtest().items() if k != "rows"},
        "model_seconds": dict(seconds),
        "targets": {"pass_rate": CONFIG.target_rate, "band": [CONFIG.band_low, CONFIG.band_high]},
    }


def prompt(n: int, ev: dict, lineage: list[dict], protected: list[str]) -> str:
    past = "\n".join(f"- gen-{g['gen']} [{g['status']}] {g.get('title', '')}"
                     + (f" | live: {g['live']['verdict']['reasons']}" if g.get("live") else "") for g in lineage[-10:]) or "- none yet: you are the first"
    return PROMPT.format(n=n, card=meter.render(ev["scorecard"]), lineage=past, protected="\n".join(f"    {p}" for p in protected),
                         evidence=json.dumps({k: v for k, v in ev.items() if k != "scorecard"}, indent=1, default=str)[:24000])


def propose(tree: Path, n: int, ev: dict, lineage: list[dict], protected: list[str], budget_usd: float | None = None, model: str | None = None) -> str:
    """Run the generation's session in its worktree. Returns the session's final message; the
    work itself is whatever it left in the tree."""
    return llm.ask(prompt(n, ev, lineage, protected), model or CONFIG.author_model, tools=TOOLS, allowed=ALLOWED,
                   cwd=tree, timeout=2 * 3600, budget_usd=budget_usd)
