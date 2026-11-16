"""Stage 4: turn a design into a Harbor task directory.

An author-model Claude Code session with file and shell tools writes the task
following docs/TASK_CONTRACT.md. It may also be asked to revise an existing task
(harden or soften) during calibration; the same contract applies.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import llm
from .config import CONFIG

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = (ROOT / "docs" / "TASK_CONTRACT.md").read_text()

BUILD = """You are building one Harbor RL task in the current directory, from this design:

{design}

Follow this contract exactly:

{contract}

Work until every file in the contract exists and `bash authoring/selfcheck.sh` (write it
as the contract says) passes locally where it can. Do not run docker. Finish with a
short list of what you wrote."""

REVISE = """The Harbor task in the current directory was measured against frontier coding
agents and needs a revision.

Measurement:
{measurement}

Instruction: {instruction}

Follow the contract (below) for every change. A change to instruction.md voids all
earlier attempts; prefer changing hidden cases over the ticket. Keep the reference
solution passing and update authoring/mutations so each criterion still has a mutation
that a hidden case catches by assertion.

{contract}"""


def run(design_path: Path, out_dir: Path) -> Path:
    design = json.loads(design_path.read_text())
    task_dir = out_dir / "task" / design["slug"]
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "authoring").mkdir(exist_ok=True)
    (task_dir / "authoring" / "design.json").write_text(json.dumps(design, indent=2))
    llm.ask(
        BUILD.format(design=json.dumps(design, indent=2), contract=CONTRACT),
        CONFIG.author_model,
        tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        cwd=task_dir,
        timeout=3 * 3600,
    )
    print(f"build: {task_dir} (author model {CONFIG.author_model})")
    return task_dir


def revise(task_dir: Path, measurement: dict, instruction: str) -> None:
    llm.ask(
        REVISE.format(
            measurement=json.dumps(measurement, indent=2)[:20000],
            instruction=instruction,
            contract=CONTRACT,
        ),
        CONFIG.author_model,
        tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        cwd=task_dir,
        timeout=2 * 3600,
    )
