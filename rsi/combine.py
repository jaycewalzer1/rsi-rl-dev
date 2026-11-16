"""Stage 3: combine several independent failure modes into one task design.

Mode choice is where the loop improves itself: `runs/history.jsonl` records, for
every calibrated task, which modes it stacked and which hidden cases attempts
failed. Modes that actually made solvers slip get picked more; modes solvers
always handled get picked less (Thompson-style sampling on a Beta posterior).
"""

from __future__ import annotations

import random
from pathlib import Path

from . import llm
from .config import CONFIG
from .store import read_json, read_jsonl, write_json

HISTORY = Path("runs/history.jsonl")

DESIGN = """You design one RL coding task for frontier coding agents. Target: a cold
frontier agent fully solves it about {target:.0%} of the time. Every failure must be FAIR:
the agent missed something the ticket plainly states.

Stack these independent failure modes, EACH in a different acceptance criterion and a
different part of the code:
{modes}

Rules learned the hard way (follow all of them):
- Frontier agents notice every trap a ticket states and repair every neighbour it names.
  They slip on exact parity at width: one general rule that must hold on many surfaces the
  ticket names only by business class, tested against a live twin (a never-used id, the
  no-failure run) on 30+ surfaces.
- Central-change test: if one filter, helper, table or lock satisfies the whole ticket,
  the design is dead. Name the textbook build and the hidden case it fails for each mode.
- Ticket: 150 to 250 words of general rules, no scenario lists, no mechanism hints.
  Exact status codes and messages go in a "Pins" section in backticks.
- Self-contained: a small but real codebase (Python or TypeScript) built in the Docker
  image, 8 to 10 criteria, 60 to 100 all-or-nothing hidden tests, 45 to 90 minutes of
  honest work. No network at test time, no timing-dependent assertions.
- Widen by cases, not by rules. More of the same kind of work is not harder.

JSON:
{{"slug": "kebab-case-name",
  "summary": "two sentences",
  "language": "python|typescript",
  "codebase": "what the starting repo contains and where the gaps are",
  "ticket": "the instruction.md text, 150-250 words",
  "criteria": [{{"id": "AC1", "text": "...", "mode": "<mode id>", "surfaces": ["..."]}}],
  "textbook_build": {{"<mode id>": {{"build": "...", "hidden_case_it_fails": "..."}}}},
  "central_change_risk": "the single change most likely to satisfy everything, and why it does not",
  "mutations": ["one plausible wrong fix per criterion, each a sentence"]}}"""

CENTRAL = """Here is an RL coding task design. Try hard to find ONE central change (a single
filter, helper, middleware, table, lock or decorator) that would satisfy every criterion.
Also check fairness: is every criterion plainly stated in the ticket?

{design}

JSON: {{"central_change_found": true|false, "change": "...", "unfair_criteria": ["AC id", ...],
"verdict": "accept|reject", "why": "..."}}"""


def mode_weights(modes: list[dict]) -> dict[str, tuple[int, int]]:
    """Beta(1 + slips, 1 + handled) per mode, from past calibrated tasks."""
    stats = {m["id"]: [1, 1] for m in modes}
    for row in read_jsonl(HISTORY):
        for mode_id, slipped in row.get("mode_slips", {}).items():
            if mode_id in stats:
                stats[mode_id][0] += slipped
                stats[mode_id][1] += row.get("attempts", 0) - slipped
    return {k: (a, b) for k, (a, b) in stats.items()}


def pick_modes(modes: list[dict], k: int, rng: random.Random) -> list[dict]:
    weights = mode_weights(modes)
    scored = sorted(modes, key=lambda m: rng.betavariate(*weights[m["id"]]), reverse=True)
    picked, cats = [], set()
    for m in scored:  # independent: one mode per category
        cat = m.get("category", m["id"])
        if cat not in cats:
            picked.append(m)
            cats.add(cat)
        if len(picked) == k:
            break
    return picked


def design(modes: list[dict], k: int, rng: random.Random, tries: int = 3) -> dict:
    for attempt in range(tries):
        chosen = pick_modes(modes, k, rng)
        text = "\n".join(
            f"- {m['id']}: {m['mode']} (textbook build: {m.get('textbook_build', '?')}; "
            f"hidden case: {m.get('hidden_case', '?')}; evidence: "
            f"{', '.join(p['url'] for p in m.get('papers', [])) or m.get('evidence', '')})"
            for m in chosen
        )
        d = llm.ask_json(DESIGN.format(modes=text, target=CONFIG.target_rate), CONFIG.author_model)
        review = llm.ask_json(CENTRAL.format(design=d), CONFIG.judge_model)
        d["modes"] = [m["id"] for m in chosen]
        d["design_review"] = review
        if review.get("verdict") == "accept" and not review.get("central_change_found"):
            return d
        print(f"combine: design {attempt + 1} rejected: {review.get('why', '')[:200]}")
    raise RuntimeError("combine: no design survived the central-change review")


def run(out_dir: Path, seed: int | None = None) -> Path:
    modes = read_json(out_dir / "failure_modes.json")
    rng = random.Random(seed)
    d = design(modes, CONFIG.min_modes, rng)
    path = out_dir / "design.json"
    write_json(path, d)
    print(f"combine: {d['slug']} stacks {', '.join(d['modes'])} -> {path}")
    return path
