"""Stage 3: combine several independent failure modes into one task design.

Mode choice is where the loop learns from its own measurements. `runs/history.jsonl` records,
for every calibrated task, which modes it stacked and, per graded attempt, which of them the
solver slipped on. From that:

    handle rate    per mode: how often the current solvers get it right (a Beta posterior;
                   rows measured with other solvers count at CONFIG.stale_weight)
    interaction    per pair of modes seen in one task: do slips on one go with slips on the
                   other? rho < 1: together they are harder than their separate rates say
    prediction     a stack is solved when every mode is handled:
                   P(solve) = product of handle rates x product of pair interactions

`pick_modes` draws handle rates from the posteriors (Thompson sampling, so untried modes
still get tried) and takes the stack whose predicted pass rate is nearest the target. The
earlier rule, "take the modes that slip most", aimed at zero solves; the target is a band.

`backtest` scores the prediction leave-one-out on the measured history. It is the offline
check a change to this file has to pass (rsi/improve.py).
"""

from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from . import archive, llm
from .config import CONFIG
from .store import read_json, read_jsonl, write_json

HISTORY = Path("runs/history.jsonl")
EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "frist-hackathon"
PRIOR_HISTORY = EXAMPLES / "history.jsonl"

SPREAD_WEIGHT = 0.15  # how much an uneven stack costs, in units of pass rate

DESIGN = """You design one RL coding task for frontier coding agents. Target: a cold
frontier agent fully solves it about {target:.0%} of the time. Every failure must be FAIR:
the agent missed something the ticket plainly states.

Stack these independent failure modes, EACH in a different acceptance criterion and a
different part of the code:
{modes}

From past measurements a task that stacks exactly these should be solved about
{predicted:.0%} of the time. Modes with a low handle rate carry the difficulty: give them the
most surfaces. A mode solvers nearly always handle adds work, not difficulty.

Measured designs from earlier runs (cold Opus 5.5 pass counts included). Copy what landed
in band, avoid what was too easy:
{examples}

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

Tasks already built (a design that is one of these in other words, the same rules on the same
kind of system, is a duplicate; the same failure modes on a different system is not):
{pool}

JSON: {{"central_change_found": true|false, "change": "...", "unfair_criteria": ["AC id", ...],
"duplicate_of": "<name of the task it duplicates, or null>", "verdict": "accept|reject", "why": "..."}}"""


def history_rows() -> list[dict]:
    return read_jsonl(PRIOR_HISTORY) + read_jsonl(HISTORY)


def _measurements(row: dict):
    """(solver model or None, attempts, {mode: slips}, patterns or None) per measurement in a row.
    New rows carry one measurement per solver model with per-attempt patterns; rows from before
    that (the hackathon history) carry pooled counts only."""
    if row.get("by_model"):
        for model, m in row["by_model"].items():
            yield model, m.get("attempts", 0), m.get("mode_slips", {}), m.get("patterns")
    else:
        models = list(row.get("models", {}))
        yield (models[0] if len(models) == 1 else None), row.get("attempts", 0), row.get("mode_slips", {}), row.get("patterns")


@dataclass
class Stats:
    """Weighted counts from history. slips/handled per mode; per pair of modes seen together:
    attempts, attempts that handled both, and attempts that handled each."""
    slips: dict[str, float] = field(default_factory=dict)
    handled: dict[str, float] = field(default_factory=dict)
    pairs: dict[frozenset, list[float]] = field(default_factory=dict)

    def beta(self, mode: str) -> tuple[float, float]:
        """Posterior over the mode's SLIP rate: Beta(1 + slips, 1 + handled)."""
        return 1 + self.slips.get(mode, 0), 1 + self.handled.get(mode, 0)

    def handle_rate(self, mode: str) -> float:
        a, b = self.beta(mode)
        return b / (a + b)

    def rho(self, a: str, b: str, pseudo: float = 4.0) -> float:
        """Interaction of two modes: handled-both / what independence predicts, shrunk to 1."""
        n, both, ha, hb = self.pairs.get(frozenset((a, b)), (0, 0, 0, 0))
        if not n:
            return 1.0
        return (both + pseudo) / (ha * hb / n + pseudo)


def stats(rows: list[dict] | None = None, solvers: tuple[str, ...] | None = None) -> Stats:
    solvers = CONFIG.solver_models if solvers is None else solvers
    st = Stats()
    for row in history_rows() if rows is None else rows:
        modes = row.get("modes") or list(row.get("mode_slips", {}))
        for model, n, slips, patterns in _measurements(row):
            w = 1.0 if model is None or model in solvers else CONFIG.stale_weight
            for m in modes:  # a mode in the task with no recorded slip was handled every time
                st.slips[m] = st.slips.get(m, 0) + w * slips.get(m, 0)
                st.handled[m] = st.handled.get(m, 0) + w * (n - slips.get(m, 0))
            if patterns is None:
                continue  # pooled counts only: nothing to say about which slips came together
            failed = [set(p) for p in patterns.get("slips", [])]
            total = patterns.get("solved", 0) + len(failed)
            for a, b in itertools.combinations(sorted(modes), 2):
                cell = st.pairs.setdefault(frozenset((a, b)), [0.0, 0.0, 0.0, 0.0])
                cell[0] += w * total
                cell[1] += w * (patterns.get("solved", 0) + sum(1 for f in failed if a not in f and b not in f))
                cell[2] += w * (patterns.get("solved", 0) + sum(1 for f in failed if a not in f))
                cell[3] += w * (patterns.get("solved", 0) + sum(1 for f in failed if b not in f))
    return st


def predict(mode_ids: list[str], st: Stats, handle: dict[str, float] | None = None) -> float:
    """P(a cold solver handles every mode in the stack)."""
    p = math.prod((handle or {}).get(m, st.handle_rate(m)) for m in mode_ids)
    for a, b in itertools.combinations(sorted(mode_ids), 2):
        p *= st.rho(a, b)
    return min(1.0, max(0.0, p))


def mode_weights(modes: list[dict]) -> dict[str, tuple[float, float]]:
    """Beta(1 + slips, 1 + handled) per mode, from past calibrated tasks."""
    st = stats()
    return {m["id"]: st.beta(m["id"]) for m in modes}


def pick_modes(modes: list[dict], k: int, rng: random.Random, k_max: int | None = None, st: Stats | None = None) -> list[dict]:
    """k to k_max modes, one per category, whose predicted pass rate is nearest the target.
    Handle rates are drawn from their posteriors, so a mode with little history is still tried."""
    st = st or stats()
    handle = {}
    for m in modes:
        a, b = st.beta(m["id"])
        handle[m["id"]] = 1 - rng.betavariate(a, b)
    cat = {m["id"]: m.get("category", m["id"]) for m in modes}
    by_id = {m["id"]: m for m in modes}
    sizes = range(min(k, len(modes)), min(k_max or k, len(modes)) + 1)
    stacks = [c for size in sizes for c in itertools.combinations(sorted(by_id), size) if len({cat[i] for i in c}) == size]
    if not stacks:  # fewer categories than k: fall back to distinct modes
        stacks = [c for size in sizes for c in itertools.combinations(sorted(by_id), size)]
    if len(stacks) > 20000:
        stacks = rng.sample(stacks, 20000)
    rng.shuffle(stacks)  # ties go to chance, not to alphabetical order

    def cost(c: tuple[str, ...]) -> float:
        # One mode carrying all the difficulty gave a coin flip at the hackathon, not a band
        # (docs/DIFFICULTY_RULES.md), so among stacks near the target prefer the one whose
        # modes share the load: a small penalty on the spread of the log handle rates.
        logs = [math.log(max(handle[i], 1e-6)) for i in c]
        mean = sum(logs) / len(logs)
        spread = math.sqrt(sum((x - mean) ** 2 for x in logs) / len(logs))
        return abs(predict(list(c), st, handle) - CONFIG.target_rate) + SPREAD_WEIGHT * spread

    return [by_id[i] for i in min(stacks, key=cost)]


def pick_extra(pool: list[dict], current_rate: float, rng: random.Random, st: Stats | None = None) -> dict:
    """One more mode for a task that measured too easy: the one whose handle rate would bring
    the measured pass rate nearest the target."""
    st = st or stats()

    def cost(m: dict) -> float:
        a, b = st.beta(m["id"])
        return abs(current_rate * (1 - rng.betavariate(a, b)) - CONFIG.target_rate)

    return min(rng.sample(pool, len(pool)), key=cost)


def backtest(rows: list[dict] | None = None) -> dict:
    """Leave-one-task-out: predict each measured task's pass rate from the others. `baseline`
    predicts the mean pass rate of the other tasks; the model earns its keep only if it beats that."""
    rows = [r for r in (history_rows() if rows is None else rows) if r.get("attempts") and r.get("modes")]
    out = []
    for i, row in enumerate(rows):
        rest = rows[:i] + rows[i + 1:]
        actual = row["solved"] / row["attempts"]
        base = sum(r["solved"] for r in rest) / max(1, sum(r["attempts"] for r in rest))
        out.append({"task": row.get("task"), "actual": round(actual, 3), "predicted": round(predict(row["modes"], stats(rest)), 3), "baseline": round(base, 3)})
    n = max(1, len(out))
    return {"n": len(out), "mae": round(sum(abs(r["predicted"] - r["actual"]) for r in out) / n, 3),
            "baseline_mae": round(sum(abs(r["baseline"] - r["actual"]) for r in out) / n, 3), "rows": out}


def example_designs() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((EXAMPLES / "designs").glob("*.json"))]


def built() -> str:
    """One line per task that already exists: the pool and the worked examples."""
    rows = [(r["task"], r.get("modes", []), r.get("summary", "")) for r in archive.pool()]
    rows += [(d["slug"], d.get("modes", []), d.get("summary", "")) for d in example_designs()]
    return "\n".join(f"- {name} [{', '.join(modes)}]: {summary}" for name, modes, summary in rows)


def examples() -> str:
    out = []
    for d in example_designs():
        m = d.get("measured", {})
        out.append(
            f"- {d['slug']} [{', '.join(d['modes'])}]: {d['summary']} "
            f"Measured {m.get('solved')}/{m.get('attempts')} solved; slips: {m.get('slips')}. {d.get('lesson', '')}"
        )
    return "\n".join(out + ([archive.saturated_examples()] if archive.pool("retired") else []))


def design(modes: list[dict], k: int, rng: random.Random, tries: int = 3) -> dict:
    st = stats()
    for attempt in range(tries):
        chosen = pick_modes(modes, k, rng, k_max=max(k, CONFIG.max_modes), st=st)
        predicted = predict([m["id"] for m in chosen], st)
        text = "\n".join(
            f"- {m['id']}: {m['mode']} (textbook build: {m.get('textbook_build', '?')}; "
            f"hidden case: {m.get('hidden_case', '?')}; evidence: "
            f"{', '.join(p['url'] for p in m.get('papers', [])) or m.get('evidence', '')}; "
            f"measured: handled in {st.handled.get(m['id'], 0):g} of {st.handled.get(m['id'], 0) + st.slips.get(m['id'], 0):g} attempts)"
            for m in chosen
        )
        d = llm.ask_json(DESIGN.format(modes=text, target=CONFIG.target_rate, predicted=predicted, examples=examples()), CONFIG.author_model)
        sim, twin = archive.nearest(d, example_designs())
        if sim >= CONFIG.novelty_max:  # a near copy by wording: no review needed to see it
            print(f"combine: design {attempt + 1} rejected: {sim:.2f} similar to {twin}")
            continue
        review = llm.ask_json(CENTRAL.format(design=d, pool=built()), CONFIG.judge_model)
        d["modes"] = [m["id"] for m in chosen]
        d["predicted_rate"] = round(predicted, 3)
        d["novelty"] = {"nearest": twin, "similarity": round(sim, 3)}
        d["design_review"] = review
        if review.get("verdict") == "accept" and not review.get("central_change_found") and not review.get("duplicate_of"):
            return d
        print(f"combine: design {attempt + 1} rejected: {review.get('why', '')[:200]}")
    raise RuntimeError("combine: no design survived the central-change and novelty review")


def run(out_dir: Path, seed: int | None = None) -> Path:
    modes = read_json(out_dir / "failure_modes.json")
    rng = random.Random(seed)
    d = design(modes, CONFIG.min_modes, rng)
    path = out_dir / "design.json"
    write_json(path, d)
    print(f"combine: {d['slug']} stacks {', '.join(d['modes'])} -> {path}")
    return path
