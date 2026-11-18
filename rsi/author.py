"""Stage 3 for business data: author agents write task families for a dataset they have never seen.

    plan      one planner call (author model) reads the catalog, the stewards' notes and the
              documented failure modes, and proposes N different business questions, each with
              candidate rules tied to failure modes
    write     one author session per question (author model, file + shell tools) writes
              family.py to the contract in rsi/family.py and runs the mechanical checker
              (rsi/family_check.py) on the real data until it passes
    prove     a DIFFERENT model (RSI_CHECK_MODEL), shown only the ticket and sample rows, must
              write a solution that matches the oracle on every slice: for the full rule stack
              and two half stacks. That is the proof the ticket is fair and the hardest level
              is possible. A miss goes back to the author once, with the diff, to decide
              whether the sentence was unclear (fix it) or the solver misread it (leave it)
    accept    checker green AND the full stack independently solved -> runs/<run>/families.json

Nothing is accepted on the author's word: the checker and the independent solve decide.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from . import events, family as F, family_check, llm
from .config import CONFIG
from .store import read_json, write_json

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]

PLAN = """You plan RL coding tasks over a real company's data export. Below: the tables
(columns, row counts per slice, a data steward's notes, sample rows) and the documented
failure modes of AI coding agents.

Propose {n} DIFFERENT business questions ("task families") this company's engineers would
really have to compute from this data. Each family:
- asks for one function that returns {{key: integer}} (money in integer cents, counts, days),
  keyed by a business entity. Use a different entity and different tables in each family.
- has a plain version a junior engineer writes in about 15 lines (one join, one sum or count).
- has 8 candidate rules: independent twists that each change the answer on real data, each
  switchable on or off, each an instance of ONE failure mode from the list (use at least 5
  different modes per family). Good rules come from how this data really behaves: repeated
  ids, empty values, several rows per order, statuses, dates that cross a month or year,
  integer division and rounding of money, a parameter that scopes the job.
- takes a job parameter from the data (a state, a category, a month, a cutoff date) with
  at least 8 possible values that change the answer.

TABLES
{catalog}

FAILURE MODES
{modes}

JSON: [{{"id": "kebab-case", "question": "what is computed, per what", "tables": ["..."],
"param": "the job parameter and where its values come from",
"rules": [{{"id": "snake_case", "mode": "<failure mode id>", "idea": "one sentence"}}]}}]"""

WRITE = """You are writing one task family for an RL environment factory, in the current
directory, over a real company's data. The family is one Python file, `family.py`.

THE BRIEF (a plan, not a spec; change any rule that does not work on the real data):
{brief}

THE CONTRACT: read `CONTRACT.md` (the contract) and `example_family.py` (a complete family
for a different dataset: copy its shape exactly).

THE DATA: `data/tables/*.json` is slice 0 of the pruned data; `data/twins/1`, `data/twins/2`
are further slices of the same business; `data/catalog.json` has columns, keys and a
steward's notes. Look at the real rows before you write a rule: use short Python one-liners.

FAILURE MODES (every rule's "mode" must be one of these ids):
{modes}

WHAT MAKES A GOOD FAMILY
- The plain job (every rule off) is one join and one sum or count: about 15 lines.
- 8 rules. Each is ONE independent twist guarded by `if "<id>" in rules`, changing the answer
  on a real share of the data in every slice. Each has a `present` sentence and an `absent`
  sentence that states the explicit default, so a ticket built from any on/off mix says
  exactly what to compute.
- Sentences are precise enough that an engineer who reads only the ticket and a few sample
  rows writes code whose output is IDENTICAL to yours: name the columns, say how ties,
  empty values, rounding (use integer arithmetic; say `//` where you divide) and boundaries
  (inclusive or not) are handled. One or two sentences per rule. No sentence may depend on
  another rule being on. Timestamps in this data are text; say how to compare them.
- `FAMILY["pins"]` says what the keys and values of the answer are and which keys to leave
  out (for example: only keys whose value is above 0).
- `params_choices(db)` returns at least 8 job parameters taken from the data.
- Standard library only, deterministic, no file or network access, under a second per call.
- Keep the ticket short: all sentences together must fit 150 to 300 words.

THE CHECK: run
    {check}
until it prints `FAMILY CHECK: ALL PASS`. It runs your oracle on every slice and fails any
rule that barely changes the answer, any two rules with the same effect, and a ticket outside
the word limits. Replace a rule that cannot pass; do not weaken the check.

Finish with one line per rule: id, mode, and how many answer keys it changes on slice 0."""

REVISE = """`family.py` in the current directory passed the mechanical check, but an
independent engineer (a different model), given only the ticket and sample rows, wrote a
solution whose output does not match your oracle. For each miss below you get the ticket
they saw, their code, and example keys where the answers differ.

{misses}

For each miss decide which it is:
 (a) the ticket was unclear or left something unsaid (a tie, an empty value, a boundary, a
     rounding step, which rows count): fix the sentence, or `FAMILY["pins"]`, so the ticket
     alone determines your output. Prefer this when in doubt: trace one differing key by
     hand in the data before deciding.
 (b) your oracle does not do what your own sentence says: fix the oracle.
 (c) the ticket plainly says it and the solver misread it: change nothing.
Then run
    {check}
until it prints `FAMILY CHECK: ALL PASS`. Finish with one line per miss: a, b or c, and why."""


def _catalog_text(business: Path) -> str:
    cat = read_json(business / "catalog.json")
    out = []
    for t, r in cat["tables"].items():
        rows = read_json(business / "tables" / f"{t}.json")[:3]
        notes = r.get("steward") or {}
        out.append(
            f"## {t}: {r.get('slice_rows', ['?'])[0]} rows per slice, key {r.get('primary_key')}, "
            f"points at {r.get('foreign_keys')}\ncolumns: {r.get('columns')}\n"
            + "".join(f"note: {n}\n" for n in (notes.get("rules", []) + notes.get("watch_outs", []))[:8])
            + "\n".join(json.dumps(x) for x in rows))
    return "\n\n".join(out)


def _modes_text(modes: list[dict]) -> str:
    return "\n".join(f"- {m['id']}: {m['mode'][:220]}" for m in modes)


def plan(business: Path, modes: list[dict], n: int) -> list[dict]:
    briefs = llm.ask_json(PLAN.format(n=n, catalog=_catalog_text(business), modes=_modes_text(modes)),
                          CONFIG.author_model, timeout=1200)
    return briefs[:n]


def _check_cmd(work: Path) -> str:
    return f"uv run --project {ROOT} python -m rsi.family_check family.py data --modes modes.json"


def _workdir(run_dir: Path, brief: dict, modes: list[dict]) -> Path:
    work = run_dir / "families" / brief["id"]
    work.mkdir(parents=True, exist_ok=True)
    if not (work / "data").exists():
        shutil.copytree(run_dir / "business", work / "data")
    shutil.copy(F.BUILTIN / "shoe_store.py", work / "example_family.py")
    (work / "CONTRACT.md").write_text(F.__doc__)
    write_json(work / "modes.json", modes)
    write_json(work / "brief.json", brief)
    return work


def write_family(run_dir: Path, brief: dict, modes: list[dict]) -> Path:
    work = _workdir(run_dir, brief, modes)
    llm.ask(WRITE.format(brief=json.dumps(brief, indent=1), modes=_modes_text(modes), check=_check_cmd(work)),
            CONFIG.author_model, tools=TOOLS, cwd=work, timeout=2 * 3600)
    return work / "family.py"


def prove(fam: F.Family, dbs: list[dict], seed: int = 0, tries: int = 2) -> list[dict]:
    """Independent solves: the full stack and two half stacks, by a model that is not the author."""
    rng = random.Random(f"{seed}-{fam.id}")
    choices = fam.params_choices(dbs[0])
    half = len(fam.rule_ids) // 2
    stacks = [tuple(fam.rule_ids)] + [tuple(sorted(rng.sample(fam.rule_ids, half), key=fam.rule_ids.index)) for _ in range(2)]
    out = []
    for stack in stacks:
        task = F.Task(fam.id, stack, rng.choice(choices))
        res = {"task": task.id, "level": task.level, "rules": list(stack), "params": task.params,
               "model": CONFIG.check_model, "solved": False, "tries": 0}
        for _ in range(tries):
            answer = llm.ask(F.prompt(fam, task, dbs[0]), CONFIG.check_model, timeout=1200)
            g = F.score(fam, task, answer, dbs)
            res["tries"] += 1
            res.update(solved=g["solved"], key_share=g["key_share"], error=g["error"], diff=g["diff"])
            if g["solved"]:
                break
            res["answer"] = F.extract_code(answer)
        out.append(res)
    return out


def author_one(run_dir: Path, brief: dict, modes: list[dict], agent: str) -> dict:
    stage = "author"
    mode_ids = {m["id"] for m in modes}
    dbs = F.load_dbs(run_dir / "business")
    path = write_family(run_dir, brief, modes)
    result = {"id": brief["id"], "path": str(path.relative_to(run_dir)), "accepted": False, "author_model": CONFIG.author_model}
    if not path.exists():
        result["why"] = "the author wrote no family.py"
        return result
    for rnd in range(2):
        check = family_check.check(path, run_dir / "business", mode_ids)
        result["check"] = check
        if not all(c["ok"] for c in check):
            result["why"] = "mechanical check failed: " + "; ".join(c["detail"] for c in check if not c["ok"])[:300]
            events.emit("work", stage=stage, agent=agent, role="checker", msg=f"{brief['id']}: check FAILED")
            return result
        fam = F.load(path)
        result["id"] = fam.id
        events.emit("work", stage=stage, agent=agent, role="checker",
                    msg=f"{fam.id}: mechanical check passed ({len(fam.rules)} rules)")
        solves = prove(fam, dbs, seed=rnd)
        result.setdefault("independent_solves", []).append([{k: v for k, v in s.items() if k != "answer"} for s in solves])
        for s in solves:
            events.emit("work", stage=stage, agent=agent, role="prover",
                        msg=f"{fam.id} level {s['level']}: {CONFIG.check_model} "
                            f"{'solved it from the ticket alone' if s['solved'] else 'missed (' + str(s.get('key_share')) + ' of keys right)'}")
        misses = [s for s in solves if not s["solved"]]
        if not misses or rnd == 1:
            break
        text = "\n\n".join(
            f"### Miss: rules on = {m['rules']}, params = {m['params']}\nTICKET THEY SAW:\n"
            f"{F.ticket(fam, F.Task(fam.id, tuple(m['rules']), m['params']))}\n\nTHEIR CODE:\n```python\n{m.get('answer', '')[:6000]}\n```\n"
            f"ERROR: {m.get('error') or 'none'}\nDIFFERENCES: {json.dumps(m.get('diff'))}" for m in misses)
        events.emit("work", stage=stage, agent=agent, role="author", msg=f"{fam.id}: {len(misses)} miss(es) back to the author")
        llm.ask(REVISE.format(misses=text, check=_check_cmd(path.parent)), CONFIG.author_model,
                tools=TOOLS, cwd=path.parent, timeout=3600)
    last = result["independent_solves"][-1]
    full = next(s for s in last if s["level"] == len(F.load(path).rules))
    result["accepted"] = bool(full["solved"])
    result["why"] = (f"checker green; {sum(s['solved'] for s in last)}/{len(last)} independent solves, full stack "
                     + ("solved" if full["solved"] else "NOT solved"))
    result["_brief"] = f"{result['id']}: {'accepted' if result['accepted'] else 'rejected'} ({result['why']})"
    return result


def run(run_dir: Path, n: int = 3) -> list[dict]:
    events.emit("stage_start", stage="author", msg=f"plan {n} task families for this dataset, write them, prove them")
    modes = read_json(run_dir / "failure_modes.json")
    briefs = plan(run_dir / "business", modes, n)
    write_json(run_dir / "family_briefs.json", briefs)
    for b in briefs:
        events.emit("work", stage="author", agent="planner", role="planner", msg=f"{b['id']}: {b['question']}")
    results = events.swarm("author", "author", briefs, lambda b, agent: author_one(run_dir, b, modes, agent),
                           workers=n, label=lambda b: b["id"])
    listing = [{k: v for k, v in r.items() if k != "_brief"} for r in results if r]
    write_json(run_dir / "families.json", listing)
    ok = sum(r["accepted"] for r in listing)
    events.emit("stage_end", stage="author", msg=f"{ok}/{len(briefs)} families accepted", accepted=ok)
    print(f"author: {ok}/{len(briefs)} families accepted -> {run_dir / 'families.json'}")
    for r in listing:
        print(f"  {r['id']}: {'ACCEPTED' if r['accepted'] else 'rejected'}: {r.get('why', '')}")
    return listing
