"""Task families: one business question over a dataset, with switchable rules and an exact oracle.

A family is a Python file (written by an author agent for a new dataset, or shipped in
rsi/families/) with this contract:

    FAMILY = {"id": "kebab-case", "background": "2-3 sentences: the job and what to compute",
              "pins": ["what the keys and values of the answer are", ...],
              "tables": ["orders", ...]}             # tables the solver is shown
    RULES = [{"id": "snake_case", "mode": "<failure mode id>",
              "present": "the ticket sentence when the rule is on",
              "absent": "the explicit default sentence when it is off"}, ...]
    def params_choices(db) -> list[dict]             # job parameters taken from the data
    def reference(db, params, rules) -> dict[str, int]   # the oracle; `rules` = ids that are on

A task switches L of the rules on (its level). Every rule that is off is replaced by its
default sentence, so the ticket always says exactly what to compute. Level R (all rules)
is the "impossible" start of the anneal; level 0 is the plain job.

Grading is all-or-nothing against the oracle on every slice of the business data: the
slice the solver was shown rows from and the hidden twin slices.

`export_harbor` writes a task in the docs/TASK_CONTRACT.md shape: reference = the family
file with this task's rules on, one mutation per rule = the same with that rule off (the
build that forgot it), hidden tests = one per answer key per slice. `rsi gate` runs on it.
"""

from __future__ import annotations

import difflib
import hashlib
import importlib.util
import itertools
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

BUILTIN = Path(__file__).resolve().parent / "families"
ENTRY = "job.py"  # the file a Harbor solver edits


@dataclass
class Family:
    path: Path
    mod: object
    id: str
    background: str
    pins: list[str]
    tables: list[str]
    rules: list[dict]

    @property
    def rule_ids(self) -> list[str]:
        return [r["id"] for r in self.rules]

    @property
    def levels(self) -> list[int]:
        return list(range(len(self.rules), -1, -1))

    def mode_of(self, rule_id: str) -> str:
        return next(r["mode"] for r in self.rules if r["id"] == rule_id)

    def reference(self, db: dict, params: dict, rules) -> dict:
        out = self.mod.reference(db, dict(params), set(rules))
        return {str(k): v for k, v in sorted(out.items(), key=lambda kv: str(kv[0]))}

    def params_choices(self, db: dict) -> list[dict]:
        return list(self.mod.params_choices(db))

    @property
    def source(self) -> str:
        return self.path.read_text()


def load(path: Path) -> Family:
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"rsi_family_{hashlib.sha256(str(path).encode()).hexdigest()[:8]}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    f = mod.FAMILY
    return Family(path, mod, f["id"], f["background"].strip(), list(f.get("pins", [])), list(f["tables"]), list(mod.RULES))


def load_run(run_dir: Path) -> list[Family]:
    """The families accepted for a run (runs/<run>/families.json), in order."""
    listing = json.loads((run_dir / "families.json").read_text())
    return [load(Path(f["path"]) if Path(f["path"]).is_absolute() else run_dir / f["path"]) for f in listing if f.get("accepted", True)]


def load_dbs(business_dir: Path) -> list[dict]:
    """Slice 0 first, then the twin slices."""
    def one(d: Path) -> dict:
        return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))}
    twins = sorted((business_dir / "twins").glob("*")) if (business_dir / "twins").exists() else []
    return [one(business_dir / "tables")] + [one(t) for t in twins]


# ---------------------------------------------------------------- tasks and tickets

@dataclass
class Task:
    family: str
    rules: tuple[str, ...]
    params: dict = field(default_factory=dict)

    @property
    def level(self) -> int:
        return len(self.rules)

    @property
    def id(self) -> str:
        key = self.family + "|" + ",".join(self.rules) + "|" + json.dumps(self.params, sort_keys=True)
        return f"{self.family}-L{self.level}-" + hashlib.sha256(key.encode()).hexdigest()[:8]

    def as_dict(self, fam: Family) -> dict:
        return {"id": self.id, "family": self.family, "level": self.level, "rules": list(self.rules),
                "modes": [fam.mode_of(r) for r in self.rules], "params": self.params}


def ticket(fam: Family, task: Task, harbor: bool = False) -> str:
    """The ticket. `harbor` words the signature pin for a solver working in the task's container."""
    rules = "\n".join(f"- {r['present'] if r['id'] in task.rules else r['absent']}" for r in fam.rules)
    sig = (f"`def solve(db, params) -> dict` in `/app/{ENTRY}`; `db` maps table name (JSON files in `/app/data`) "
           f"to row dicts; `params` here is `{json.dumps(task.params)}`." if harbor else
           "Signature: `def solve(db, params) -> dict`. `db` maps table name to a list of row dicts; "
           f"`params` is a dict, e.g. `{json.dumps(task.params)}`.")
    pins = "\n".join(f"- {p}" for p in [sig, *fam.pins, "Standard library only."])
    return f"{fam.background}\n\nRules:\n{rules}\n\nPins:\n{pins}"


def schema_brief(fam: Family, db: dict, n: int = 4) -> str:
    out = []
    for name in fam.tables:
        rows = db[name]
        cat = [c for c in rows[0] if len({str(r[c]) for r in rows[:400]}) <= 12]
        picks, seen = [rows[0]], {(c, str(rows[0][c])) for c in cat}
        for r in rows[1:400]:  # rows that show a value not yet seen in a categorical column
            new = {(c, str(r[c])) for c in cat} - seen
            if new and len(picks) < n:
                picks.append(r)
                seen |= new
        out.append(f"{name} ({len(rows)} rows), columns {list(rows[0])}:\n" + "\n".join(json.dumps(r) for r in picks))
    return "\n\n".join(out)


def prompt(fam: Family, task: Task, db: dict) -> str:
    return (f"{ticket(fam, task)}\n\nThe data (sample rows; the grader runs your function on the full tables and on "
            f"other slices of the same business's data):\n\n{schema_brief(fam, db)}\n\nparams for this job: "
            f"{json.dumps(task.params)}\n\nReply with the complete function in one ```python block.")


def split_combos(fam: Family, seed: int = 0, bench_share: float = 0.34) -> dict[int, dict[str, list[tuple[str, ...]]]]:
    """Per level, rule combinations held out for the benchmark vs open to training. A level with
    one combination (0 and R) is benchmark-only, so training never sees a benchmark ticket."""
    out = {}
    for lvl in fam.levels:
        cs = list(itertools.combinations(fam.rule_ids, lvl))
        random.Random(f"{seed}-{fam.id}-{lvl}").shuffle(cs)
        k = len(cs) if len(cs) == 1 else max(1, round(len(cs) * bench_share))
        out[lvl] = {"bench": cs[:k], "train": cs[k:]}
    return out


def make_tasks(fam: Family, level: int, n: int, pool: str, seed: int, db: dict) -> list[Task]:
    """n distinct tasks at a level from the bench or train pool of rule combinations."""
    rng = random.Random(f"{seed}-{fam.id}-{level}-{pool}")
    cs = split_combos(fam)[level][pool]
    choices = fam.params_choices(db)
    if not cs:
        return []
    by_combo = {}
    for c in cs:
        ps = list(choices)
        rng.shuffle(ps)
        by_combo[c] = ps
    order = list(cs)
    rng.shuffle(order)
    tasks = []
    for i in range(len(choices)):  # round-robin over combinations, a new params each pass
        for c in order:
            if len(tasks) < n:
                tasks.append(Task(fam.id, tuple(c), by_combo[c][i]))
    return tasks


# ---------------------------------------------------------------- grading

HARNESS = """
import json, sys
{code}
_db = json.load(open(sys.argv[1]))
_params = json.load(open(sys.argv[2]))
_out = solve(_db, _params)
print("\\n" + json.dumps({{str(k): v for k, v in (_out or {{}}).items()}}))
"""
HARNESS_LINES = HARNESS[:HARNESS.index("{code}")].count("\n")  # lines the harness puts above the candidate
_DB_FILES: dict[int, Path] = {}
_TMP = None
_SANDBOX: list[str] | None = None


def _db_file(db: dict) -> Path:
    global _TMP
    if id(db) not in _DB_FILES:
        _TMP = _TMP or Path(tempfile.mkdtemp(prefix="rsi-dbs-"))
        p = _TMP / f"db{len(_DB_FILES)}.json"
        p.write_text(json.dumps(db))
        _DB_FILES[id(db)] = p
    return _DB_FILES[id(db)]


def _sandbox() -> list[str]:
    """Candidate code is model-written: no network, no writes outside temp, where macOS allows it."""
    global _SANDBOX
    if _SANDBOX is None:
        profile = '(version 1)(allow default)(deny network*)(deny file-write* (subpath "/Users"))'
        ok = shutil.which("sandbox-exec") and subprocess.run(
            ["sandbox-exec", "-p", profile, "/usr/bin/true"], capture_output=True).returncode == 0
        _SANDBOX = ["sandbox-exec", "-p", profile] if ok else []
    return _SANDBOX


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len)
    cut = re.search(r"```(?:python|py)?\s*\n(.*)", text, re.S)  # a block the token limit cut off
    return cut.group(1) if cut else text


def _run(code: str, db: dict, params: dict, timeout: int = 30) -> tuple[dict | None, str, str]:
    """(answer, one-line error, traceback tail). Line numbers in the tail are the candidate's own."""
    with tempfile.TemporaryDirectory(prefix="rsi-cand-") as tmp:
        tmp = Path(tmp)
        (tmp / "params.json").write_text(json.dumps(params))
        (tmp / "cand.py").write_text(HARNESS.format(code=code))
        try:
            p = subprocess.run([*_sandbox(), sys.executable, "-I", str(tmp / "cand.py"), str(_db_file(db)), str(tmp / "params.json")],
                               capture_output=True, text=True, timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired:
            return None, "timeout", f"the function did not finish in {timeout} s"
        if p.returncode != 0:
            lines = p.stderr.strip().splitlines() or [f"exit {p.returncode}"]
            own = [i for i, line in enumerate(lines) if "cand.py" in line]
            tail = "\n".join(lines[own[-1]:] if own else lines[-1:])
            tail = re.sub(r'File "[^"]*cand\.py", line (\d+)', lambda m: f"line {int(m.group(1)) - HARNESS_LINES}", tail)
            return None, lines[-1][:300], tail[-700:]
        try:
            out = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            return None, "no JSON output", "solve() did not return a dict that json.dumps accepts"
        return (out, "", "") if isinstance(out, dict) else (None, "answer is not a dict", "solve() must return a dict")


def run_candidate(code: str, db: dict, params: dict, timeout: int = 30) -> tuple[dict | None, str]:
    return _run(code, db, params, timeout)[:2]


def crash_on_shown(task: Task, answer_text: str, db: dict) -> str:
    """What running an answer on the slice its author was shown raises; '' if it runs.

    This is all a repair turn is told. It is feedback an agent can get by running its own code
    on the data it was given, and it says nothing about what the right answer is."""
    return _run(extract_code(answer_text), db, task.params)[2]


def score(fam: Family, task: Task, answer_text: str, dbs: list[dict]) -> dict:
    code = extract_code(answer_text)
    right = total = 0
    solved, err, first_diff = True, "", None
    for i, db in enumerate(dbs):
        want = fam.reference(db, task.params, task.rules)
        got, e = run_candidate(code, db, task.params)
        keys = sorted(set(want) | set(got or {}))
        total += len(keys)
        if got is None:
            solved, err = False, err or e
            continue
        bad = [k for k in keys if got.get(k) != want.get(k)]
        right += len(keys) - len(bad)
        if bad:
            solved = False
            first_diff = first_diff or {"slice": i, "wrong_keys": len(bad), "of": len(keys),
                                        "examples": [{"key": k, "want": want.get(k), "got": got.get(k)} for k in bad[:5]]}
    return {"solved": solved, "key_share": round(right / total, 3) if total else 0.0, "error": err, "diff": first_diff}


# ---------------------------------------------------------------- Harbor export

def effect(fam: Family, dbs: list[dict], params: dict, rules: tuple[str, ...]) -> dict[str, int]:
    """Per rule that is on: how many answer keys change, over all slices, when it is forgotten."""
    full = [fam.reference(db, params, rules) for db in dbs]
    out = {}
    for rid in rules:
        wrong = tuple(r for r in rules if r != rid)
        out[rid] = sum(
            sum(1 for k in set(f) | set(m) if f.get(k) != m.get(k))
            for f, m in zip(full, (fam.reference(db, params, wrong) for db in dbs))
        )
    return out


def export_harbor(fam: Family, task: Task, dbs: list[dict], out_root: Path) -> Path:
    """Write the task as a Harbor task directory (docs/TASK_CONTRACT.md)."""
    names = ["main"] + [f"twin{i}" for i in range(1, len(dbs))]
    d = out_root / task.id.lower()
    for sub in ("environment/app/data", "tests/data", "solution", "authoring/mutations"):
        (d / sub).mkdir(parents=True, exist_ok=True)

    (d / "task.toml").write_text(textwrap.dedent(f"""\
        schema_version = "1.1"

        [task]
        name = "rsi-rl-dev/{task.id.lower()}"
        description = "{fam.id}: rules {', '.join(task.rules) or 'none'}."
        keywords = ["rsi", "{fam.id}"]

        [metadata]
        difficulty = "level-{task.level}"
        category = "programming"
        tags = {json.dumps(sorted({fam.mode_of(r) for r in task.rules}))}

        [verifier]
        timeout_sec = 300.0

        [agent]
        timeout_sec = 1800.0

        [environment]
        build_timeout_sec = 600.0
        cpus = 1
        memory_mb = 1024
        storage_mb = 4096
        gpus = 0
        allow_internet = true
        mcp_servers = []

        [verifier.env]

        [environment.env]

        [solution.env]
        """))
    (d / "instruction.md").write_text(ticket(fam, task, harbor=True) + "\n")
    for t in fam.tables:
        (d / "environment/app/data" / f"{t}.json").write_text(json.dumps(dbs[0][t]))
    stub = '"""The job."""\n\n\ndef solve(db, params):\n    raise NotImplementedError\n'
    (d / "environment/app" / ENTRY).write_text(stub)
    (d / "environment/Dockerfile").write_text(textwrap.dedent("""\
        FROM python:3.12-slim
        RUN apt-get update && apt-get install -y --no-install-recommends git tzdata && rm -rf /var/lib/apt/lists/* \\
         && pip install --no-cache-dir pytest==8.4.1 pytest-json-ctrf==0.3.5
        WORKDIR /app
        COPY app/ /app/
        RUN git init -q && git -c user.email=t@t -c user.name=t add -A \\
         && git -c user.email=t@t -c user.name=t commit -qm base
        """))

    def body(rules) -> str:
        return fam.source.rstrip() + f"\n\n\ndef solve(db, params):\n    return reference(db, params, set({sorted(rules)!r}))\n"

    ref_body = body(task.rules)
    (d / "solution/reference.patch").write_text(_diff(ENTRY, stub, ref_body))
    (d / "solution/solve.sh").write_text("#!/bin/bash\nset -e\ngit -C /app apply --whitespace=nowarn /solution/reference.patch\n")

    expected = {n: fam.reference(db, task.params, task.rules) for n, db in zip(names, dbs)}
    cases = {n: set(expected[n]) for n in names}
    criteria = []
    for i, rid in enumerate(task.rules, 1):
        wrong = tuple(r for r in task.rules if r != rid)
        tests = []
        for n, db in zip(names, dbs):
            mut = fam.reference(db, task.params, wrong)
            changed = sorted(k for k in set(mut) | set(expected[n]) if mut.get(k) != expected[n].get(k))
            cases[n] |= set(changed)
            tests += [f"test_key[{n}-{k}]" for k in changed]
        rule = next(r for r in fam.rules if r["id"] == rid)
        criteria.append({"id": f"AC{i}", "rule": rid, "mode": rule["mode"], "sentence": rule["present"], "tests": tests})
        (d / "authoring/mutations" / f"{i:02d}-AC{i}-forgets-{rid}.patch").write_text(_diff(ENTRY, ref_body, body(wrong)))
    for n, db in zip(names, dbs):
        (d / "tests/data" / n).mkdir(exist_ok=True)
        for t in fam.tables:
            (d / "tests/data" / n / f"{t}.json").write_text(json.dumps(db[t]))
    (d / "tests/expected.json").write_text(json.dumps(
        {"params": task.params, "expected": expected, "cases": {n: sorted(v) for n, v in cases.items()}}))
    (d / "tests/test_job.py").write_text(TEST_FILE)
    (d / "tests/test.sh").write_text(textwrap.dedent("""\
        #!/bin/bash
        mkdir -p /logs/verifier
        cd /tests
        if pytest --ctrf /logs/verifier/ctrf.json /tests/test_job.py -q -p no:cacheprovider; then
          echo 1 > /logs/verifier/reward.txt
        else
          echo 0 > /logs/verifier/reward.txt
        fi
        """))
    (d / "authoring/criteria.json").write_text(json.dumps(criteria, indent=1))
    (d / "authoring/design.json").write_text(json.dumps({**task.as_dict(fam), "slug": d.name, "source": "rsi.family"}, indent=1))
    (d / "authoring/selfcheck.sh").write_text("#!/bin/bash\n# family tasks are checked by rsi gate (oracle, nop, mutations)\nexit 0\n")
    return d


def _diff(name: str, old: str, new: str) -> str:
    body = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{name}", f"b/{name}"))
    return f"diff --git a/{name} b/{name}\n{body}"


TEST_FILE = '''import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "/app")

HERE = Path(__file__).parent
SPEC = json.loads((HERE / "expected.json").read_text())
DBS = {d.name: {p.stem: json.loads(p.read_text()) for p in d.glob("*.json")} for d in (HERE / "data").iterdir()}
CASES = [(n, k) for n in sorted(SPEC["cases"]) for k in SPEC["cases"][n]]
_cache = {}


def answer(n):
    if n not in _cache:
        from job import solve
        _cache[n] = {str(a): b for a, b in (solve(DBS[n], dict(SPEC["params"])) or {}).items()}
    return _cache[n]


@pytest.mark.parametrize("n,k", CASES, ids=[f"{n}-{k}" for n, k in CASES])
def test_key(n, k):
    assert answer(n).get(k) == SPEC["expected"][n].get(k)


@pytest.mark.parametrize("n", sorted(DBS))
def test_no_extra_keys(n):
    assert set(answer(n)) == set(SPEC["expected"][n])
'''
