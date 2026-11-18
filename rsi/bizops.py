"""Tasks made from the pruned business data: stacked rules, exact oracles, levels 1 to 8.

Each task asks for one function, `solve(db, params)`, that computes the refund owed per
order from the returns table. A task stacks L of the eight rules below; every rule is
one documented failure mode (the ids match seeds/failure_modes.json and the scan), and
every rule the task leaves out is replaced by an explicit default sentence, so the
ticket always says exactly what to compute. Level 8 (all rules) is the "impossible"
start of the anneal; level 1 is one rule.

Grading is all-or-nothing over the real pruned dataset plus twin datasets (the same
generator, other seeds, run through the same intake), so a solution that fits the
sample rows by luck fails. `score` also reports the share of orders right, for analysis.

`export_harbor` writes any task as a Harbor task in the docs/TASK_CONTRACT.md shape:
reference = the oracle, one mutation per rule = the oracle with that rule swapped for
its default (the textbook build that forgot it), hidden tests = one per order per
dataset, mapped to the rule whose mutation changes that order. So `rsi gate` (oracle,
nop, mutations) runs on it unchanged.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import subprocess
import sys
import tempfile
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from . import business_gen, intake

# id, failure mode it stacks, the ticket sentence when present, the default when absent
RULES = [
    ("in_store", "sibling-path-missed",
     "In-store returns follow every rule exactly as mail returns do. Their request time is the store's local wall-clock time, in requested_local; mail returns carry UTC in requested_at.",
     "Only mail returns (channel mail) count; ignore in-store returns."),
    ("window", "timezone-math",
     "A return is accepted only within 30 days of the order, counted in whole calendar days in the store's own time zone: day 30 is accepted, day 31 is not.",
     "There is no time limit on returns."),
    ("final_sale", "existing-rule-carry",
     "A product marked final_sale is returnable only when the reason is defective; any other final-sale return is rejected.",
     "Final-sale products are returnable like any other."),
    ("dedupe", "idempotency-under-failure",
     "The webhook sometimes delivers a return twice: rows sharing an event_id are one return and count once.",
     "Every row is its own return, even when event_ids repeat."),
    ("qty_cap", "cross-step-state",
     "Process returns in request-time order (ties by return_id). A return that would bring its line's accepted quantity above the quantity bought is rejected whole; rejected returns use up nothing.",
     "There is no limit on returned quantity."),
    ("discount", "derived-surface-omitted",
     "Refund the net price: unit_price_cents * qty minus the line discount's share, discount_cents * qty // line qty (integer division).",
     "Refund unit_price_cents * qty; ignore discounts."),
    ("exchange", "single-manifestation-fix",
     "Exchanges (kind exchange) refund nothing, yet must pass every other rule and use up returnable quantity like a refund.",
     "There is no special case for exchanges: rows whose kind is exchange are refunded like any return."),
    ("tenant", "authz-by-id",
     "Report only orders placed at store params['store_id']; another store's returns never appear, whatever line_id they cite.",
     "Report orders from every store."),
]
RULE_IDS = [r[0] for r in RULES]
MODE_OF = {r[0]: r[1] for r in RULES}
LEVELS = list(range(len(RULES), 0, -1))  # 8 (impossible) down to 1

BACKGROUND = """You maintain the refunds job for a shoe retailer with stores in several time zones.
Each returns row is one return request for an order line. Compute the refund owed per order."""

PINS = """Pins:
- Signature: `def solve(db, params) -> dict`. `db` maps table name to a list of row dicts;
  `params` is a dict, e.g. `{{"store_id": "S02"}}`.
- Return `{{order_id: refund_cents}}` with an int for every order whose refund is above 0, and
  no other keys.
- Order time is `orders.placed_at` (UTC, ISO 8601 with Z); a store's IANA zone is `stores.timezone`.
- Standard library only."""


@dataclass
class Task:
    rules: tuple[str, ...]
    params: dict = field(default_factory=dict)

    @property
    def level(self) -> int:
        return len(self.rules)

    @property
    def id(self) -> str:
        key = ",".join(self.rules) + "|" + json.dumps(self.params, sort_keys=True)
        return f"L{self.level}-" + hashlib.sha256(key.encode()).hexdigest()[:8]

    @property
    def modes(self) -> list[str]:
        return [MODE_OF[r] for r in self.rules]

    def as_dict(self) -> dict:
        return {"id": self.id, "level": self.level, "rules": list(self.rules), "modes": self.modes, "params": self.params}


def ticket(task: Task) -> str:
    lines = []
    for rid, _, present, absent in RULES:
        lines.append(f"- {present if rid in task.rules else absent}")
    return f"{BACKGROUND}\n\nRules:\n" + "\n".join(lines) + "\n\n" + PINS.format()


# ---------------------------------------------------------------- the oracle

def _utc(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def reference(db: dict, params: dict, rules: tuple[str, ...] | list[str]) -> dict:
    rules = set(rules)
    stores = {s["store_id"]: ZoneInfo(s["timezone"]) for s in db["stores"]}
    orders = {o["order_id"]: o for o in db["orders"]}
    lines = {ln["line_id"]: ln for ln in db["order_lines"]}
    final = {p["sku"]: p["final_sale"] in (True, "true") for p in db["products"]}

    rows = []
    for r in db["returns"]:
        line = lines.get(r["line_id"])
        if line is None or line["order_id"] not in orders:
            continue
        order = orders[line["order_id"]]
        tz = stores[order["store_id"]]
        if r["channel"] == "store":
            if "in_store" not in rules:
                continue
            req = datetime.strptime(r["requested_local"], "%Y-%m-%d %H:%M").replace(tzinfo=tz)
        else:
            req = _utc(r["requested_at"])
        if "tenant" in rules and order["store_id"] != params.get("store_id"):
            continue
        rows.append((req.astimezone(timezone.utc), r["return_id"], r, line, order, tz))
    rows.sort(key=lambda x: (x[0], x[1]))

    seen_events, used, refund = set(), defaultdict(int), defaultdict(int)
    for req, _, r, line, order, tz in rows:
        if "dedupe" in rules:
            if r["event_id"] in seen_events:
                continue
            seen_events.add(r["event_id"])
        if "window" in rules:
            placed = _utc(order["placed_at"]).astimezone(tz).date()
            if (req.astimezone(tz).date() - placed).days > 30:
                continue
        if "final_sale" in rules and final[line["sku"]] and r["reason"] != "defective":
            continue
        qty = int(r["qty"])
        if "qty_cap" in rules:
            if used[line["line_id"]] + qty > int(line["qty"]):
                continue
            used[line["line_id"]] += qty
        if r["kind"] == "exchange" and "exchange" in rules:
            continue
        amount = int(line["unit_price_cents"]) * qty
        if "discount" in rules:
            amount -= int(line["discount_cents"]) * qty // int(line["qty"])
        refund[order["order_id"]] += amount
    return {k: v for k, v in sorted(refund.items()) if v > 0}


# ---------------------------------------------------------------- datasets

def load_db(business_dir: Path) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted((business_dir / "tables").glob("*.json"))}


def twin_dbs(seeds=(11, 23)) -> list[dict]:
    """Same generator, other seeds, same pruning: hidden datasets the solver never sees."""
    out = []
    for s in seeds:
        raw = {k: [{c: str(v) for c, v in r.items()} for r in rows] for k, rows in business_gen.generate(s).items()}
        out.append(intake.prune(raw)[0])
    return out


def schema_brief(db: dict, n: int = 3) -> str:
    out = []
    for name in ("stores", "products", "orders", "order_lines", "returns"):
        rows = db[name]
        picks = rows[:n]
        if name == "returns":  # show both channels and both kinds
            picks = ([r for r in rows if r["channel"] == "mail"][:2] + [r for r in rows if r["channel"] == "store"][:1]
                     + [r for r in rows if r["kind"] == "exchange"][:1])
        out.append(f"{name} ({len(rows)} rows), columns {list(rows[0])}:\n" + "\n".join(json.dumps(r) for r in picks))
    return "\n\n".join(out)


def prompt(task: Task, db: dict) -> str:
    return (f"{ticket(task)}\n\nThe data (sample rows; the grader runs your function on the full tables and on "
            f"other datasets with the same schema):\n\n{schema_brief(db)}\n\nparams for this job: "
            f"{json.dumps(task.params)}\n\nReply with the complete function in one ```python block.")


# ---------------------------------------------------------------- task sets

def combos(level: int) -> list[tuple[str, ...]]:
    return list(itertools.combinations(RULE_IDS, level))


def split_combos(seed: int = 0, bench_share: float = 0.34) -> dict[int, dict[str, list[tuple[str, ...]]]]:
    """Per level, rule combinations held out for the benchmark vs open to training. A level
    with one combination (8) is benchmark-only, so training never sees a benchmark ticket."""
    rng = random.Random(seed)
    out = {}
    for lvl in LEVELS:
        cs = combos(lvl)
        rng.shuffle(cs)
        k = len(cs) if len(cs) == 1 else max(1, round(len(cs) * bench_share))
        out[lvl] = {"bench": cs[:k], "train": cs[k:]}
    return out


def make_tasks(level: int, n: int, pool: str, seed: int, stores: list[str]) -> list[Task]:
    rng = random.Random(f"{seed}-{level}-{pool}")
    cs = split_combos()[level][pool] or split_combos()[level]["bench"]
    tasks, seen = [], set()
    for i in range(n * 4):
        rules = cs[i % len(cs)] if i < len(cs) else rng.choice(cs)
        t = Task(tuple(rules), {"store_id": rng.choice(stores)})
        if t.id not in seen:
            seen.add(t.id)
            tasks.append(t)
        if len(tasks) == n:
            break
    return tasks


# ---------------------------------------------------------------- grading

HARNESS = """
import json, sys
{code}
_db = json.load(open(sys.argv[1]))
_params = json.load(open(sys.argv[2]))
_out = solve(_db, _params)
print(json.dumps({{str(k): v for k, v in (_out or {{}}).items()}}))
"""


def extract_code(text: str) -> str:
    import re
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else text


def run_candidate(code: str, db: dict, params: dict, timeout: int = 20) -> tuple[dict | None, str]:
    with tempfile.TemporaryDirectory(prefix="rsi-biz-") as tmp:
        tmp = Path(tmp)
        (tmp / "db.json").write_text(json.dumps(db))
        (tmp / "params.json").write_text(json.dumps(params))
        (tmp / "cand.py").write_text(HARNESS.format(code=code))
        try:
            p = subprocess.run([sys.executable, "-I", str(tmp / "cand.py"), str(tmp / "db.json"), str(tmp / "params.json")],
                               capture_output=True, text=True, timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired:
            return None, "timeout"
        if p.returncode != 0:
            return None, p.stderr.strip().splitlines()[-1][:300] if p.stderr.strip() else f"exit {p.returncode}"
        try:
            return json.loads(p.stdout.strip().splitlines()[-1]), ""
        except Exception:
            return None, "no JSON output"


def score(task: Task, answer_text: str, dbs: list[dict]) -> dict:
    code = extract_code(answer_text)
    right = total = 0
    solved, err = True, ""
    for db in dbs:
        want = reference(db, task.params, task.rules)
        got, e = run_candidate(code, db, task.params)
        keys = set(want) | set(got or {})
        total += len(keys)
        if got is None:
            solved, err = False, err or e
            continue
        good = sum(1 for k in keys if got.get(k) == want.get(k))
        right += good
        if good != len(keys):
            solved = False
    return {"solved": solved, "order_share": round(right / total, 3) if total else 0.0, "error": err}


# ---------------------------------------------------------------- Harbor export

def export_harbor(task: Task, db: dict, out_root: Path, twins: list[dict] | None = None) -> Path:
    """Write the task as a Harbor task directory (docs/TASK_CONTRACT.md)."""
    twins = twins if twins is not None else twin_dbs()
    dbs = {"main": db, **{f"twin{i + 1}": t for i, t in enumerate(twins)}}
    d = out_root / f"bizops-{task.id.lower()}"
    for sub in ("environment/app/data", "tests/data", "solution", "authoring/mutations"):
        (d / sub).mkdir(parents=True, exist_ok=True)

    (d / "task.toml").write_text(textwrap.dedent(f"""\
        schema_version = "1.1"

        [task]
        name = "rsi-rl-dev/bizops-{task.id.lower()}"
        description = "Refunds per order over the pruned shoe-store export; rules {', '.join(task.rules)}."
        keywords = ["bizops", "rsi"]

        [metadata]
        difficulty = "level-{task.level}"
        category = "programming"
        tags = {json.dumps(task.modes)}

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
    (d / "instruction.md").write_text(
        ticket(task).replace("Standard library only.", "Standard library only. Write it in /app/refunds.py; "
                             f"/app/data holds the full tables as JSON. params for this job: {json.dumps(task.params)}.")
        + "\n")
    for name, rows in db.items():
        (d / "environment/app/data" / f"{name}.json").write_text(json.dumps(rows))
    (d / "environment/app/refunds.py").write_text('"""Refunds owed per order."""\n\n\ndef solve(db, params):\n    raise NotImplementedError\n')
    (d / "environment/Dockerfile").write_text(textwrap.dedent("""\
        FROM python:3.12-slim
        RUN apt-get update && apt-get install -y --no-install-recommends git tzdata && rm -rf /var/lib/apt/lists/* \\
         && pip install --no-cache-dir pytest==8.4.1 pytest-json-ctrf==0.3.5
        WORKDIR /app
        COPY app/ /app/
        RUN git init -q && git -c user.email=t@t -c user.name=t add -A \\
         && git -c user.email=t@t -c user.name=t commit -qm base
        """))

    # reference = the oracle with this task's rules baked in
    import inspect
    oracle_src = inspect.getsource(reference)
    ref_body = ("\"\"\"Refunds owed per order.\"\"\"\n\nfrom collections import defaultdict\n"
                "from datetime import datetime, timezone\nfrom zoneinfo import ZoneInfo\n\n"
                f"RULES = {tuple(task.rules)!r}\n\n\n" + inspect.getsource(_utc) + "\n\n" + oracle_src
                + "\n\ndef solve(db, params):\n    return reference(db, params, RULES)\n")
    base = (d / "environment/app/refunds.py").read_text()
    (d / "solution/reference.patch").write_text(_diff("refunds.py", base, ref_body))
    (d / "solution/solve.sh").write_text("#!/bin/bash\nset -e\ngit -C /app apply --whitespace=nowarn /solution/reference.patch\n")

    # hidden tests: one per (dataset, order) over the union of keys any mutation or the reference produces
    expected = {k: reference(v, task.params, task.rules) for k, v in dbs.items()}
    criteria, tests = [], []
    for i, rid in enumerate(task.rules, 1):
        wrong = tuple(r for r in task.rules if r != rid)
        names = []
        for k, v in dbs.items():
            mut = reference(v, task.params, wrong)
            for oid in sorted(set(mut) | set(expected[k])):
                if mut.get(oid) != expected[k].get(oid):
                    names.append(f"test_order[{k}-{oid}]")
        sentence = next(r[2] for r in RULES if r[0] == rid)
        criteria.append({"id": f"AC{i}", "rule": rid, "mode": MODE_OF[rid], "sentence": sentence, "tests": names})
        mut_body = ref_body.replace(f"RULES = {tuple(task.rules)!r}", f"RULES = {wrong!r}")
        (d / "authoring/mutations" / f"{i:02d}-AC{i}-forgets-{rid}.patch").write_text(_diff("refunds.py", ref_body, mut_body))
    for k in dbs:
        (d / "tests/data" / f"{k}").mkdir(exist_ok=True)
        for name, rows in dbs[k].items():
            (d / "tests/data" / k / f"{name}.json").write_text(json.dumps(rows))
        for oid in sorted(set(expected[k]) | {o["order_id"] for o in dbs[k]["orders"]}):
            tests.append((k, oid))
    (d / "tests/expected.json").write_text(json.dumps({"params": task.params, "expected": expected}))
    (d / "tests/test_refunds.py").write_text(TEST_FILE)
    (d / "tests/test.sh").write_text(textwrap.dedent("""\
        #!/bin/bash
        mkdir -p /logs/verifier
        cd /tests
        if pytest --ctrf /logs/verifier/ctrf.json /tests/test_refunds.py -q -p no:cacheprovider; then
          echo 1 > /logs/verifier/reward.txt
        else
          echo 0 > /logs/verifier/reward.txt
        fi
        """))
    (d / "authoring/criteria.json").write_text(json.dumps(criteria, indent=1))
    (d / "authoring/design.json").write_text(json.dumps({**task.as_dict(), "slug": d.name, "source": "rsi.bizops"}, indent=1))
    (d / "authoring/selfcheck.sh").write_text("#!/bin/bash\n# bizops tasks are checked by rsi gate (oracle, nop, mutations)\nexit 0\n")
    return d


def _diff(name: str, old: str, new: str) -> str:
    import difflib
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
CASES = [(k, o["order_id"]) for k in sorted(DBS) for o in DBS[k]["orders"]]
_cache = {}


def answer(k):
    if k not in _cache:
        from refunds import solve
        _cache[k] = {str(a): b for a, b in (solve(DBS[k], dict(SPEC["params"])) or {}).items()}
    return _cache[k]


@pytest.mark.parametrize("k,oid", CASES, ids=[f"{k}-{o}" for k, o in CASES])
def test_order(k, oid):
    assert answer(k).get(oid) == SPEC["expected"][k].get(oid)


@pytest.mark.parametrize("k", sorted(DBS))
def test_no_extra_keys(k):
    assert set(answer(k)) == set(SPEC["expected"][k])
'''
