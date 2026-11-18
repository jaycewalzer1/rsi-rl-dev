"""The mechanical checker a task family must pass before anything is built on it.

    python -m rsi.family_check FAMILY.py BUSINESS_DIR [--modes failure_modes.json] [--json]

Nothing here trusts the author. Every check runs the family's own oracle on every slice
of the real data:

  contract     FAMILY / RULES / params_choices / reference exist with the right shapes;
               6 to 8 rules with unique ids; every rule's mode is a documented failure mode
  stdlib       the file imports the standard library only
  params       at least 5 parameter choices that give at least 5 different answers
  answers      {str: int}, at least 15 keys on every slice, same output twice, under 2 s
  base         with every rule off the answer is still non-trivial (the plain job is a job)
  rule effect  every rule changes the answer when switched on alone AND when dropped from the
               full stack, on every slice (for at least one params), and on at least half of
               all (slice, params) pairs; a rule that barely touches real data teaches nothing
  distinct     no two rules change exactly the same keys in the same way (they are independent)
  ticket       150 to 300 words with all rules on and with all rules off
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

from . import family as F


def check(path: Path, business: Path, modes: set[str] | None = None, max_params: int = 6) -> list[dict]:
    res: list[dict] = []

    def add(name, ok, detail=""):
        res.append({"check": name, "ok": bool(ok), "detail": detail})
        return ok

    try:
        fam = F.load(path)
    except Exception as e:
        add("contract", False, f"cannot load: {type(e).__name__}: {e}")
        return res
    dbs = F.load_dbs(business)

    problems = []
    ids = fam.rule_ids
    if not 6 <= len(ids) <= 8:
        problems.append(f"{len(ids)} rules (want 6 to 8)")
    if len(set(ids)) != len(ids):
        problems.append("rule ids repeat")
    for r in fam.rules:
        for k in ("id", "mode", "present", "absent"):
            if not str(r.get(k, "")).strip():
                problems.append(f"rule {r.get('id')}: empty {k}")
        if modes and r.get("mode") not in modes:
            problems.append(f"rule {r.get('id')}: mode {r.get('mode')!r} is not a documented failure mode")
    if modes and len({r["mode"] for r in fam.rules}) < min(5, len(ids)):
        problems.append("fewer than 5 different failure modes across the rules")
    missing = [t for t in fam.tables if t not in dbs[0]]
    if missing or not fam.tables:
        problems.append(f"FAMILY['tables'] names tables not in the data: {missing}")
    if not all(hasattr(fam.mod, a) for a in ("reference", "params_choices")):
        problems.append("reference or params_choices missing")
    if not add("contract", not problems, "; ".join(problems) or f"{len(ids)} rules, tables {fam.tables}"):
        return res

    std = set(sys.stdlib_module_names)
    bad = sorted({(n.module or "").split(".")[0] if isinstance(n, ast.ImportFrom) else a.name.split(".")[0]
                  for n in ast.walk(ast.parse(fam.source)) if isinstance(n, (ast.Import, ast.ImportFrom))
                  for a in (n.names if isinstance(n, ast.Import) else [None])} - std)
    add("stdlib", not bad, f"non-stdlib imports: {bad}" if bad else "standard library only")

    try:
        choices = fam.params_choices(dbs[0])
        json.dumps(choices)
    except Exception as e:
        add("params", False, f"params_choices failed: {type(e).__name__}: {e}")
        return res
    full = tuple(ids)
    try:
        distinct = len({json.dumps(fam.reference(dbs[0], p, full), sort_keys=True) for p in choices})
    except Exception as e:
        add("params", False, f"reference failed on a params choice: {type(e).__name__}: {e}")
        return res
    if not add("params", len(choices) >= 5 and distinct >= 5,
               f"{len(choices)} choices, {distinct} different answers (want >= 5 of each; 8+ choices is better)"):
        return res
    sample = choices[:: max(1, len(choices) // max_params)][:max_params]

    problems, slow = [], 0.0
    for i, db in enumerate(dbs):
        for p in sample:
            t0 = time.time()
            a = fam.reference(db, p, full)
            slow = max(slow, time.time() - t0)
            if a != fam.reference(db, p, full):
                problems.append(f"slice {i}: two runs differ")
            if not all(isinstance(k, str) and type(v) is int for k, v in a.items()):
                problems.append(f"slice {i}: answer is not {{str: int}}")
            if len(a) < 15:
                problems.append(f"slice {i} params {p}: only {len(a)} keys")
    if slow > 2:
        problems.append(f"reference took {slow:.1f} s")
    add("answers", not problems, "; ".join(sorted(set(problems))[:6]) or f"deterministic, slowest call {slow:.2f} s")

    thin = [f"slice {i} params {p}: {len(a)} keys" for i, db in enumerate(dbs) for p in sample
            if len(a := fam.reference(db, p, ())) < 15]
    add("base", not thin, "; ".join(thin[:4]) or "plain job (all rules off) gives 15+ keys everywhere")

    problems, signatures = [], {}
    for rid in ids:
        rest = tuple(r for r in ids if r != rid)
        hits = pairs = 0
        per_slice_top, per_slice_alone, sig = [], [], []
        for i, db in enumerate(dbs):
            top = alone = False
            for p in sample:
                f, m = fam.reference(db, p, full), fam.reference(db, p, rest)
                changed = sorted(k for k in set(f) | set(m) if f.get(k) != m.get(k))
                pairs += 1
                hits += bool(changed)
                top |= bool(changed)
                alone |= fam.reference(db, p, (rid,)) != fam.reference(db, p, ())
                sig.append(tuple((k, f.get(k), m.get(k)) for k in changed))
            per_slice_top.append(top)
            per_slice_alone.append(alone)
        signatures[rid] = tuple(sig)
        if not all(per_slice_top):
            problems.append(f"{rid}: dropping it from the full stack changes nothing on slice(s) {[i for i, x in enumerate(per_slice_top) if not x]}")
        if not all(per_slice_alone):
            problems.append(f"{rid}: switching it on alone changes nothing on slice(s) {[i for i, x in enumerate(per_slice_alone) if not x]}")
        if hits / pairs < 0.5:
            problems.append(f"{rid}: matters in only {hits}/{pairs} (slice, params) pairs")
    add("rule effect", not problems, "; ".join(problems[:8]) or "every rule changes the answer on every slice")
    same = [f"{a} and {b}" for i, a in enumerate(ids) for b in ids[i + 1:] if signatures[a] == signatures[b]]
    add("distinct", not same, f"rules with identical effect: {same}" if same else "no two rules have the same effect")

    hi = len(F.ticket(fam, F.Task(fam.id, full, sample[0])).split())
    lo = len(F.ticket(fam, F.Task(fam.id, (), sample[0])).split())
    add("ticket", 150 <= lo and hi <= 300, f"{lo} words with all rules off, {hi} with all on (want 150 to 300)")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family", type=Path)
    ap.add_argument("business", type=Path)
    ap.add_argument("--modes", type=Path)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    modes = {m["id"] for m in json.loads(a.modes.read_text())} if a.modes else None
    res = check(a.family, a.business, modes)
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        for r in res:
            print(f"{'PASS' if r['ok'] else 'FAIL'}  {r['check']:<12} {r['detail']}")
        print("FAMILY CHECK: " + ("ALL PASS" if all(r["ok"] for r in res) else "NOT READY"))
    return 0 if all(r["ok"] for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
