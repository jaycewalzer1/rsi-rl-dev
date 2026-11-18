"""Stage 0: take in a business data export, prune it, and cut it into slices tasks can use.

Input: a folder of CSV files (one table each). Output, under runs/<run>/business/:
  tables/<name>.json     slice 0 of the pruned data: what agents and solvers are shown
  twins/1, twins/2 ...   further disjoint slices of the SAME business's data, same schema;
                         answers are graded on these too, so fitting slice 0 by luck fails
  catalog.json           per table: columns, keys, row counts, every prune with its count,
                         and the business rules a steward agent read out of the data

Pruning is deterministic, so the same export always gives the same dataset:
  pii          columns that hold people's names, emails, phones or addresses are dropped
  free text    long free-text columns (comments, messages) are dropped: they can hold
               personal details and no task computes on them
  test rows    rows whose id or status marks QA data
  duplicates   exact duplicate rows kept once (returns are left alone: a resent event
               with its own row id is real data the tasks reason about)
  sparse       columns at least 90% empty, and columns with one value everywhere
  orphans      rows whose foreign key points at a missing or pruned row
  enums        stray case and spaces trimmed in short categorical text
  types        whole-number columns become ints; money columns (exactly two decimals wherever there is a decimal point)
               become integer cents and gain a _cents suffix, so grading is exact

Slicing: the root table is the one most other tables point at (orders, usually). Its rows
are ordered by a seeded hash and cut into `slices` groups of `rows`; every other table
follows by foreign key, so each slice is a small, complete copy of the business.

Then one steward agent per table (judge model) reads the schema and sample rows and
writes down the business rules the data implies. Those go to the task authors.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from . import events, llm
from .config import CONFIG
from .store import write_json

PII_NAME = re.compile(r"^((customer|user|contact|person|buyer|client)_)?name$|(^|_)(full_name|first_name|last_name|email|phone|address|street|street_address|ssn|dob)$")
TEXT_NAME = re.compile(r"(comment|message|note|feedback|body)")
PII_VALUE = re.compile(r"[^@\s]+@[^@\s]+\.[a-z]{2,}|(?<![\w-])\+?\d{1,3}[-\s(]+\d{3}[-\s)]+\d{3}[-\s]\d{4}\b")
TEST_ID = re.compile(r"^(TEST|C0000|O00000)$")
KEEP_DUPES = {"returns"}
KEEP_COLUMNS = {"name"}  # a store's or product's name is not personal data
KEEP_SPARSE = {"requested_at", "requested_local"}  # one of the two is always empty by design
INT = re.compile(r"-?\d+")
MONEY = re.compile(r"-?\d+\.\d{2}")  # exactly two decimals: 58.90 is money, 1.5 is a measure
FLOAT = re.compile(r"-?\d+\.\d+")
IS_KEY = re.compile(r"(_id|^sku)$")


def read_tables(src: Path) -> dict[str, list[dict]]:
    out = {}
    for p in sorted(src.glob("*.csv")):
        with p.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if rows:
            out[p.stem] = rows
    if not out:
        raise SystemExit(f"intake: no CSV files in {src}")
    return out


def pii_columns(table: str, rows: list[dict]) -> list[str]:
    cols = []
    for col in rows[0]:
        if col in KEEP_COLUMNS and table in ("stores", "products"):
            continue
        values = [r[col] for r in rows[:2000] if r[col]][:50]
        hits = sum(bool(PII_VALUE.search(v)) for v in values)
        if PII_NAME.search(col) or (values and hits / len(values) > 0.5):
            cols.append(col)
    return cols


def free_text_columns(rows: list[dict]) -> list[str]:
    cols = []
    for col in rows[0]:
        values = [r[col] for r in rows[:5000] if r[col]]
        if len(values) < 20 or IS_KEY.search(col) or all(INT.fullmatch(v) for v in values[:200]):
            continue
        if TEXT_NAME.search(col):
            cols.append(col)
            continue
        spaced = sum(" " in v.strip() for v in values) / len(values)
        long_ = sum(len(v) for v in values) / len(values)
        if spaced > 0.5 and long_ > 25 and len(set(values)) / len(values) > 0.5:
            cols.append(col)
    return cols


def keys(table: str, rows: list[dict], all_pks: dict[str, str] | None = None) -> tuple[str | None, list[str]]:
    """Primary key: an id column with no repeats (None for a table keyed by several columns).
    Foreign keys: the other id columns, or, once every table's key is known, those that name one."""
    cols = [c for c in rows[0] if IS_KEY.search(c)]
    unique = [c for c in cols if len({r[c] for r in rows}) == len(rows)]
    stem = table.rstrip("s")
    pk = next((c for c in unique if c.startswith(stem)), unique[0] if unique else None)
    fks = [c for c in cols if c != pk]
    if all_pks is not None:
        fks = [c for c in fks if c in all_pks.values()]
    return pk, fks


def retype(rows: list[dict]) -> dict[str, str]:
    """Column-wise typing. Returns {old column: new column} for renamed money columns."""
    renamed = {}
    for c in list(rows[0]):
        vals = [r[c] for r in rows if r[c] not in ("", None)]
        if not vals or not all(isinstance(v, str) for v in vals):
            continue
        if all(v in ("true", "false") for v in vals):
            conv = lambda v: v == "true"  # noqa: E731
        elif all(INT.fullmatch(v) for v in vals) and not any(len(v) > 1 and v.lstrip("-")[0] == "0" for v in vals):
            conv = int
        elif all(MONEY.fullmatch(v) or INT.fullmatch(v) for v in vals) and any("." in v for v in vals) and not IS_KEY.search(c):
            conv = lambda v: int(round(float(v) * 100))  # noqa: E731
            renamed[c] = c if c.endswith("_cents") else f"{c}_cents"
        elif all(FLOAT.fullmatch(v) or INT.fullmatch(v) for v in vals) and any("." in v for v in vals):
            conv = float
        else:
            continue
        for r in rows:
            r[c] = conv(r[c]) if r[c] not in ("", None) else None
    if renamed:
        for i, r in enumerate(rows):
            rows[i] = {renamed.get(k, k): v for k, v in r.items()}
    return renamed


def prune(raw: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], dict]:
    tables = {k: [dict(r) for r in v] for k, v in raw.items()}
    report: dict[str, dict] = {k: {"rows_in": len(v), "prunes": []} for k, v in tables.items()}

    def note(t, what, n, detail=""):
        if n:
            report[t]["prunes"].append({"what": what, "count": n, "detail": detail})

    for t, rows in tables.items():
        pii = pii_columns(t, rows)
        text = [c for c in free_text_columns(rows) if c not in pii]
        for r in rows:
            for c in pii + text:
                r.pop(c, None)
        note(t, "pii columns dropped", len(pii), ", ".join(pii))
        note(t, "free-text columns dropped", len(text), ", ".join(text))

        before = len(rows)
        rows[:] = [r for r in rows if not any(TEST_ID.match(str(v)) for v in r.values()) and r.get("status") != "test"]
        note(t, "test rows", before - len(rows))

        if t not in KEEP_DUPES:
            seen, kept = set(), []
            for r in rows:
                sig = tuple(r.items())
                if sig not in seen:
                    seen.add(sig)
                    kept.append(r)
            note(t, "exact duplicates", len(rows) - len(kept))
            rows[:] = kept

        for c in list(rows[0]) if rows else []:
            vals = [r[c] for r in rows]
            empty = sum(1 for v in vals if v in ("", None))
            if (empty / len(vals) >= 0.9 and c not in KEEP_SPARSE) or (len(set(vals)) == 1 and len(vals) > 3):
                for r in rows:
                    r.pop(c, None)
                note(t, "sparse or constant column", 1, c)

        fixed = 0
        for c in list(rows[0]) if rows else []:
            norm = {str(r[c]).strip().lower() for r in rows}
            if len(norm) > 6 or not all(v.isalpha() for v in norm):
                continue  # only short categorical text
            for r in rows:
                if r[c] != r[c].strip().lower():
                    r[c] = r[c].strip().lower()
                    fixed += 1
        note(t, "enum values normalised", fixed)

    # orphans: repeat until nothing changes, since pruning a parent can orphan its children
    pks = {t: keys(t, rows)[0] for t, rows in tables.items() if rows}
    pks = {t: k for t, k in pks.items() if k}
    changed = True
    while changed:
        changed = False
        for t, rows in tables.items():
            if not rows:
                continue
            for fk in keys(t, rows, pks)[1]:
                parent = next((p for p, k in pks.items() if k == fk and p != t), None)
                if parent is None:
                    continue
                ids = {r[fk] for r in tables[parent]}
                before = len(rows)
                rows[:] = [r for r in rows if r[fk] in ids]
                if before != len(rows):
                    note(t, "orphan rows", before - len(rows), f"{fk} not in {parent}")
                    changed = True

    for t, rows in tables.items():
        if not rows:
            continue
        pk, fks = keys(t, rows, pks)  # before retyping: ids stay text
        renamed = retype(rows)
        note(t, "money columns turned into integer cents", len(renamed), ", ".join(renamed.values()))
        report[t]["rows_out"] = len(rows)
        report[t]["columns"] = list(rows[0])
        report[t]["primary_key"], report[t]["foreign_keys"] = pk, fks
    return tables, report


def cut(tables: dict[str, list[dict]], report: dict, rows: int, slices: int, seed: int = 0) -> list[dict[str, list[dict]]]:
    """Disjoint, complete slices of the business: root rows by seeded hash, the rest by foreign key."""
    pks = {t: r["primary_key"] for t, r in report.items() if r.get("primary_key")}
    fks = {t: r.get("foreign_keys", []) for t, r in report.items()}
    inbound = Counter(p for t in tables for fk in fks[t] for p, k in pks.items() if k == fk and p != t)
    if not inbound:
        return [tables]
    root = inbound.most_common(1)[0][0]
    order = sorted(tables[root], key=lambda r: hashlib.sha256(f"{seed}|{r[pks[root]]}".encode()).hexdigest())
    if len(order) < rows * slices:
        rows = max(1, len(order) // slices)
    out = []
    for i in range(slices):
        part = {root: order[i * rows:(i + 1) * rows]}
        pending = [t for t in tables if t != root]
        progressed = True
        while pending and progressed:
            progressed = False
            for t in list(pending):
                # a child of a table already cut: keep rows whose foreign key survives
                parent = next((p for p in part for fk in fks[t] if pks.get(p) == fk), None)
                if parent is not None:
                    ids = {r[pks[parent]] for r in part[parent]}
                    part[t] = [r for r in tables[t] if r[pks[parent]] in ids]
                # a parent of a table already cut: keep the rows it points at
                elif t in pks and any(pks[t] in fks[c] for c in part):
                    ids = {r[pks[t]] for c in part if pks[t] in fks[c] for r in part[c]}
                    part[t] = [r for r in tables[t] if r[pks[t]] in ids]
                else:
                    continue
                pending.remove(t)
                progressed = True
        for t in pending:  # lookup tables with no key relation: whole
            part[t] = tables[t]
        out.append(part)
    return out


STEWARD = """You are a data steward. Below is one table of a company's pruned data export
(schema, value counts, sample rows). Write down the business rules this data implies that
code working with it must respect (units, time columns and what they mean, which column
holds what, repeated ids, empty values, edge values). Plain statements only, no guesses
about other tables.

TABLE {name}
COLUMNS {columns}
VALUE COUNTS {counts}
SAMPLE ROWS
{sample}

JSON: {{"rules": ["...", "..."], "watch_outs": ["..."]}}"""


def steward(name: str, rows: list[dict]) -> dict:
    counts = {c: Counter(str(r[c]) for r in rows).most_common(8) for c in rows[0]
              if len({r[c] for r in rows}) <= 12}
    return llm.ask_json(
        STEWARD.format(name=name, columns=list(rows[0]), counts=json.dumps(counts),
                       sample="\n".join(json.dumps(r) for r in rows[:8])),
        CONFIG.judge_model,
    )


def run(src: Path, out_dir: Path, *, stewards: bool = True, rows: int = 1500, slices: int = 3) -> Path:
    events.emit("stage_start", stage="intake", msg=f"export {src.parent.name}/{src.name}")
    raw = read_tables(src)
    for t, rws in raw.items():
        events.emit("work", stage="intake", agent=f"loader-{t}", role="loader",
                    msg=f"{t}: {len(rws)} rows, {len(rws[0])} columns")
    tables, report = prune(raw)
    for t, r in report.items():
        for p in r["prunes"]:
            events.emit("work", stage="prune", agent=f"pruner-{t}", role="pruner",
                        msg=f"{t}: {p['what']} {p['count']} {p['detail']}".strip(), count=p["count"])
    parts = cut(tables, report, rows, slices)
    base = out_dir / "business"
    for i, part in enumerate(parts):
        for t, rws in part.items():
            write_json(base / ("tables" if i == 0 else f"twins/{i}") / f"{t}.json", rws)
    for t in tables:
        report[t]["slice_rows"] = [len(p.get(t, [])) for p in parts]
    events.emit("work", stage="prune", agent="slicer", role="pruner",
                msg=f"{len(parts)} disjoint slices; rows per table in slice 0: "
                    + ", ".join(f"{t} {len(v)}" for t, v in parts[0].items()))

    if stewards:
        names = [t for t in parts[0] if parts[0][t]]
        notes = events.swarm("prune", "steward", names, lambda t, agent: steward(t, parts[0][t]), workers=6)
        for t, n in zip(names, notes):
            report[t]["steward"] = n or {}
    catalog = {"source": str(src), "synthetic": "shoe-store" in str(src), "slices": len(parts), "tables": report}
    write_json(base / "catalog.json", catalog)
    kept = sum(r.get("rows_out", 0) for r in report.values())
    dropped = sum(r["rows_in"] for r in report.values()) - kept
    events.emit("stage_end", stage="prune", msg=f"{kept} rows kept, {dropped} pruned, {len(parts)} slices",
                kept=kept, dropped=dropped)
    print(f"intake: {len(tables)} tables, {kept} rows kept, {dropped} pruned, {len(parts)} slices -> {base}")
    return base
