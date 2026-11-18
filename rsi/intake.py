"""Stage 0: take in a business data export and prune it to what tasks may use.

Input: a folder of CSV files (one table each). Output, under runs/<run>/business/:
  tables/<name>.json   pruned rows
  catalog.json         per table: columns, keys, row counts, every prune with its count,
                       and the business rules a steward agent read out of the data

Pruning is deterministic, so the same export always gives the same dataset:
  pii          columns that hold people's names, emails, phones or addresses are dropped
  test rows    rows whose id or status marks QA data (TEST, C0000, O00000, status=test)
  duplicates   exact duplicate rows kept once (returns are left alone: a resent event
               with its own row id is real data the tasks reason about)
  sparse       columns at least 90% empty, and columns with one value everywhere
  orphans      rows whose foreign key points at a missing or pruned row
  enums        stray case and spaces trimmed in short categorical text

Then one steward agent per table (judge model) reads the schema and sample rows and
writes down the business rules the data implies. Those go to the designers.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

from . import events, llm
from .config import CONFIG
from .store import write_json

PII_NAME = re.compile(r"(^|_)(name|full_name|email|phone|address|street|street_address|ssn|dob|customer_email)$")
PII_VALUE = re.compile(r"[^@\s]+@[^@\s]+\.[a-z]{2,}|(?<![\w-])\+?\d{1,3}[-\s(]+\d{3}[-\s)]+\d{3}[-\s]\d{4}\b")
TEST_ID = re.compile(r"^(TEST|C0000|O00000)$")
KEEP_DUPES = {"returns"}
KEEP_COLUMNS = {"name"}  # a store's or product's name is not personal data


def read_tables(src: Path) -> dict[str, list[dict]]:
    out = {}
    for p in sorted(src.glob("*.csv")):
        with p.open(newline="") as f:
            out[p.stem] = list(csv.DictReader(f))
    if not out:
        raise SystemExit(f"intake: no CSV files in {src}")
    return out


def _coerce(v: str):
    if re.fullmatch(r"-?\d+", v or ""):
        return int(v)
    if v in ("true", "false"):
        return v == "true"
    return v


def pii_columns(table: str, rows: list[dict]) -> list[str]:
    cols = []
    for col in rows[0]:
        if col in KEEP_COLUMNS and table in ("stores", "products"):
            continue
        values = [r[col] for r in rows if r[col]][:50]
        hits = sum(bool(PII_VALUE.search(v)) for v in values)
        if PII_NAME.search(col) or (values and hits / len(values) > 0.5):
            cols.append(col)
    return cols


def keys(rows: list[dict]) -> tuple[str | None, list[str]]:
    cols = list(rows[0])
    pk = next((c for c in cols if c.endswith("_id") and len({r[c] for r in rows}) == len(rows)), None)
    if pk is None:  # duplicates hide the key; the first *_id column is it
        pk = next((c for c in cols if c.endswith("_id") or c == "sku"), None)
    return pk, [c for c in cols if (c.endswith("_id") or c == "sku") and c != pk]


def prune(raw: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], dict]:
    tables = {k: [dict(r) for r in v] for k, v in raw.items()}
    report: dict[str, dict] = {k: {"rows_in": len(v), "prunes": []} for k, v in tables.items()}

    def note(t, what, n, detail=""):
        if n:
            report[t]["prunes"].append({"what": what, "count": n, "detail": detail})

    for t, rows in tables.items():
        pii = pii_columns(t, rows)
        for r in rows:
            for c in pii:
                r.pop(c, None)
        note(t, "pii columns dropped", len(pii), ", ".join(pii))

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

        if rows:
            for c in list(rows[0]):
                vals = [r[c] for r in rows]
                empty = sum(1 for v in vals if v in ("", None))
                if (empty / len(vals) >= 0.9 and c not in ("requested_at", "requested_local")) or (
                    len(set(vals)) == 1 and len(vals) > 3
                ):
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
    pks = {t: keys(rows)[0] for t, rows in tables.items() if rows}
    changed = True
    while changed:
        changed = False
        for t, rows in tables.items():
            if not rows:
                continue
            for fk in keys(rows)[1]:
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
        for r in rows:
            for c, v in r.items():
                r[c] = _coerce(v) if isinstance(v, str) else v
        report[t]["rows_out"] = len(rows)
        report[t]["columns"] = list(rows[0]) if rows else []
        report[t]["primary_key"], report[t]["foreign_keys"] = keys(rows) if rows else (None, [])
    return tables, report


STEWARD = """You are a data steward for a shoe retailer. Below is one table of its pruned
export (schema, value counts, sample rows). Write down the business rules this data
implies that code working with it must respect (time zones, units, which column holds
what, duplicate events, edge values). Plain statements only, no guesses about other tables.

TABLE {name}
COLUMNS {columns}
VALUE COUNTS {counts}
SAMPLE ROWS
{sample}

JSON: {{"rules": ["...", "..."], "watch_outs": ["..."]}}"""


def steward(name: str, rows: list[dict]) -> dict:
    counts = {c: Counter(str(r[c]) for r in rows).most_common(6) for c in rows[0]
              if len({r[c] for r in rows}) <= 12}
    return llm.ask_json(
        STEWARD.format(name=name, columns=list(rows[0]), counts=json.dumps(counts),
                       sample="\n".join(json.dumps(r) for r in rows[:8])),
        CONFIG.judge_model,
    )


def run(src: Path, out_dir: Path, *, stewards: bool = True) -> Path:
    events.emit("stage_start", stage="intake", msg=f"export {src.parent.name}/{src.name}")
    raw = read_tables(src)
    for t, rows in raw.items():
        events.emit("work", stage="intake", agent=f"loader-{t}", role="loader",
                    msg=f"{t}: {len(rows)} rows, {len(rows[0]) if rows else 0} columns")
    tables, report = prune(raw)
    for t, r in report.items():
        for p in r["prunes"]:
            events.emit("work", stage="prune", agent=f"pruner-{t}", role="pruner",
                        msg=f"{t}: {p['what']} {p['count']} {p['detail']}".strip(), count=p["count"])
    base = out_dir / "business"
    for t, rows in tables.items():
        write_json(base / "tables" / f"{t}.json", rows)

    if stewards:
        notes = events.swarm(
            "prune", "steward", [t for t in tables if tables[t]],
            lambda t, agent: steward(t, tables[t]), workers=6,
        )
        for t, n in zip([t for t in tables if tables[t]], notes):
            report[t]["steward"] = n or {}
    catalog = {"source": str(src), "synthetic": "shoe-store" in str(src), "tables": report}
    write_json(base / "catalog.json", catalog)
    kept = sum(r["rows_out"] for r in report.values())
    dropped = sum(r["rows_in"] for r in report.values()) - kept
    events.emit("stage_end", stage="prune", msg=f"{kept} rows kept, {dropped} pruned", kept=kept, dropped=dropped)
    print(f"intake: {len(tables)} tables, {kept} rows kept, {dropped} pruned -> {base}")
    return base
