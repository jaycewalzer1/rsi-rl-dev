import inspect
import json

from rsi import bizops, business_gen, events, gate, intake, scan_hn


def raw(seed=7):
    return {k: [{c: str(v) for c, v in r.items()} for r in rows] for k, rows in business_gen.generate(seed).items()}


def test_prune_drops_pii_test_rows_dupes_and_orphans():
    tables, report = intake.prune(raw())
    assert "email" not in tables["customers"][0] and "full_name" not in tables["customers"][0]
    assert "name" in tables["stores"][0]  # a store's name is not personal data
    assert all(s["store_id"] != "TEST" for s in tables["stores"])
    whats = {p["what"] for p in report["order_lines"]["prunes"]}
    assert {"exact duplicates", "orphan rows"} <= whats
    order_ids = {o["order_id"] for o in tables["orders"]}
    assert all(line["order_id"] in order_ids for line in tables["order_lines"])
    assert {o["channel"] for o in tables["orders"]} == {"online", "store"}
    # resent return events are data, not duplicates
    ev = [r["event_id"] for r in tables["returns"]]
    assert len(ev) > len(set(ev))


def test_every_rule_changes_some_answer():
    dbs = [intake.prune(raw(s))[0] for s in (7, 11, 23)]
    full = tuple(bizops.RULE_IDS)
    for rid in full:
        wrong = tuple(r for r in full if r != rid)
        changed = sum(
            bizops.reference(db, {"store_id": st}, full) != bizops.reference(db, {"store_id": st}, wrong)
            for db in dbs for st in ("S01", "S02", "S03")
        )
        assert changed, rid


def test_bench_and_train_combinations_never_overlap():
    for lvl, split in bizops.split_combos().items():
        assert not set(split["bench"]) & set(split["train"])
        assert split["bench"]
    assert bizops.split_combos()[8]["train"] == []


def test_grader_accepts_oracle_and_rejects_forgotten_rule():
    db = intake.prune(raw())[0]
    t = bizops.Task(tuple(bizops.RULE_IDS), {"store_id": "S02"})
    head = "from collections import defaultdict\nfrom datetime import datetime, timezone\nfrom zoneinfo import ZoneInfo\n"
    src = head + inspect.getsource(bizops._utc) + inspect.getsource(bizops.reference)
    good = src + f"\ndef solve(db, params):\n    return reference(db, params, {t.rules!r})\n"
    bad = src + f"\ndef solve(db, params):\n    return reference(db, params, {t.rules[1:]!r})\n"
    assert bizops.score(t, f"```python\n{good}```", [db])["solved"]
    assert not bizops.score(t, f"```python\n{bad}```", [db])["solved"]


def test_export_matches_the_task_contract(tmp_path):
    db = intake.prune(raw())[0]
    twins = [intake.prune(raw(11))[0]]
    t = bizops.Task(("window", "dedupe", "discount"), {"store_id": "S02"})
    d = bizops.export_harbor(t, db, tmp_path, twins)
    assert gate.stage_shape(d)["ok"]
    crit = json.loads((d / "authoring/criteria.json").read_text())
    assert [c["rule"] for c in crit] == list(t.rules) and all(c["tests"] for c in crit)
    assert len(list((d / "authoring/mutations").glob("*.patch"))) == 3


def test_swarm_logs_every_agent(tmp_path):
    events.open_log(tmp_path, fresh=True)
    out = events.swarm("scan", "scout", [1, 2, 0], lambda x, a: 10 // x)
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert out == [10, 5, None]
    assert [r["kind"] for r in rows].count("spawn") == 3
    assert [r["kind"] for r in rows].count("fail") == 1


def test_hn_text_is_cleaned():
    assert scan_hn._clean("<p>it said &quot;done&quot;</p><i>but</i>") == 'it said "done" but'
