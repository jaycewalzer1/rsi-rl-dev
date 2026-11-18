import json

import pytest

from rsi import business_gen, events, family as F, family_check, gate, intake, scan_hn
from rsi.qwen import mcnemar


def raw(seed=7, n=900):
    rows = business_gen.generate(seed, n_customers=200, n_orders=n)
    return {k: [{c: str(v) for c, v in r.items()} for r in v] for k, v in rows.items()}


@pytest.fixture(scope="module")
def business(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    src = out / "raw"
    business_gen.write_csv(business_gen.generate(7), src)
    return intake.run(src, out, stewards=False, rows=800)


@pytest.fixture(scope="module")
def shoe():
    return F.load(F.BUILTIN / "shoe_store.py")


def test_prune_drops_pii_test_rows_dupes_and_orphans():
    tables, report = intake.prune(raw())
    assert "email" not in tables["customers"][0] and "full_name" not in tables["customers"][0]
    assert "name" in tables["stores"][0]  # a store's name is not personal data
    assert "body" not in tables["support_tickets"][0]  # free text goes
    assert all(s["store_id"] != "TEST" for s in tables["stores"])
    whats = {p["what"] for p in report["order_lines"]["prunes"]}
    assert {"exact duplicates", "orphan rows"} <= whats
    order_ids = {o["order_id"] for o in tables["orders"]}
    assert all(line["order_id"] in order_ids for line in tables["order_lines"])
    assert {o["channel"] for o in tables["orders"]} == {"online", "store"}
    ev = [r["event_id"] for r in tables["returns"]]
    assert len(ev) > len(set(ev))  # resent return events are data, not duplicates


def test_types_money_becomes_cents_and_ids_stay_text():
    rows = [{"order_id": "0012", "zip": "01234", "price": "58.90", "qty": "2", "weight": "1.25"},
            {"order_id": "0013", "zip": "99999", "price": "5", "qty": "", "weight": "0.5"}]
    renamed = intake.retype(rows)
    assert renamed == {"price": "price_cents"}
    assert rows[0] == {"order_id": "0012", "zip": "01234", "price_cents": 5890, "qty": 2, "weight": 1.25}
    assert rows[1]["price_cents"] == 500 and rows[1]["qty"] is None


def test_category_name_is_not_personal_data():
    rows = [{"product_id": str(i), "product_category_name": "perfumaria"} for i in range(30)]
    assert intake.pii_columns("products", rows) == []


def test_slices_are_disjoint_and_complete(business):
    dbs = F.load_dbs(business)
    assert len(dbs) == 3
    ids = [{o["order_id"] for o in db["orders"]} for db in dbs]
    assert not ids[0] & ids[1] and not ids[1] & ids[2]
    for db, own in zip(dbs, ids):
        assert db["order_lines"] and all(line["order_id"] in own for line in db["order_lines"])
        lines = {line["line_id"] for line in db["order_lines"]}
        assert all(r["line_id"] in lines for r in db["returns"])


def test_checker_passes_the_shipped_family_and_fails_a_dead_rule(business, shoe, tmp_path):
    modes = {r["mode"] for r in shoe.rules}
    assert all(c["ok"] for c in family_check.check(shoe.path, business, modes))
    dead = tmp_path / "family.py"
    dead.write_text(shoe.source.replace('if "dedupe" in rules:', 'if "dedupe" in rules and False:'))
    res = {c["check"]: c for c in family_check.check(dead, business, modes)}
    assert not res["rule effect"]["ok"] and "dedupe" in res["rule effect"]["detail"]
    res = {c["check"]: c for c in family_check.check(shoe.path, business, {"timezone-math"})}
    assert not res["contract"]["ok"]


def test_bench_and_train_combinations_never_overlap(shoe):
    split = F.split_combos(shoe)
    for lvl, s in split.items():
        assert s["bench"] and not set(s["bench"]) & set(s["train"])
    assert split[8]["train"] == [] and split[0]["train"] == []


def test_tasks_are_distinct_and_from_their_pool(business, shoe):
    db = F.load_dbs(business)[0]
    bench = F.make_tasks(shoe, 2, 12, "bench", 1, db)
    train = F.make_tasks(shoe, 2, 12, "train", 1, db)
    assert len({t.id for t in bench}) == 12
    assert not {t.rules for t in bench} & {t.rules for t in train}


def test_grader_accepts_oracle_and_rejects_forgotten_rule(business, shoe):
    dbs = F.load_dbs(business)
    t = F.Task(shoe.id, tuple(shoe.rule_ids), {"store_id": "S02"})
    good = shoe.source + f"\ndef solve(db, params):\n    return reference(db, params, {set(t.rules)!r})\n"
    bad = shoe.source + f"\ndef solve(db, params):\n    return reference(db, params, {set(t.rules[1:])!r})\n"
    assert F.score(shoe, t, f"```python\n{good}```", dbs)["solved"]
    g = F.score(shoe, t, f"```python\n{bad}```", dbs)
    assert not g["solved"] and g["diff"]["examples"]
    assert F.score(shoe, t, "```python\nimport pytz\n```", dbs)["error"].startswith("ModuleNotFoundError")


def test_export_matches_the_task_contract(business, shoe, tmp_path):
    dbs = F.load_dbs(business)
    t = F.Task(shoe.id, ("window", "dedupe", "discount"), {"store_id": "S02"})
    d = F.export_harbor(shoe, t, dbs, tmp_path)
    assert gate.stage_shape(d)["ok"]
    crit = json.loads((d / "authoring/criteria.json").read_text())
    assert [c["rule"] for c in crit] == list(t.rules) and all(c["tests"] for c in crit)
    assert len(list((d / "authoring/mutations").glob("*.patch"))) == 3


def test_mcnemar_counts_only_changed_tasks():
    before = [{"id": i, "solved": i < 2} for i in range(10)]
    after = [{"id": i, "solved": i < 8} for i in range(10)]
    assert mcnemar(before, after) == {"gained": 6, "lost": 0, "p": 0.03125}
    assert mcnemar(before, before)["p"] == 1.0


def test_swarm_logs_every_agent(tmp_path):
    events.open_log(tmp_path, fresh=True)
    out = events.swarm("scan", "scout", [1, 2, 0], lambda x, a: 10 // x)
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    events._path = None
    assert out == [10, 5, None]
    assert [r["kind"] for r in rows].count("spawn") == 3
    assert [r["kind"] for r in rows].count("fail") == 1


def test_hn_text_is_cleaned():
    assert scan_hn._clean("<p>it said &quot;done&quot;</p><i>but</i>") == 'it said "done" but'
