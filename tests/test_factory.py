import json

import pytest

from rsi import business_gen, events, family as F, family_check, gate, intake, scan_hn
from rsi.qwen import Ctx, SETS, by_combination, choose_levels, collect_tasks, compare, dev_tasks, explore_tasks, markdown, mcnemar, rate, train_plan


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


def test_tasks_that_share_a_rule_combination_count_as_one_unit():
    rows = lambda solved: [{"id": i, "combo": i // 5, "solved": i in solved} for i in range(30)]  # 6 combinations x 5 tasks
    one = by_combination(rows(set()), rows(set(range(10))))  # 10 tasks gained, but they are 2 combinations
    assert (one["combinations"], one["up"], one["down"], one["p_by_combination"]) == (6, 2, 0, 0.5)
    assert mcnemar(rows(set()), rows(set(range(10))))["p"] < 0.01  # the task-level test calls the same data significant
    spread = by_combination(rows(set()), rows({0, 5, 10, 15, 20, 25}))  # 6 tasks gained across 6 combinations
    assert spread["p_by_combination"] == 0.03125
    assert by_combination(rows({1}), rows({1}))["p_by_combination"] == 1.0


def test_crash_feedback_is_the_candidates_own_traceback(business, shoe):
    db = F.load_dbs(business)[0]
    t = F.Task(shoe.id, (), {"store_id": "S02"})
    trace = F.crash_on_shown(t, "```python\ndef solve(db, params):\n    day = '2026-01-02'\n    return {'x': day - 1}\n```", db)
    assert trace.splitlines()[0].strip() == "line 3, in solve" and trace.endswith("'str' and 'int'")
    assert "cand.py" not in trace
    assert F.crash_on_shown(t, "```python\ndef solve(db, params):\n    return {'x': 1}\n```", db) == ""  # wrong but runs: no feedback


def test_levels_are_chosen_so_pooled_before_rates_sit_on_the_targets():
    a = {0: (14, 20), 1: (10, 20), 2: (5, 20), 3: (1, 20)}
    assert choose_levels({"a": a}) == {"a": (1, 2)}  # 50% and 25% exactly
    b = {0: (8, 20), 1: (4, 20), 2: (0, 20)}  # a weaker family: its plain job is its headline
    got = choose_levels({"a": a, "b": b})
    assert got["b"] == (0, 1) and got["a"][0] <= 1
    pooled = sum(x[got[f][0]][0] for f, x in (("a", a), ("b", b))) / 40
    assert abs(pooled - 0.5) <= 0.1
    assert choose_levels({"a": {0: (3, 20)}}) == {}  # one level cannot give a headline and a harder set


def test_training_never_uses_a_benchmark_combination(shoe):
    split = F.split_combos(shoe)
    plan = train_plan([shoe], {shoe.id: {"headline": 2, "harder": 3}}, 6)[shoe.id]
    assert plan["levels"] == [0, 1, 2, 3]  # level 0 is free to train on: the benchmark sits at 2 and 3
    assert train_plan([shoe], {shoe.id: {"headline": 0, "harder": 2}}, 6)[shoe.id]["levels"] == [1, 2]
    assert train_plan([shoe], {}, 6)[shoe.id] == {"levels": [1], "n": 3}
    assert all(split[l]["train"] for l in plan["levels"] if l)


def test_dev_tasks_are_never_trained_on_and_never_in_the_benchmark(business, shoe):
    ctx = Ctx()
    ctx.add(shoe, F.load_dbs(business))
    plan = train_plan([shoe], {shoe.id: {"headline": 1, "harder": 3}}, 6)
    dev = dev_tasks(ctx, plan, 12)
    assert len({t.id for t in dev}) == 12 and {t.level for t in dev} == {1, 2, 3}
    dev_rules = {t.rules for t in dev}
    split = F.split_combos(shoe)
    assert not dev_rules & {c for s in split.values() for c in s["bench"]}
    for rnd in (1, 2):
        tasks = collect_tasks(ctx, plan, rnd)
        assert not dev_rules & {t.rules for t in tasks}
        assert [sum(t.level == l for t in tasks) for l in (1, 2, 3)] == [6, 6, 6]  # the set-aside ones are replaced, not just dropped
    rule = split[1]["bench"][0][0]  # alone it is a benchmark task, so the extra samples must come from a pair
    more = explore_tasks(ctx, plan, {shoe.id: {rule}}, 1)[(shoe.id, rule)]
    assert more and all(rule in t.rules and t.level == 2 for t in more)
    assert not {t.rules for t in more} & (dev_rules | set(split[2]["bench"]))


def test_report_renders_before_training_and_after_a_round():
    def bench(solved):
        rows = {n: [{"id": f"{n}-{i}", "solved": i < solved[n]} for i in range(20 if n != "out_of_reach" else 0)] for n in SETS}
        return {"model": {"adapter": "runs/x/qwen/adapter_r1"}, "rows": rows,
                "sets": {n: {**rate(v), "by_family": {"fam L2": rate(v)}, "failures": {"wrong answer": 3}} for n, v in rows.items()}}
    r = {"model": {"id": "m", "weights": "w", "quantization": None}, "dataset": "d", "slices": 3, "families": ["fam"], "control": "shoe",
         "settings": {"headline_target": 0.5, "harder_target": 0.25},
         "calibrate": {"fam": {"reach": True, "note": "", "curve": [{"level": 2, **rate([{"solved": True}] * 20)}]}},
         "settle": {"targets": {"headline": 0.5, "harder": 0.25, "tolerance": 0.1}, "reached": True,
                    "steps": [{"levels": {"fam": [2, 3]}, "headline": 0.5, "harder": 0.25, "measured": True}],
                    "before_by_level": {"fam": {"2": {"solved": 10, "n": 20}}}},
         "before": bench({"headline": 10, "harder": 5, "control": 4, "out_of_reach": 0})}
    assert "50%" in markdown(r) and "has not finished" in markdown(r)
    r["after_r1"] = bench({"headline": 16, "harder": 9, "control": 4, "out_of_reach": 0})
    r["compare_r1"] = compare(r["before"], r["after_r1"])
    r["collect_r1"] = {"solved": 30, "sampled": 100, "solved_after_fix": 8, "fix_turns": 40, "rules_with_a_kept_answer": {"fam": "7/8"}}
    dev = lambda k: {"solved": k, "n": 32, "crashed": 2}
    r["dev_base"] = dev(9)
    r["train_r1"] = {"rows": 38, "rows_by_family": {"fam": 38}, "steps": 114, "first_loss": 0.2, "last_loss": 0.1, "seconds": 300, "peak_memory_gb": 12.0,
                     "settings": {"learning_rate": 5e-5}, "chosen_epoch": 2,
                     "epochs": [{"epoch": e, "steps": 38, "loss": 0.1, "dev": dev(k)} for e, k in ((1, 12), (2, 17), (3, 15))]}
    r["last_round"] = 1
    text = markdown(r)
    assert "| headline | 10/20 (50%" in text and "16/20 (80%" in text and "**met**" in text and "out_of_reach" not in text
    assert "| headline + harder | 15/40 (38%) | 25/40 (62%) | 10 | 0 |" in text
    assert "12/32 (2 crash), 17/32 (2 crash), 15/32 (2 crash); epoch 2 kept" in text and "solves 9/32" in text


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
