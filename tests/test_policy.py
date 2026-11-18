"""The design policy: handle rates, pair interactions, the stack prediction, and what it picks."""

import json
import random
from collections import Counter

import pytest

from rsi import calibrate, combine, gate

SOLVER = "anthropic/claude-opus-5-5"


def row(modes, attempts, solved, slips, patterns=None, model=SOLVER):
    r = {"task": "-".join(modes), "modes": modes, "attempts": attempts, "solved": solved, "mode_slips": slips, "models": {model: {}}}
    if patterns is not None:
        r["by_model"] = {model: {"attempts": attempts, "solved": solved, "mode_slips": slips, "patterns": patterns}}
    return r


@pytest.fixture
def history(tmp_path, monkeypatch):
    """Use only the rows a test writes: no hackathon prior, no local runs/."""
    path = tmp_path / "history.jsonl"
    monkeypatch.setattr(combine, "HISTORY", path)
    monkeypatch.setattr(combine, "PRIOR_HISTORY", tmp_path / "none.jsonl")
    return lambda rows: path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_a_mode_that_never_slipped_is_credited_as_handled():
    """The hackathon rows list only modes that slipped; idempotency (0 slips in 15 attempts) used to
    get no update at all, so its handicap never applied."""
    st = combine.stats(combine.read_jsonl(combine.PRIOR_HISTORY))
    assert (st.slips["idempotency-under-failure"], st.handled["idempotency-under-failure"]) == (0, 15)
    assert st.handle_rate("idempotency-under-failure") > 0.9 > st.handle_rate("derived-surface-omitted")


def test_prediction_is_the_product_of_handle_rates_without_pattern_data():
    st = combine.stats([row(["a"], 98, 49, {"a": 49}), row(["b"], 98, 49, {"b": 49})])
    assert combine.predict(["a", "b"], st) == pytest.approx(0.25, abs=0.01)
    assert st.rho("a", "b") == 1.0


def test_slips_that_come_together_make_a_stack_easier_than_the_product_and_slips_apart_harder():
    together = combine.stats([row(["a", "b"], 40, 20, {"a": 20, "b": 20}, {"solved": 20, "slips": [["a", "b"]] * 20})])
    apart = combine.stats([row(["a", "b"], 40, 0, {"a": 20, "b": 20}, {"solved": 0, "slips": [["a"]] * 20 + [["b"]] * 20})])
    assert together.rho("a", "b") > 1.3 and apart.rho("a", "b") < 0.4
    assert combine.predict(["a", "b"], together) > 0.3 > 0.2 > combine.predict(["a", "b"], apart)


def test_history_from_another_solver_counts_for_less():
    st = combine.stats([row(["a"], 40, 0, {"a": 40}, model="some/older-model")])
    assert st.slips["a"] == pytest.approx(40 * combine.CONFIG.stale_weight)


def test_the_pick_aims_at_the_target_rate_not_at_the_most_slips(history):
    rows = [row([f"mid{i}"], 200, 126, {f"mid{i}": 74}) for i in range(3)]      # handled ~63% of the time
    rows += [row([f"easy{i}"], 200, 190, {f"easy{i}": 10}) for i in range(3)]   # ~95%
    rows += [row([f"wall{i}"], 200, 4, {f"wall{i}": 196}) for i in range(3)]    # ~2%: these slip most
    history(rows)
    modes = [{"id": r["modes"][0]} for r in rows]
    picks = Counter(tuple(sorted(m["id"] for m in combine.pick_modes(modes, 3, random.Random(s)))) for s in range(40))
    assert picks.most_common(1)[0][0] == ("mid0", "mid1", "mid2") and picks[("mid0", "mid1", "mid2")] >= 36
    assert combine.predict(["mid0", "mid1", "mid2"], combine.stats()) == pytest.approx(0.25, abs=0.02)


def test_the_extra_mode_for_a_too_easy_task_is_the_one_that_lands_it_on_target(history):
    history([row(["half"], 200, 100, {"half": 100}), row(["wall"], 200, 4, {"wall": 196}), row(["easy"], 200, 190, {"easy": 10})])
    pool = [{"id": "half"}, {"id": "wall"}, {"id": "easy"}]
    assert {combine.pick_extra(pool, 0.5, random.Random(s))["id"] for s in range(20)} == {"half"}  # 0.5 x 0.5 = the target


def test_backtest_reports_the_model_next_to_the_baseline_it_has_to_beat():
    b = combine.backtest(combine.read_jsonl(combine.PRIOR_HISTORY))
    assert b["n"] == 6 and {"task", "actual", "predicted", "baseline"} <= set(b["rows"][0])
    assert 0 <= b["mae"] <= 1 and 0 <= b["baseline_mae"] <= 1  # no claim here that the model wins: six tasks cannot show it


def test_record_keeps_who_slipped_on_what_together_and_the_first_shot(tmp_path, history):
    history([])
    t = tmp_path / "task"
    (t / "authoring" / "gate").mkdir(parents=True)
    (t / "authoring" / "criteria.json").write_text(json.dumps([
        {"id": "AC1", "mode": "a", "tests": ["a1", "a2"]}, {"id": "AC2", "mode": "b", "tests": ["b1"]}]))
    (t / "authoring" / "design.json").write_text(json.dumps({"modes": ["a", "b"], "predicted_rate": 0.3}))
    (t / "authoring" / "gate" / "difficulty_log.jsonl").write_text(
        json.dumps({"models": {SOLVER: {"solved": 4, "graded": 8}}}) + "\n" + json.dumps({"models": {SOLVER: {"solved": 1, "graded": 4}}}) + "\n")
    (t / "authoring" / "gate" / "hinted.json").write_text(json.dumps({"regret": None, "score": 0.6}))
    diff = {"ok": True, "models": {SOLVER: {"agent": "claude-code", "graded": 4, "solved": 1, "rate": 0.25, "unreproduced": ["t4"],
                                            "failed_tests": {"t1": ["a1", "b1"], "t2": ["a2"]}}}}
    r = calibrate.record(t, diff, rounds=2)
    assert r["by_model"][SOLVER]["patterns"] == {"solved": 1, "slips": [["a", "b"], ["a"], []]}
    assert r["by_model"][SOLVER]["mode_slips"] == {"a": 2, "b": 1} and r["mode_slips"] == {"a": 2, "b": 1}
    assert (r["predicted"], r["first_rate"], r["design_score"], r["kind"]) == (0.3, 0.5, 0.6, "calibrate")
    st = combine.stats()  # and the next design reads it back
    assert st.slips["a"] == 2 and st.handled["a"] == 2 and st.pairs[frozenset(("a", "b"))][0] == 4


def test_a_test_shared_by_two_criteria_is_not_booked_as_two_slips():
    """In a task family one answer key can be changed by two rules. Forgetting one rule fails the
    keys it shares with the other; that is one slip, on the rule whose own keys failed."""
    crit = [{"id": "AC1", "tests": ["a", "ab", "abc"]}, {"id": "AC2", "tests": ["b", "ab", "abc"]}, {"id": "AC3", "tests": ["c", "abc"]}]
    assert gate.slipped_criteria(["a", "ab", "abc"], crit) == {"AC1"}
    assert gate.slipped_criteria(["a", "c", "abc"], crit) == {"AC1", "AC3"}
    assert gate.slipped_criteria(["ab"], crit) == {"AC1", "AC2"}  # only a shared key failed: either rule could be the one
    assert gate.slipped_criteria(["not-in-any-criterion"], crit) == set() == gate.slipped_criteria([], crit)
