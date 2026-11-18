"""Reading failed attempts for the slips they made (judge and Harbor output faked)."""

import json
import shutil
from pathlib import Path

import pytest

from rsi import calibrate, combine, llm, mine, papers

ROOT = Path(__file__).resolve().parent.parent
SMOKE = ROOT / "examples" / "tasks" / "smoke-ledger"
PATCH_A = "diff --git a/ledger.py b/ledger.py\n+        if amount <= 0:\n+            raise LedgerError('amount must be positive')\n"
PATCH_B = "diff --git a/ledger.py b/ledger.py\n+        if not isinstance(amount, int):\n+            raise LedgerError('bad amount')\n"


@pytest.fixture(autouse=True)
def files(tmp_path, monkeypatch):
    monkeypatch.setattr(mine, "MINED", tmp_path / "mined.jsonl")
    monkeypatch.setattr(mine, "MODES", tmp_path / "mined_modes.json")
    monkeypatch.setattr(combine, "HISTORY", tmp_path / "history.jsonl")


@pytest.fixture
def task(tmp_path):
    t = tmp_path / "runs" / "r" / "task" / "smoke-ledger"
    shutil.copytree(SMOKE, t)
    job = tmp_path / "runs" / "r" / "jobs" / "smoke-ledger" / "gate-difficulty-r0-opus"
    for name, patch in [("t1", PATCH_A), ("t2", PATCH_A), ("t3", PATCH_B), ("t4", None)]:
        d = job / f"dir-{name}"
        (d / "verifier").mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({"trial_name": name, "verifier_result": {"rewards": {"reward": 0.0}}}))
        if patch:
            (d / "verifier" / "agent.patch").write_text(patch)
    (t / "authoring" / "gate").mkdir(parents=True)
    (t / "authoring" / "gate" / "difficulty.json").write_text(json.dumps({"models": {"anthropic/claude-opus-5-5": {
        "failed_tests": {"t1": ["test_bad_amount_refused[1.5]"], "t2": ["test_bad_amount_refused[1.5]"],
                         "t3": ["test_overdraft_refused"], "t4": ["test_overdraft_refused"], "t5": []},
        "failure_messages": {"t1": {"test_bad_amount_refused[1.5]": "DID NOT RAISE LedgerError"}}}}}))
    return t


def reading(criterion, quote, kind="documented", other=None):
    return {"criterion": criterion, "quotes": [quote], "slip": f"slip on {criterion}", "kind": kind, "other": other}


def test_only_failed_attempts_that_recorded_a_patch_are_read(task):
    got = mine.attempts(task)
    assert [a["trial"] for a in got] == ["t1", "t2", "t3"]  # t4 recorded no patch, t5 failed nothing
    assert got[0]["messages"] == {"test_bad_amount_refused[1.5]": "DID NOT RAISE LedgerError"}


def test_a_reading_must_quote_the_patch(task, monkeypatch):
    prompts = []
    monkeypatch.setattr(llm, "ask_json", lambda prompt, model, **k: prompts.append(prompt) or [
        reading("AC1", "+        if amount <= 0:"), reading("AC1", "if type(amount) is not int:"), reading("AC9", "if amount <= 0:")])
    rows = mine.read_attempt(task, mine.attempts(task)[0])
    assert [(r["criterion"], r["mode"], r["kind"], r["quotes"]) for r in rows] == [("AC1", "smoke", "documented", ["+        if amount <= 0:"])]
    assert "DID NOT RAISE LedgerError" in prompts[0] and "AC3" not in prompts[0].split("FAILED CRITERIA")[1].split("For each failed")[0]


def test_each_distinct_failure_is_read_once_and_filed(task, monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "ask_json", lambda prompt, model, **k: calls.append(1) or [
        reading("AC1" if "amount <= 0" in prompt.split("THE ATTEMPT")[1] else "AC3", "raise LedgerError", "other",
                {"id": "type-check-skipped", "mode": "validates the value and not its type", "textbook_build": "compare", "hidden_case": "a float"})])
    rows = mine.run(task)
    assert len(calls) == 2  # t2 wrote the same patch as t1 and failed the same tests: one reading serves both
    assert [(r["trial"], r["criterion"]) for r in rows] == [("t1", "AC1"), ("t2", "AC1"), ("t3", "AC3")]
    assert mine.run(task) == [] and len(calls) == 2  # everything is filed; nothing is read twice
    assert len({(r["task"], r["trial"]) for r in mine.read_jsonl(mine.MINED)}) == 3


def test_a_new_mode_needs_two_attempts_and_a_new_name(task, monkeypatch):
    other = {"id": "type-check-skipped", "mode": "validates the value and not its type", "textbook_build": "compare", "hidden_case": "a float"}
    rows = [{"task": "a", "trial": "t1", "kind": "other", "slip": "s1", "quotes": ["q1"], "other": other},
            {"task": "b", "trial": "t7", "kind": "other", "slip": "s2", "quotes": ["q2"], "other": other},
            {"task": "b", "trial": "t8", "kind": "other", "slip": "s3", "quotes": ["q3"], "other": {**other, "id": "lonely"}},
            {"task": "b", "trial": "t9", "kind": "documented", "slip": "s4", "quotes": ["q4"], "other": None}]
    mine.MINED.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(llm, "ask_json", lambda *a, **k: [
        {"id": "type-check-skipped", "mode": "Validates a value's range and not its type.", "textbook_build": "x", "hidden_case": "y", "members": [0, 1]},
        {"id": "lonely", "mode": "seen once", "members": [2]}, {"id": "authz-by-id", "mode": "a known one", "members": [0, 1]}])
    known = papers.merge_seeds([])
    modes = mine.promote(known)
    assert [m["id"] for m in modes] == ["type-check-skipped"] and modes[0]["origin"] == "rollout"
    assert modes[0]["evidence"] == "read from 2 failed attempts on 2 task(s) in this factory" and modes[0]["rollouts"][0]["quote"] == "q1"
    pool = {m["id"]: m for m in papers.merge_seeds([])}
    assert pool["type-check-skipped"]["tier"] == "measured" and pool["early-stop"]["tier"] == "paper"


def test_one_attempt_is_not_enough_to_ask(monkeypatch):
    mine.MINED.write_text(json.dumps({"task": "a", "trial": "t1", "kind": "other", "slip": "s", "quotes": ["q"], "other": {"id": "x"}}) + "\n")
    monkeypatch.setattr(llm, "ask_json", lambda *a, **k: pytest.fail("no grouping call without two attempts"))
    assert mine.promote([]) == []


def test_evidence_tiers():
    assert papers.tier({"id": "m", "papers": [{}], "complaints": [{"provenance": "hn"}]}) == "paper+field"
    assert papers.tier({"id": "m", "papers": [], "complaints": [{"provenance": "web_snippet"}]}) == "snippet"
    assert papers.tier({"id": "m", "papers": [], "complaints": [{"provenance": "web_snippet"}]}, measured={"m"}) == "measured"


def test_a_reading_error_never_costs_a_calibration(task, monkeypatch, capsys):
    monkeypatch.setattr(mine, "run", lambda task_dir: 1 / 0)
    calibrate._mine(task)
    assert "mine: skipped (ZeroDivisionError" in capsys.readouterr().out
