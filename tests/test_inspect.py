"""Fairness review: repeated reviews, the verbatim-quote check, and what the attempts show."""

import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from rsi import gate, harbor, llm

ROOT = Path(__file__).resolve().parent.parent
SMOKE = ROOT / "examples" / "tasks" / "smoke-ledger"
SENTENCE = "A transfer that would take the source balance below zero is refused."


@pytest.fixture
def task(tmp_path, monkeypatch):
    t = tmp_path / "smoke-ledger"
    shutil.copytree(SMOKE, t)
    monkeypatch.setattr(gate, "CONFIG", dataclasses.replace(gate.CONFIG, inspect_reviews=2))
    return t


def measured(task, failed_tests, messages=None, always_failed=None):
    d = task / "authoring" / "gate"
    d.mkdir(parents=True, exist_ok=True)
    (d / "difficulty.json").write_text(json.dumps({"checksum": "c1", "models": {"m": {
        "failed_tests": failed_tests, "failure_messages": messages or {}}}}))
    if always_failed is not None:
        (d / "hinted.json").write_text(json.dumps({"checksum": "c1", "always_failed": always_failed}))


def test_divergent_tests_are_the_ones_everyone_fails_differently():
    failed = {"t1": ["a", "b"], "t2": ["a"], "t3": ["a", "c"], "t4": []}
    msgs = {"t1": {"a": "got 3"}, "t2": {"a": "KeyError"}, "t3": {"a": "got None"}}
    assert gate.divergent_tests(failed, msgs) == ["a"]
    assert gate.divergent_tests(failed, {**msgs, "t3": {"a": "got 3"}}) == []  # two attempts agree: a shared misreading
    assert gate.divergent_tests(failed, {"t1": {"a": "got 3"}}) == []  # no messages, no claim
    assert gate.divergent_tests({"t1": ["a"], "t2": ["a"]}, msgs) == []  # too few attempts


def test_each_distinct_failure_is_reviewed_k_times_and_one_unfair_vote_is_enough(task, monkeypatch):
    measured(task, {"t1": ["x"], "t2": ["x"], "t3": ["x", "y"], "t4": []})
    answers = iter([
        {"tests": [{"test": "x", "sentence": SENTENCE, "fair": True}]},
        {"tests": [{"test": "x", "sentence": SENTENCE, "fair": True}]},
        {"tests": [{"test": "x", "sentence": SENTENCE, "fair": True}, {"test": "y", "sentence": SENTENCE, "fair": True}]},
        {"tests": [{"test": "x", "sentence": SENTENCE, "fair": True}, {"test": "y", "sentence": None, "fair": False}]},
    ])
    calls = []
    monkeypatch.setattr(llm, "ask_json", lambda prompt, model, **k: calls.append(prompt) or next(answers))
    r = gate.stage_inspect(task)
    assert len(calls) == 4  # two distinct failing sets x two reviews, not one call per attempt
    assert r["unfair_tests"] == ["y"] and r["unfair_votes"] == {"y": 1} and not r["ok"]


def test_a_quote_that_is_not_in_the_ticket_is_unfair_whatever_the_judge_says(task, monkeypatch):
    measured(task, {"t1": ["x"]})
    monkeypatch.setattr(llm, "ask_json", lambda *a, **k: {"tests": [{"test": "x", "sentence": "Overdrafts are refused.", "fair": True}]})
    assert gate.stage_inspect(task)["unfair_tests"] == ["x"]


def test_what_the_attempts_did_is_put_in_front_of_the_judge(task, monkeypatch):
    measured(task, {"t1": ["x", "z"], "t2": ["x"], "t3": ["x"]},
             messages={"t1": {"x": "got 1"}, "t2": {"x": "got 2"}, "t3": {"x": "got 3"}}, always_failed=["z"])
    prompts = []
    monkeypatch.setattr(llm, "ask_json", lambda prompt, model, **k: prompts.append(prompt) or {"tests": [{"test": "x", "sentence": SENTENCE, "fair": True}]})
    r = gate.stage_inspect(task)
    assert r["suspect_tests"] == {"x": "every attempt failed it, each with a different error", "z": "every hinted attempt failed it too"}
    with_z = next(p for p in prompts if "'z'" in p)
    assert "- z: every hinted attempt failed it too" in with_z and "- x: every attempt failed it, each with a different error" in with_z
    assert r["ok"]  # flags inform the judge; only its verdict or a bad quote makes a test unfair


def test_failure_messages_come_from_the_ctrf_report(tmp_path):
    d = tmp_path / "t"
    (d / "verifier").mkdir(parents=True)
    (d / "verifier" / "ctrf.json").write_text(json.dumps({"results": {"tests": [
        {"name": "a", "status": "failed", "message": "assert 3 == 4\n  where 3 = f()"},
        {"name": "b", "status": "failed", "trace": "Traceback...\nKeyError: 'k'"}, {"name": "c", "status": "passed"}]}}))
    assert harbor.failure_messages(harbor.Trial("t", 0.0, None, d)) == {"a": "assert 3 == 4", "b": "KeyError: 'k'"}
