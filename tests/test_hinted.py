"""The hinted stage (the feasibility proof), with Harbor faked."""

import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from rsi import calibrate, gate, harbor

ROOT = Path(__file__).resolve().parent.parent
SMOKE = ROOT / "examples" / "tasks" / "smoke-ledger"


@pytest.fixture
def task(tmp_path):
    t = tmp_path / "runs" / "r" / "task" / "smoke-ledger"
    shutil.copytree(SMOKE, t)
    (t / "authoring" / "design.json").write_text(json.dumps({
        "modes": ["smoke"], "central_change_risk": "one guard at the top gets close",
        "textbook_build": {"smoke": {"build": "check amount > 0", "hidden_case_it_fails": "a float amount"}}}))
    return t


def cold(task, solved, graded):
    d = task / "authoring" / "gate"
    d.mkdir(parents=True, exist_ok=True)
    (d / "difficulty.json").write_text(json.dumps({"checksum": gate.checksum(task), "models": {
        "m": {"solved": solved, "graded": graded, "failed_tests": {}}}}))


def trial(tmp_path, name, reward, failed=()):
    d = tmp_path / "trials" / name
    (d / "verifier").mkdir(parents=True)
    (d / "verifier" / "ctrf.json").write_text(json.dumps({"results": {"tests": [{"name": n, "status": "failed"} for n in failed]}}))
    return harbor.Trial(name, reward, None, d)


def test_hint_names_every_criterion_and_the_trap_but_no_test(task):
    hint = gate.hint_text(task)
    for c in json.loads((task / "authoring" / "criteria.json").read_text()):
        assert c["sentence"] in hint
    assert "a float amount" in hint and "one guard at the top gets close" in hint
    assert "test_" not in hint


def test_skipped_when_a_cold_solve_exists_or_too_few_attempts(task, monkeypatch):
    monkeypatch.setattr(harbor, "run_job", lambda *a, **k: pytest.fail("no hinted attempts should be spent"))
    cold(task, 2, 8)
    r = gate.stage_hinted(task)
    assert r["ok"] and r["skipped"] and r["regret"] is None and "cold solve" in r["note"]
    cold(task, 0, 3)
    assert gate.stage_hinted(task)["skipped"]


def test_feasible_when_the_hint_closes_the_gap(task, tmp_path, monkeypatch):
    cold(task, 0, 8)
    seen = {}

    def fake(task_dir, agent, **kw):
        seen["ticket"] = (task_dir / "instruction.md").read_text()
        return [trial(tmp_path, "h1", 1.0), trial(tmp_path, "h2", 0.0, ["test_overdraft_refused"]),
                trial(tmp_path, "h3", 1.0), trial(tmp_path, "h4", 0.0, ["test_overdraft_refused"])]

    monkeypatch.setattr(harbor, "run_job", fake)
    r = gate.stage_hinted(task)
    assert "REVIEWER NOTES" in seen["ticket"] and "REVIEWER NOTES" not in (task / "instruction.md").read_text()
    assert r["ok"] and r["verdict"] == "feasible" and (r["solved"], r["graded"]) == (2, 4)
    assert r["regret"] == 0.5 and r["anchor"] == pytest.approx(1 - 0.25 / 0.75, abs=1e-3)
    monkeypatch.setattr(harbor, "run_job", lambda *a, **k: pytest.fail("same task: the measurement is reused"))
    assert gate.stage_hinted(task)["solved"] == 2


def test_unproven_when_even_hinted_attempts_all_fail(task, tmp_path, monkeypatch):
    cold(task, 0, 8)
    monkeypatch.setattr(harbor, "run_job", lambda *a, **k: [
        trial(tmp_path, "h1", 0.0, ["test_overdraft_refused", "test_bad_amount_refused[1.5]"]),
        trial(tmp_path, "h2", 0.0, ["test_overdraft_refused"]), trial(tmp_path, "h3", 0.0, ["test_overdraft_refused", "x"])])
    r = gate.stage_hinted(task)
    assert not r["ok"] and r["verdict"] == "unproven" and r["always_failed"] == ["test_overdraft_refused"]
    assert r["regret"] == 0.0


def test_design_score_peaks_in_band_with_a_large_hint_gap():
    in_band, too_easy = gate.design_score(0.25, 0.9), gate.design_score(0.9, 0.9)
    assert in_band["score"] == pytest.approx(0.4 * 0.65 + 0.6) and in_band["score"] > too_easy["score"]
    assert gate.design_score(0.0, 0.0)["score"] < gate.design_score(0.0, 0.75)["score"]


def test_calibrate_sends_an_unproven_task_back_with_the_tests_nobody_passed(task, monkeypatch):
    sent = []
    monkeypatch.setattr(calibrate, "CONFIG", dataclasses.replace(calibrate.CONFIG, max_calibration_rounds=1))
    monkeypatch.setattr(gate, "run", lambda task_dir, stages=None: True)
    monkeypatch.setattr(gate, "stage_difficulty", lambda task_dir, round_no=0: (task / "authoring" / "gate").mkdir(parents=True, exist_ok=True) or (
        task / "authoring" / "gate" / "difficulty.json").write_text(json.dumps({"checksum": "c", "ok": False, "models": {
            "m": {"agent": "a", "attempts": 8, "graded": 8, "voided": 0, "solved": 0, "rate": 0.0, "verdict": "too_hard", "failed_tests": {}}}})))
    monkeypatch.setattr(gate, "stage_hinted", lambda task_dir: {"ok": False, "verdict": "unproven", "always_failed": ["test_overdraft_refused"], "graded": 4})
    monkeypatch.setattr(gate, "stage_inspect", lambda task_dir: {"unfair_tests": []})
    monkeypatch.setattr(calibrate.build, "revise", lambda task_dir, measurement, instruction: sent.append(instruction))
    monkeypatch.setattr(calibrate, "record", lambda *a, **k: None)
    assert calibrate.run(task) is False
    assert len(sent) == 1 and "unproven, not hard" in sent[0] and "test_overdraft_refused" in sent[0]
