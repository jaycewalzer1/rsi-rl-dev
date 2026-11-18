"""The task pool: novelty, admission, re-measuring with a new solver, retirement."""

import itertools
import json
import random
import shutil
from pathlib import Path

import pytest

from rsi import archive, calibrate, combine, gate, llm

ROOT = Path(__file__).resolve().parent.parent
SMOKE = ROOT / "examples" / "tasks" / "smoke-ledger"
SOLVER = "anthropic/claude-opus-5-5"


@pytest.fixture(autouse=True)
def empty_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "ARCHIVE", tmp_path / "archive.jsonl")
    monkeypatch.setattr(combine, "HISTORY", tmp_path / "history.jsonl")


@pytest.fixture
def task(tmp_path):
    t = tmp_path / "runs" / "r" / "task" / "smoke-ledger"
    shutil.copytree(SMOKE, t)
    (t / "authoring" / "design.json").write_text(json.dumps({"modes": ["smoke"], "summary": "A ledger whose transfers need guarding."}))
    return t


def diff(solved, graded, verdict):
    return {"ok": verdict == "in_band", "note": f"{solved}/{graded} {verdict}", "models": {SOLVER: {
        "agent": "claude-code", "graded": graded, "solved": solved, "rate": solved / graded, "verdict": verdict, "failed_tests": {}}}}


def test_distinct_designs_are_far_apart_and_a_light_rewording_is_not():
    designs = combine.example_designs()
    assert max(archive.similarity(a, b) for a, b in itertools.combinations(designs, 2)) < 0.45 < combine.CONFIG.novelty_max
    a = designs[0]
    reworded = {**a, "slug": "copy", "summary": a["summary"].replace("Security levels", "Access tiers"),
                "ticket_rules": [r.replace("item", "ticket") for r in a["ticket_rules"]]}
    assert archive.similarity(a, reworded) > 0.9
    assert archive.nearest(reworded, designs) == (pytest.approx(archive.similarity(a, reworded)), a["slug"])
    assert archive.nearest(a, [a]) == (0.0, None)  # a design is not compared with itself


def test_a_calibrated_task_joins_the_pool_and_is_listed(task):
    row = archive.admit(task, diff(2, 8, "in_band"))
    assert row["status"] == "active" and row["measured"] == {SOLVER: [2, 8]} and "LedgerError" in row["ticket"]
    assert [r["task"] for r in archive.pool("active")] == ["smoke-ledger"] and "active" in archive.listing()
    assert "- smoke-ledger [smoke]: A ledger whose transfers need guarding." in combine.built()


def test_recheck_retires_a_task_the_solver_has_outgrown_and_tells_history(task, monkeypatch):
    archive.admit(task, diff(2, 8, "in_band"))
    monkeypatch.setattr(gate, "run", lambda task_dir, stages=None: True)
    monkeypatch.setattr(gate, "stage_difficulty", lambda task_dir, attempts=None, round_no=0: diff(7, 8, "too_easy"))
    out = archive.recheck()
    assert [(r["task"], r["status"]) for r in out] == [("smoke-ledger", "retired")]
    assert archive.pool("active") == [] and "RETIRED smoke-ledger [smoke]" in archive.saturated_examples()
    assert "Outgrown: do not rebuild it." in combine.examples()  # the author sees it
    rows = combine.read_jsonl(combine.HISTORY)
    assert rows[-1]["kind"] == "recheck" and rows[-1]["solved"] == 7
    assert combine.stats(rows).handled["smoke"] == 8  # the policy now knows today's solver handles this mode


def test_recheck_keeps_a_task_that_still_teaches_and_skips_one_that_is_gone(task, monkeypatch):
    archive.admit(task, diff(2, 8, "in_band"))
    gone = task.parent / "gone"
    archive._append({"task": "gone", "status": "active", "date": "2026-01-01", "path": str(gone), "modes": []})
    monkeypatch.setattr(gate, "run", lambda task_dir, stages=None: True)
    monkeypatch.setattr(gate, "stage_difficulty", lambda task_dir, attempts=None, round_no=0: diff(2, 8, "in_band"))
    assert [(r["task"], r["status"]) for r in archive.recheck()] == [("smoke-ledger", "active")]
    assert {r["task"] for r in archive.pool("active")} == {"smoke-ledger", "gone"}


def test_design_rejects_copies_by_wording_and_duplicates_named_by_the_reviewer(monkeypatch):
    twin = combine.example_designs()[0]
    fresh = {"slug": "parking-permits", "summary": "Permit quotas per garage are stored and never enforced.", "ticket": "Enforce permit quotas.",
             "criteria": [{"id": "AC1", "text": "a full garage refuses a new permit", "surfaces": ["web", "import"]}]}
    authored = iter([{**twin, "slug": "copy-of-it"}, fresh, fresh])
    reviews = iter([{"verdict": "accept", "central_change_found": False, "duplicate_of": "issue-tracker-security-levels"},
                    {"verdict": "accept", "central_change_found": False, "duplicate_of": None}])
    seen = []

    def fake(prompt, model, **k):
        seen.append("author" if model == combine.CONFIG.author_model and "You design one RL coding task" in prompt else "review")
        return next(authored) if seen[-1] == "author" else next(reviews)

    monkeypatch.setattr(llm, "ask_json", fake)
    modes = json.loads((ROOT / "seeds" / "failure_modes.json").read_text())
    d = combine.design(modes, 3, random.Random(0))
    assert seen == ["author", "author", "review", "author", "review"]  # the wording copy never reached the reviewer
    assert d["slug"] == "parking-permits" and d["novelty"]["similarity"] < 0.3 and 0 <= d["predicted_rate"] <= 1
