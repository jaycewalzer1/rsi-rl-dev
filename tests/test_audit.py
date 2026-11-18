"""Fixtures for the static audit: a clean task stays clean, each known leak keeps firing."""

import json
import shutil
from pathlib import Path

import pytest

from rsi import audit, gate

ROOT = Path(__file__).resolve().parent.parent
SMOKE = ROOT / "examples" / "tasks" / "smoke-ledger"


@pytest.fixture
def task(tmp_path):
    t = tmp_path / "smoke-ledger"
    shutil.copytree(SMOKE, t)
    return t


def blocks(t):
    return {f["check"] for f in audit.audit(t) if f["severity"] == "block"}


def test_shipped_task_is_clean(task):
    assert audit.audit(task) == []
    assert gate.stage_audit(task)["ok"]


def test_dockerfile_copying_hidden_folders_is_blocked(task):
    df = task / "environment" / "Dockerfile"
    df.write_text(df.read_text().replace("COPY app/ /app/", "COPY app/ /app/\nCOPY ../tests /app/tests"))
    assert "dockerfile-copies-hidden" in blocks(task)


def test_hidden_test_file_inside_the_image_is_blocked(task):
    shutil.copy(task / "tests" / "test_ledger.py", task / "environment" / "app" / "check.py")
    assert "hidden-copy" in blocks(task)


def test_starting_repo_that_already_holds_the_reference_is_blocked(task):
    ledger = task / "environment" / "app" / "ledger.py"
    added = [l[1:] for l in (task / "solution" / "reference.patch").read_text().splitlines() if l.startswith("+") and not l.startswith("+++")]
    ledger.write_text(ledger.read_text() + "\n# leftover from an earlier build\n" + "\n".join(added) + "\n")
    assert "reference-leak" in blocks(task)


def test_ticket_that_points_at_the_grader_is_blocked(task):
    ins = task / "instruction.md"
    ins.write_text(ins.read_text() + "\nRun /tests/test_ledger.py; test_unknown_account_refused must pass.\n")
    assert {"ticket-leak", "test-name-leak"} <= blocks(task)


@pytest.mark.parametrize("script,check", [
    ("#!/bin/bash\ncd /tests\npytest /tests/test_ledger.py | tee /logs/verifier/out.txt\nif [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi\n", "verifier-masked-status"),
    ("#!/bin/bash\npytest /tests/test_ledger.py || true\nif [ $? -eq 0 ]; then echo 1 > /logs/verifier/reward.txt; fi\n", "verifier-masked-status"),
    ("#!/bin/bash\nif uvx pytest /tests/test_ledger.py; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi\n", "verifier-downloads"),
    ("#!/bin/bash\npytest /tests/test_ledger.py\necho 1 > /logs/verifier/reward.txt\n", "verifier-unconditional"),
    ("#!/bin/bash\npytest /tests/test_ledger.py\n", "verifier-no-reward"),
])
def test_broken_verifiers_are_blocked(task, script, check):
    (task / "tests" / "test.sh").write_text(script)
    assert check in blocks(task)


def test_pipefail_makes_a_pipe_safe(task):
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\nset -o pipefail\nif pytest /tests/test_ledger.py | tee /logs/verifier/out.txt; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi\n")
    assert not blocks(task)


def test_tests_that_assert_nothing_are_blocked(task):
    (task / "tests" / "test_extra.py").write_text("def test_runs():\n    import ledger\n    ledger.Ledger({})\n")
    assert "test-asserts-nothing" in blocks(task)


def test_shape_wants_every_criterion_sentence_word_for_word(task):
    crit = task / "authoring" / "criteria.json"
    rows = json.loads(crit.read_text())
    assert "word for word" not in gate.stage_shape(task)["note"]
    rows[0]["sentence"] = "Transfers with a bad amount are refused."
    crit.write_text(json.dumps(rows))
    assert "AC1 sentence is not in the ticket word for word" in gate.stage_shape(task)["note"]
