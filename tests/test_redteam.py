"""Replay and the red-team stage, with Harbor and the judge faked (no Docker, no model)."""

import json
import shutil
from pathlib import Path

import pytest

from rsi import gate, harbor, llm

ROOT = Path(__file__).resolve().parent.parent
SMOKE = ROOT / "examples" / "tasks" / "smoke-ledger"
PATCH = "diff --git a/ledger.py b/ledger.py\n--- a/ledger.py\n+++ b/ledger.py\n@@ -1 +1,2 @@\n+        if amount == 3: raise LedgerError('x')\n"


@pytest.fixture
def task(tmp_path):
    t = tmp_path / "runs" / "r" / "task" / "smoke-ledger"
    shutil.copytree(SMOKE, t)
    return t


def trial(tmp_path, name, reward, patch=None, exc=None):
    d = tmp_path / "trials" / name
    (d / "verifier").mkdir(parents=True)
    if patch is not None:
        (d / "verifier" / "agent.patch").write_text(patch)
    return harbor.Trial(name, reward, exc, d)


class FakeHarbor:
    """run_job stand-in: solver jobs get the queued trials, oracle (replay) jobs the replay reward."""

    def __init__(self, tmp_path, solver_trials, replay_reward=1.0):
        self.tmp, self.solver_trials, self.replay_reward = tmp_path, list(solver_trials), replay_reward
        self.calls = []

    def __call__(self, task_dir, agent, *, model=None, attempts=1, jobs_dir, job_name=None, extra=None):
        self.calls.append({"agent": agent, "job": job_name, "ticket": (task_dir / "instruction.md").read_text(),
                           "solve": (task_dir / "solution" / "solve.sh").read_text()})
        if agent == "oracle":
            return [trial(self.tmp, f"replay-{len(self.calls)}", self.replay_reward)]
        return [self.solver_trials.pop(0) for _ in range(attempts)]


def test_replay_task_applies_only_the_recorded_patch(task, tmp_path):
    rt = gate.replay_task(task, PATCH, tmp_path / "out", "t1")
    solve = (rt / "solution" / "solve.sh").read_text()
    assert "agent.patch" in solve and "reference.patch" not in solve
    assert (rt / "solution" / "agent.patch").read_text() == PATCH
    assert not (rt / "authoring" / "gate").exists()


def test_replay_outcomes(task, tmp_path, monkeypatch):
    fake = FakeHarbor(tmp_path, [], replay_reward=0.0)
    monkeypatch.setattr(harbor, "run_job", fake)
    assert gate.replay(task, trial(tmp_path, "no-patch", 1.0)) is None  # the task records nothing
    assert gate.replay(task, trial(tmp_path, "empty", 1.0, patch="\n")) == 0.0 and not fake.calls  # no code changed
    assert gate.replay(task, trial(tmp_path, "tampered", 1.0, patch=PATCH)) == 0.0 and len(fake.calls) == 1
    fake.replay_reward = None  # the replay itself errored: no verdict
    assert gate.replay(task, trial(tmp_path, "infra", 1.0, patch=PATCH)) is None


@pytest.mark.parametrize("reward,patch,replay_reward,verdict,evidence,want", [
    (0.0, PATCH, 1.0, None, None, "held"),
    (1.0, None, 1.0, None, None, "unreplayed"),
    (1.0, PATCH, 0.0, None, None, "tamper"),
    (1.0, PATCH, 1.0, "shortcut", ["if amount == 3: raise LedgerError('x')"], "shortcut"),
    (1.0, PATCH, 1.0, "implementation", ["if amount == 3: raise LedgerError('x')"], "implemented"),
    (1.0, PATCH, 1.0, "implementation", ["a line that is not in the patch"], "shortcut"),  # unverifiable praise
])
def test_shortcut_outcomes(task, tmp_path, monkeypatch, reward, patch, replay_reward, verdict, evidence, want):
    monkeypatch.setattr(harbor, "run_job", FakeHarbor(tmp_path, [], replay_reward))
    monkeypatch.setattr(llm, "ask_json", lambda *a, **k: {"verdict": verdict, "evidence": evidence, "why": "w"})
    assert gate.shortcut_outcome(task, trial(tmp_path, "t", reward, patch))["outcome"] == want


def test_shortcut_stage_runs_every_strategy_and_blocks_on_a_shortcut(task, tmp_path, monkeypatch):
    fake = FakeHarbor(tmp_path, [trial(tmp_path, "a", 0.0, PATCH), trial(tmp_path, "b", 1.0, PATCH), trial(tmp_path, "c", 0.0, PATCH)])
    monkeypatch.setattr(harbor, "run_job", fake)
    monkeypatch.setattr(llm, "ask_json", lambda *a, **k: {"verdict": "shortcut", "evidence": ["if amount == 3: raise LedgerError('x')"]})
    r = gate.stage_shortcut(task)
    solver_calls = [c for c in fake.calls if c["agent"] != "oracle"]
    assert [c["job"].split("-")[2] for c in solver_calls] == ["exploit", "please", "hard"]
    assert all("RED-TEAM MODE" in c["ticket"] for c in solver_calls)
    assert "- test_overdraft_refused" in solver_calls[1]["ticket"] and "test_overdraft_refused" not in solver_calls[0]["ticket"]
    assert not r["ok"] and "PASSED WITHOUT THE FEATURE: please-hack" in r["note"]
    assert "RED-TEAM" not in (task / "instruction.md").read_text()  # the task itself is untouched


def test_difficulty_counts_only_passes_that_survive_replay(task, tmp_path, monkeypatch):
    trials = [trial(tmp_path, "t1", 1.0, PATCH), trial(tmp_path, "t2", 1.0, PATCH), trial(tmp_path, "t3", 0.0, PATCH),
              trial(tmp_path, "t4", None, exc="RateLimit")]
    rewards = iter([1.0, 0.0])

    def fake(task_dir, agent, **kw):
        return [trial(tmp_path, f"r{kw['job_name']}", next(rewards))] if agent == "oracle" else trials

    monkeypatch.setattr(harbor, "run_job", fake)
    m = next(iter(gate.stage_difficulty(task, attempts=4)["models"].values()))
    assert (m["graded"], m["solved"], m["voided"], m["unreproduced"]) == (3, 1, 1, ["t2"])
    assert "1 passes did not survive replay" in json.loads((task / "authoring" / "gate" / "difficulty.json").read_text())["note"]
