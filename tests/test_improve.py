"""The improvement loop's shell: the protected list, the tamper lint, the scorecard, and whole
generations run in a throwaway git repository with the proposing session faked."""

import json
import subprocess
from pathlib import Path

import pytest

from rsi import improve, meter, proposer

NOTE = "# Aim designs lower\n\n## Evidence\nfirst_shot_error 0.3\n\n## Change\nx\n\n## Prediction\nyield up\n\n## How this could be wrong\nyield flat\n"


# ---------------------------------------------------------------- protected list and lint

@pytest.mark.parametrize("path,protected", [
    ("rsi/gate.py", True), ("rsi/meter.py", True), ("rsi/improve.py", True), ("rsi/families/shoe_store.py", True),
    ("tests/test_core.py", True), ("examples/frist-hackathon/history.jsonl", True), ("docs/TASK_CONTRACT.md", True), ("uv.lock", True),
    ("rsi/combine.py", False), ("rsi/proposer.py", False), ("rsi/audit.py", False), ("tests/gen/test_new.py", False),
    ("docs/DIFFICULTY_RULES.md", False), ("docs/improvements/gen-3.md", False), ("seeds/failure_modes.json", False),
])
def test_protected_paths(path, protected):
    assert improve.is_protected(path) is protected


def test_the_real_protected_files_exist():
    """A protected path that names nothing protects nothing."""
    root = Path(improve.__file__).resolve().parent.parent
    for p in improve.PROTECTED:
        assert list(root.glob(p)), p


def diff(path, *added):
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1,9 @@\n" + "".join(f"+{line}\n" for line in added)


@pytest.mark.parametrize("line", [
    'gate.band_verdict = lambda solved, graded: "in_band"', "setattr(gate, 'wilson', fake)", 'object.__setattr__(CONFIG, "band_high", 0.9)',
    'os.environ["RSI_BAND_HIGH"] = "0.9"', "importlib.reload(gate)", '(task_dir / "authoring" / "gate" / "difficulty.json").write_text(text)',
    "CONFIG.target_rate = 0.5",
])
def test_lines_that_reach_around_protected_code_are_caught(line):
    assert improve.tamper_lint(diff("rsi/calibrate.py", line))
    assert not improve.tamper_lint(diff("tests/gen/test_x.py", line))  # tests may monkeypatch; only rsi/ is linted


def test_ordinary_lines_are_not_flagged():
    assert improve.tamper_lint(diff("rsi/combine.py", "lo, hi = gate.wilson(k, n)", "if gate.band_verdict(k, n) == 'in_band':",
                                    "verdict = llm.ask_json(prompt, CONFIG.judge_model)", "SPREAD_WEIGHT = 0.2")) == []


# ---------------------------------------------------------------- the scorecard

def measured(run, name, log, predicted=None, unfair=None, shortcut_ok=None, hinted=None):
    g = run / "task" / name / "authoring" / "gate"
    g.mkdir(parents=True)
    (g / "difficulty_log.jsonl").write_text("".join(json.dumps({"checksum": c, "models": {"m": {"solved": k, "graded": n}}}) + "\n" for c, k, n in log))
    (g.parent / "design.json").write_text(json.dumps({"predicted_rate": predicted}))
    if unfair is not None:
        (g / "inspect.json").write_text(json.dumps({"unfair_tests": unfair}))
    if shortcut_ok is not None:
        (g / "shortcut.json").write_text(json.dumps({"ok": shortcut_ok}))
    if hinted:
        (g / "hinted.json").write_text(json.dumps({"verdict": hinted}))


def test_scorecard_reads_only_gate_output(tmp_path):
    run = tmp_path / "r1"
    measured(run, "a", [("c1", 2, 8)], predicted=0.3, unfair=[], shortcut_ok=True)                       # in band first time
    measured(run, "b", [("c1", 6, 8), ("c2", 1, 4), ("c2", 1, 4)], predicted=0.25, unfair=["t"], shortcut_ok=True)  # revised, pooled 2/8
    measured(run, "c", [("c1", 0, 8)], predicted=0.2, shortcut_ok=False, hinted="unproven")              # nobody solved it
    (run / "task" / "never-measured" / "authoring").mkdir(parents=True)
    card = meter.scorecard([run])
    assert (card["tasks"], card["in_band"], card["attempts"]) == (3, 2, 32)
    assert card["efficiency"] == 6.25 and card["yield"] == 0.667 and card["attempts_per_hit"] == 16.0
    assert card["first_shot_error"] == pytest.approx((0 + 0.5 + 0.25) / 3, abs=1e-3)
    assert card["prediction_error"] == pytest.approx((0.05 + 0.5 + 0.2) / 3, abs=1e-3)
    assert (card["unfair_rate"], card["hack_rate"], card["unproven_rate"]) == (0.5, 0.333, 1.0)


def card(efficiency, tasks=4, **guardrails):
    return {"tasks": tasks, "efficiency": efficiency, "unfair_rate": 0.0, "hack_rate": 0.0, "unproven_rate": 0.0, **guardrails}


def test_a_candidate_is_accepted_only_if_more_efficient_with_no_guardrail_worse():
    assert meter.compare(card(5.0), card(8.0))["accept"]
    assert not meter.compare(card(5.0), card(5.0))["accept"]  # the same is not better
    worse = meter.compare(card(5.0), card(9.0, unfair_rate=0.25))
    assert not worse["accept"] and "unfair_rate went from 0.0 to 0.25" in worse["reasons"]
    assert not meter.compare(card(5.0), card(None, tasks=0))["accept"]


# ---------------------------------------------------------------- whole generations

@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    for rel, text in {"rsi/build.py": "PROMPT = 'build it'\n", "rsi/gate.py": "BAND = (0.15, 0.35)\n", "rsi/combine.py": "X = 1\n",
                      "tests/test_core.py": "def test_ok():\n    assert True\n", ".gitignore": "runs/\n"}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    for k, v in {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}.items():
        monkeypatch.setenv(k, v)
    for args in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-qm", "base"]):
        subprocess.run(["git", *args], cwd=root, check=True)
    monkeypatch.setattr(improve, "ROOT", root)
    monkeypatch.setattr(improve, "IMPROVE", root / "runs" / "improve")
    monkeypatch.setattr(meter, "scorecard", lambda dirs: {"tasks": 5, "efficiency": 4.0, "by_task": []})
    monkeypatch.setattr(proposer, "evidence", lambda runs_dir: {"scorecard": {}})
    return root


def generation(monkeypatch, edits, **kw):
    def fake(tree, n, ev, lineage, protected, budget_usd=None, model=None):
        for rel, text in edits.items():
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            (tree / rel).write_text(text)
        return "done"
    monkeypatch.setattr(proposer, "propose", fake)
    return improve.run(test_cmd=kw.pop("test_cmd", ["true"]), **kw)


def branches(repo):
    return subprocess.run(["git", "branch", "--list", "rsi/*"], cwd=repo, capture_output=True, text=True).stdout.split()


def test_a_sound_generation_is_committed_on_its_own_branch_and_nothing_else_moves(repo, monkeypatch):
    row = generation(monkeypatch, {"rsi/build.py": "PROMPT = 'build it, and name the textbook build first'\n",
                                   "tests/gen/test_build.py": "def test_x():\n    assert True\n", "docs/improvements/gen-1.md": NOTE})
    assert row["status"] == "validated-offline" and row["title"] == "Aim designs lower"
    assert [c["check"] for c in row["checks"]] == ["protected", "tamper", "note", "tests"] and all(c["ok"] for c in row["checks"])
    assert branches(repo) == ["rsi/gen-1"]
    shown = subprocess.run(["git", "show", "rsi/gen-1:rsi/build.py"], cwd=repo, capture_output=True, text=True).stdout
    assert "textbook build first" in shown and (repo / "rsi" / "build.py").read_text() == "PROMPT = 'build it'\n"  # main is untouched
    assert not (repo / "runs" / "improve" / "gen-1" / "tree").exists()
    assert [g["gen"] for g in improve.lineage()] == [1] and "validated-offline" in improve.log()


@pytest.mark.parametrize("edits,test_cmd,status", [
    ({"rsi/gate.py": "BAND = (0.0, 1.0)\n", "docs/improvements/gen-1.md": NOTE}, ["true"], "rejected: protected"),
    ({"tests/test_core.py": "def test_ok():\n    pass\n", "docs/improvements/gen-1.md": NOTE}, ["true"], "rejected: protected"),
    ({"rsi/build.py": "from . import gate\ngate.BAND = (0.0, 1.0)\n", "docs/improvements/gen-1.md": NOTE}, ["true"], "rejected: tamper"),
    ({"rsi/build.py": "PROMPT = 'new'\n"}, ["true"], "rejected: note"),
    ({"rsi/build.py": "PROMPT = 'new'\n", "docs/improvements/gen-1.md": "# t\n\n## Evidence\ne\n"}, ["true"], "rejected: note"),
    ({"rsi/build.py": "PROMPT = 'new'\n", "docs/improvements/gen-1.md": NOTE}, ["false"], "rejected: tests"),
    ({}, ["true"], "no-change"),
    ({"docs/improvements/gen-1.md": NOTE.replace("Aim designs lower", "Not enough data yet")}, ["true"], "no-change"),
])
def test_a_generation_that_breaks_a_rule_leaves_a_patch_and_no_branch(repo, monkeypatch, edits, test_cmd, status):
    row = generation(monkeypatch, edits, test_cmd=test_cmd)
    assert row["status"] == status and row["branch"] is None and branches(repo) == []
    patch = (repo / "runs" / "improve" / "gen-1" / "change.patch").read_text()
    assert all(rel in patch for rel in edits)
    assert subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout == ""


def test_a_change_to_the_design_policy_must_not_predict_worse(repo, monkeypatch):
    edits = {"rsi/combine.py": "X = 2\n", "docs/improvements/gen-1.md": NOTE}
    monkeypatch.setattr(improve.combine, "backtest", lambda rows=None: {"mae": 0.20})
    monkeypatch.setattr(improve, "candidate_backtest", lambda tree, rows: 0.30)
    assert generation(monkeypatch, edits)["status"] == "rejected: backtest"
    monkeypatch.setattr(improve, "candidate_backtest", lambda tree, rows: 0.21)
    assert generation(monkeypatch, {**edits, "docs/improvements/gen-2.md": NOTE})["status"] == "validated-offline"


def test_without_measured_tasks_there_is_nothing_to_improve_on(repo, monkeypatch):
    monkeypatch.setattr(meter, "scorecard", lambda dirs: {"tasks": 1, "by_task": []})
    monkeypatch.setattr(proposer, "propose", lambda *a, **k: pytest.fail("no session without evidence"))
    assert improve.run()["status"] == "no-evidence" and branches(repo) == [] and improve.lineage() == []


def test_live_verdict_decides_and_adopt_fast_forwards_only_an_accepted_generation(repo, monkeypatch):
    edits = {"rsi/build.py": "PROMPT = 'v2'\n", "docs/improvements/gen-1.md": NOTE}
    monkeypatch.setattr(improve, "live", lambda gen_dir, parent_tree, tree, tasks, seed: {
        "tasks_per_arm": tasks, "verdict": {"accept": False, "reasons": ["efficiency 3.0 is not above the parent's 4.0"]}})
    row = generation(monkeypatch, edits, live_tasks=2, adopt=True)
    assert row["status"] == "rejected-live" and (repo / "rsi" / "build.py").read_text() == "PROMPT = 'build it'\n"
    monkeypatch.setattr(improve, "live", lambda gen_dir, parent_tree, tree, tasks, seed: {
        "tasks_per_arm": tasks, "verdict": {"accept": True, "reasons": ["efficiency 4.0 -> 6.0, no guardrail worse"]}})
    row = generation(monkeypatch, {"rsi/build.py": "PROMPT = 'v3'\n", "docs/improvements/gen-2.md": NOTE}, live_tasks=2, adopt=True)
    assert row["status"] == "accepted-live" and (repo / "rsi" / "build.py").read_text() == "PROMPT = 'v3'\n"
    assert "weak evidence at this size" in improve.report(row)


def test_the_request_shows_the_scorecard_the_lineage_and_the_protected_list():
    ev = {"scorecard": {k: None for k in ["tasks", "in_band", "attempts", "efficiency", "yield", "attempts_per_hit", "first_shot_error",
                                         "prediction_error", *meter.GUARDRAILS]}, "gate_failures": {"mutations": {"tasks": 3}}}
    text = proposer.prompt(4, ev, [{"gen": 3, "status": "rejected-live", "title": "Longer tickets",
                                    "live": {"verdict": {"reasons": ["unfair_rate went from 0.0 to 0.5"]}}}], improve.PROTECTED)
    assert "docs/improvements/gen-4.md" in text and "gen-3 [rejected-live] Longer tickets" in text and "unfair_rate went from 0.0 to 0.5" in text
    assert "rsi/meter.py" in text and '"mutations"' in text and "## How this could be wrong" in text
