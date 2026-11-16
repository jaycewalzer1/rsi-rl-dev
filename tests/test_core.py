import json
import random
from pathlib import Path

import pytest

from rsi import calibrate, combine, gate, harbor
from rsi.llm import extract_json

ROOT = Path(__file__).resolve().parent.parent
SMOKE = ROOT / "examples" / "tasks" / "smoke-ledger"


def test_wilson_bounds():
    lo, hi = gate.wilson(2, 8)
    assert 0 < lo < 0.25 < hi < 1


@pytest.mark.parametrize(
    "solved,graded,want",
    [(0, 3, "need_more"), (0, 8, "too_hard"), (2, 8, "in_band"), (7, 8, "too_easy"), (1, 2, "need_more")],
)
def test_band_verdict(solved, graded, want):
    assert gate.band_verdict(solved, graded) == want


def test_extract_json_from_fenced_answer():
    assert extract_json('sure:\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert extract_json('noise [1, {"b": 2}] tail') == [1, {"b": 2}]


def test_read_trials(tmp_path):
    for name, reward, exc in [("t1", 1.0, None), ("t2", 0.0, None), ("t3", None, {"exception_type": "RateLimit"})]:
        d = tmp_path / name
        (d / "verifier").mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({
            "trial_name": name,
            "verifier_result": {"rewards": {"reward": reward}} if reward is not None else None,
            "exception_info": exc,
        }))
    (tmp_path / "t2" / "verifier" / "ctrf.json").write_text(json.dumps(
        {"results": {"tests": [{"name": "a", "status": "passed"}, {"name": "b", "status": "failed"}]}}))
    trials = {t.name: t for t in harbor.read_trials(tmp_path)}
    assert trials["t1"].solved and not trials["t2"].solved
    assert trials["t3"].reward is None and trials["t3"].exception == "RateLimit"
    assert harbor.failed_tests(trials["t2"]) == ["b"]


def test_mutant_task_appends_patch(tmp_path):
    patch = SMOKE / "authoring" / "mutations" / "01-AC1-allows-floats.patch"
    mt = gate.mutant_task(SMOKE, patch, tmp_path)
    solve = (mt / "solution" / "solve.sh").read_text()
    assert solve.index("reference.patch") < solve.index("mutation.patch")
    assert (mt / "solution" / "mutation.patch").read_text() == patch.read_text()


def test_smoke_shape_flags_small_task(tmp_path):
    import shutil
    t = tmp_path / "smoke"
    shutil.copytree(SMOKE, t)
    r = gate.stage_shape(t)
    assert not r["ok"] and "words" in r["note"] and "hidden tests" in r["note"]
    assert "has no mutation" not in r["note"]


def test_pick_modes_one_per_category_and_learns(tmp_path, monkeypatch):
    modes = [{"id": f"m{i}", "category": f"c{i % 3}"} for i in range(6)]
    picked = combine.pick_modes(modes, 3, random.Random(0))
    assert len({m["category"] for m in picked}) == 3

    hist = tmp_path / "history.jsonl"
    hist.write_text(json.dumps({"attempts": 40, "mode_slips": {"m0": 40, "m1": 0}}) + "\n")
    monkeypatch.setattr(combine, "HISTORY", hist)
    w = combine.mode_weights(modes)
    assert w["m0"] == (41, 1) and w["m1"] == (1, 41)
    wins = sum(combine.pick_modes(modes[:2], 1, random.Random(s))[0]["id"] == "m0" for s in range(50))
    assert wins > 45


def test_mode_slips_and_hardest_family(tmp_path):
    t = tmp_path / "task"
    (t / "authoring").mkdir(parents=True)
    (t / "authoring" / "criteria.json").write_text(json.dumps([
        {"id": "AC1", "mode": "authz-by-id", "tests": ["a1", "a2"]},
        {"id": "AC2", "mode": "derived-surface-omitted", "tests": ["b1"]},
    ]))
    diff = {"models": {"m": {"failed_tests": {"t1": ["a1", "a2"], "t2": ["a1", "b1"], "t3": []}}}}
    assert calibrate.mode_slips(t, diff) == {"authz-by-id": 2, "derived-surface-omitted": 1}
    assert calibrate.hardest_family(t, diff) == "AC1"
