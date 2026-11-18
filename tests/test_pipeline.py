"""One whole calibration, every real stage in order, with only Harbor and the models faked.

The unit tests check each stage alone. This one checks that what a stage writes is what the
next one reads: gate -> pooled difficulty -> hinted stamp -> inspect -> red team -> history ->
pool -> rollout reader -> scorecard -> the evidence an improvement generation is shown.
"""

import ast
import json
import re

import pytest

from rsi import archive, calibrate, combine, family as F, gate, harbor, llm, meter, mine, proposer

SOLVER = "claude-opus-5-5"
PATCH = "diff --git a/job.py b/job.py\n--- a/job.py\n+++ b/job.py\n@@ -1 +1,3 @@\n+def solve(db, params):\n+    total = unit_price_cents * qty  # attempt {i}\n+    return {{}}\n"


@pytest.fixture
def task(business, shoe, tmp_path, monkeypatch):
    for mod, name, file in [(combine, "HISTORY", "history.jsonl"), (archive, "ARCHIVE", "archive.jsonl"), (mine, "MINED", "mined.jsonl"),
                            (mine, "MODES", "mined_modes.json"), (llm, "LOG", "llm_calls.jsonl")]:
        monkeypatch.setattr(mod, name, tmp_path / "runs" / file)
    t = F.Task(shoe.id, ("window", "dedupe", "discount"), {"store_id": "S02"})
    return F.export_harbor(shoe, t, F.load_dbs(business), tmp_path / "runs" / "r1" / "task")


class FakeHarbor:
    """Writes trial folders the way Harbor does and lets the real reader parse them."""

    def __init__(self, criteria):
        self.criteria, self.jobs = criteria, []

    def __call__(self, task_dir, agent, *, model=None, attempts=1, jobs_dir, job_name=None, extra=None):
        job = jobs_dir / job_name
        self.jobs.append(job_name)

        def trial(i, reward, failed=(), patch=None):
            d = job / f"trial-{i}"
            (d / "verifier").mkdir(parents=True, exist_ok=True)
            (d / "result.json").write_text(json.dumps({"trial_name": f"{job_name}-{i}", "verifier_result": {"rewards": {"reward": reward}}}))
            (d / "verifier" / "ctrf.json").write_text(json.dumps({"results": {"tests": [
                {"name": n, "status": "failed", "message": f"assert {i} == 9"} for n in failed]}}))
            if patch:
                (d / "verifier" / "agent.patch").write_text(patch)

        tests_of = {c["id"]: c["tests"] for c in self.criteria}
        owned = {t for c in self.criteria for t in c["tests"] if sum(t in o["tests"] for o in self.criteria) == 1}
        own = {cid: [t for t in ts if t in owned] for cid, ts in tests_of.items()}      # keys only this rule changes
        both = {cid: [t for t in ts if t not in owned] for cid, ts in tests_of.items()}  # keys another rule changes too
        if agent == "oracle" and job_name.startswith("gate-mut-"):
            trial(0, 0.0, failed=tests_of[job_name.split("-")[3]][:1])  # the mutant is caught by its own criterion
        elif agent == "oracle":
            for i in range(attempts):  # the reference, and replays of passing attempts
                trial(i, 1.0)
        elif agent == "nop" or job_name.startswith("gate-shortcut-"):
            trial(0, 0.0, failed=tests_of["AC1"][:1], patch=PATCH.format(i="red"))
        else:  # eight cold attempts: two pass, six forget one rule each (its own keys and one it shares)
            for i in range(attempts):
                cid = f"AC{i % 3 + 1}"
                failed = [] if i < 2 else own[cid][:2] + both[cid][:1]
                trial(i, 0.0 if failed else 1.0, failed, PATCH.format(i=i))
        return harbor.read_trials(job)


def fake_judge(criteria):
    sentence = {t: c["sentence"] for c in criteria for t in c["tests"]}

    def ask_json(prompt, model, **kw):
        if "A cold attempt failed these hidden tests:" in prompt:  # inspect
            failed = ast.literal_eval(re.search(r"failed these hidden tests: (\[.*?\])\n", prompt).group(1))
            return {"tests": [{"test": t, "sentence": sentence[t], "fair": True} for t in failed]}
        if "THE ATTEMPT" in prompt:  # the rollout reader
            return [{"criterion": cid, "quotes": ["+    total = unit_price_cents * qty"], "slip": "forgot the rule", "kind": "documented"}
                    for cid in re.findall(r"^- (AC\d+) \(mode", prompt, re.M)]
        raise AssertionError(f"unexpected model call: {prompt[:80]}")

    return ask_json


def test_a_whole_calibration_hangs_together(task, tmp_path, monkeypatch):
    criteria = json.loads((task / "authoring" / "criteria.json").read_text())
    fake = FakeHarbor(criteria)
    monkeypatch.setattr(harbor, "run_job", fake)
    monkeypatch.setattr(llm, "ask_json", fake_judge(criteria))
    monkeypatch.setattr(calibrate.build, "revise", lambda *a, **k: pytest.fail(f"no revision expected: {a[2][:120]}"))

    assert calibrate.run(task) is True

    # every stage ran, in an order that spends Docker before models and models before the pool
    kinds = [re.sub(r"-(r\d+-)?(claude.*|\d\d-AC.*|gate-difficulty.*)$", "", j) for j in fake.jobs]
    assert kinds == ["gate-oracle", "gate-nop", "gate-mut", "gate-mut", "gate-mut", "gate-difficulty", "gate-replay", "gate-replay",
                     "gate-shortcut-exploit", "gate-shortcut-please-hack", "gate-shortcut-hard-code"]
    assert gate.summary(task).endswith("GATE: ALL PASS")

    # history: who slipped on what, per attempt, and where the first shot landed
    row = combine.read_jsonl(combine.HISTORY)[-1]
    m = row["by_model"][f"anthropic/{SOLVER}"]
    assert (row["solved"], row["attempts"], row["first_rate"], row["in_band"]) == (2, 8, 0.25, True)
    assert m["patterns"]["solved"] == 2 and sorted(map(tuple, m["patterns"]["slips"])) == sorted(
        [("timezone-math",)] * 2 + [("idempotency-under-failure",)] * 2 + [("derived-surface-omitted",)] * 2)
    st = combine.stats(combine.read_jsonl(combine.HISTORY))  # this run's rows alone, without the hackathon prior
    assert (st.slips["timezone-math"], st.handled["timezone-math"]) == (2, 6)  # the next design reads it
    assert st.rho("timezone-math", "derived-surface-omitted") < 1  # slips on different rules in different attempts: harder together

    # the pool and the rollout reader
    assert [r["task"] for r in archive.pool("active")] == [task.name]
    readings = combine.read_jsonl(mine.MINED)
    assert len({r["trial"] for r in readings}) == 6 and {r["mode"] for r in readings} == set(row["modes"])

    # the scorecard, from gate output alone, and what a generation would be shown
    card = meter.scorecard([tmp_path / "runs" / "r1"])
    assert (card["tasks"], card["in_band"], card["attempts"], card["efficiency"]) == (1, 1, 8, 12.5)
    assert (card["first_shot_error"], card["unfair_rate"], card["hack_rate"], card["unproven_rate"]) == (0.0, 0.0, 0.0, None)
    ev = proposer.evidence(tmp_path / "runs")
    assert ev["scorecard"]["tasks"] == 1 and ev["gate_failures"] == {} and ev["slips_read_from_failed_attempts"]["readings"] == 6
    assert ev["recent_history"][-1]["task"] == task.name


def test_a_red_team_pass_sends_an_in_band_task_back_instead_of_into_the_pool(task, monkeypatch):
    criteria = json.loads((task / "authoring" / "criteria.json").read_text())
    fake = FakeHarbor(criteria)

    def cheating_red_team(task_dir, agent, **kw):
        trials = fake(task_dir, agent, **kw)
        if kw["job_name"].startswith("gate-shortcut-hard-code"):
            trials[0].reward = 1.0  # the hard-coded build passes, and its replay passes too
        return trials

    sent = []
    judge = fake_judge(criteria)
    monkeypatch.setattr(harbor, "run_job", cheating_red_team)
    monkeypatch.setattr(llm, "ask_json", lambda prompt, model, **k: {"verdict": "shortcut", "evidence": []} if "RED-TEAM CHANGE" in prompt else judge(prompt, model, **k))
    monkeypatch.setattr(calibrate.build, "revise", lambda task_dir, measurement, instruction: sent.append(instruction))
    monkeypatch.setattr(calibrate, "CONFIG", __import__("dataclasses").replace(calibrate.CONFIG, max_calibration_rounds=1))

    assert calibrate.run(task) is False
    assert len(sent) == 1 and "red-team attempts got full reward" in sent[0]
    assert archive.pool() == []  # never admitted
    assert meter.scorecard([task.parent.parent])["hack_rate"] == 1.0
