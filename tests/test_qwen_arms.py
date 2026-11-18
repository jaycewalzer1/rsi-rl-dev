"""The Qwen leg's reward-variance report and the single-vs-stacked ablation arms (no model needed)."""

import pytest

from rsi import family as F
from rsi.qwen import METHOD, SETS, compare, markdown, rate, share_before, train_plan, trainability


def test_trainability_counts_the_groups_a_group_relative_step_could_learn_from():
    rows = [{"id": "a", "solved": s} for s in (True, False, False, False)] + [{"id": "b", "solved": False}] * 2 \
        + [{"id": "c", "solved": True}] * 2 + [{"id": "d", "solved": s} for s in (True, False)]
    t = trainability(rows)
    assert (t["tasks"], t["all_failed"], t["all_solved"], t["mixed"], t["signal_share"]) == (4, 1, 1, 2, 0.5)
    assert t["mean_reward_variance"] == pytest.approx((0.25 * 0.75 + 0.5 * 0.5) / 4, abs=1e-4)
    assert trainability([])["signal_share"] is None


def test_ablation_arms_change_what_is_on_the_tickets_and_not_how_many(shoe):
    levels = {shoe.id: {"headline": 2, "harder": 3}}
    cur, single, stacked = (train_plan([shoe], levels, 6, arm)[shoe.id] for arm in ("curriculum", "single", "stacked"))
    assert (cur["levels"], single["levels"], stacked["levels"]) == ([0, 1, 2, 3], [1], [2, 3])
    assert cur["n"] * len(cur["levels"]) == single["n"] * len(single["levels"]) == stacked["n"] * len(stacked["levels"]) == 24
    split = F.split_combos(shoe)
    assert all(split[l]["train"] for l in single["levels"] + stacked["levels"])  # never a benchmark combination
    assert train_plan([shoe], levels, 6) == train_plan([shoe], levels, 6, "curriculum")


def test_an_arm_starts_from_the_main_runs_before_measurements(tmp_path):
    main, arm = tmp_path / "qwen", tmp_path / "qwen_stacked"
    main.mkdir(), arm.mkdir()
    (main / "calibrate.jsonl").write_text("c\n")
    (main / "bench_before.jsonl").write_text("b\n")
    (arm / "bench_before.jsonl").write_text("already here\n")
    share_before(main, arm)
    assert (arm / "calibrate.jsonl").read_text() == "c\n" and (arm / "bench_before.jsonl").read_text() == "already here\n"
    share_before(main, main)  # the main run copies nothing onto itself


def test_report_names_the_method_the_arm_and_the_reward_variance():
    def bench(solved):
        rows = {n: [{"id": i, "solved": i < solved[n]} for i in range(20 if n != "out_of_reach" else 0)] for n in SETS}
        return {"model": {"adapter": "a"}, "rows": rows,
                "sets": {n: {**rate(v), "by_family": {"fam L2": rate(v)}, "failures": {}} for n, v in rows.items()}}
    r = {"arm": "stacked", "method": METHOD, "model": {"id": "m", "weights": "w", "quantization": None}, "dataset": "d", "slices": 3,
         "families": ["fam"], "control": "shoe", "calibrate": {"fam": {"reach": True, "note": "", "curve": []}},
         "settle": {"targets": {"headline": 0.5, "harder": 0.25, "tolerance": 0.1}, "reached": True, "steps": [], "before_by_level": {}},
         "before": bench({"headline": 10, "harder": 5, "control": 4, "out_of_reach": 0})}
    r["after_r1"] = bench({"headline": 12, "harder": 6, "control": 4, "out_of_reach": 0})
    r["compare_r1"] = compare(r["before"], r["after_r1"])
    r["collect_r1"] = {"solved": 30, "sampled": 96, "solved_after_fix": 8, "fix_turns": 40, "rules_with_a_kept_answer": {"fam": "7/8"},
                       "trainability": {"tasks": 24, "all_failed": 12, "all_solved": 2, "mixed": 10, "signal_share": 0.417, "mean_reward_variance": 0.08}}
    r["train_r1"] = {"rows": 38, "rows_by_family": {"fam": 38}, "steps": 114, "first_loss": 0.2, "last_loss": 0.1, "seconds": 300, "peak_memory_gb": 12.0}
    r["last_round"] = 1
    text = markdown(r)
    assert "rejection fine-tuning (RFT)" in text and "Training arm: stacked." in text
    assert "round 1: 10 of 24 tasks had both passing and failing samples (42%); 12 all failed, 2 all passed; mean p(1-p) 0.08" in text
