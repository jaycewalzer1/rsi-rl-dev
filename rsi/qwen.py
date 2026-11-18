"""Post-train a local Qwen on a run's task families and measure it before and after.

    calibrate      per family, start at the top level (every rule on, "impossible") and come
                   down. Levels are screened with a few tasks until the first solve; from there
                   every level gets 20 tasks, and the descent stops once the pass rate reaches
                   the headline target. A family Qwen cannot solve even with every rule off is
                   reported as out of reach and kept out of the headline.
    settle         choose, per family, a headline level and a harder level so that the BEFORE
                   pass rates over all families land on the targets (headline 50%, harder 25%;
                   --headline-target, --harder-target). The choice is tested on the held-out
                   benchmark tasks themselves and adjusted until the levels it picks are the
                   levels it measured. Greedy decoding is deterministic, so the rates it
                   settles on are the before rates, not estimates of them.
    bench before   held-out tasks, greedy decoding, all-or-nothing on every data slice:
                     headline      the headline level of every calibrated family
                     harder        the harder level of the same families
                     out_of_reach  level 0 of families that never produced a solve
                     control       a family from a DIFFERENT dataset nothing here trains on
    collect        on TRAINING rule combinations from level 0 up to the harder level: k samples per task.
                   An answer that crashes on the rows the model was shown gets its own traceback
                   back and another go (twice at most); one that runs but is graded wrong is told
                   only that. A rule that still has no correct answer gets extra samples. Only
                   answers the oracle grades fully right are kept. The reward is the only teacher: no reference solution and no
                   expected value is ever shown to the model.
    train          LoRA on every answer kept so far, each as ticket -> final correct code
                   (mlx-vlm SFT trainer, loss on the answer only), from the base weights.
                   A few training rule combinations per level are set aside as dev tasks:
                   never trained on, never in the benchmark. After each epoch the model answers
                   them greedily; the epoch that solves most is the adapter, and training stops
                   once an epoch does worse than an earlier one. How long to train is therefore
                   never decided on the benchmark.
    bench after    the same held-out tasks with the adapter; paired test against before
    (collect, train and bench after repeat per round; round N samples with round N-1's adapter)

Benchmark rule combinations are never trained on (family.split_combos), and every answer is
graded on slices of the business's data the model was never shown rows from.

Every stage writes its rows as it goes and picks up where it left off, so a stopped job
resumes. It loads Qwen, so it runs with kokopelli's venv (mlx-vlm), only through its GPU queue:

    uv run rsi qwen --run NAME            # queues it; see cli.py for the exact command

Every result file names the model and weights it used.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

from . import business_gen, events, family as F, intake
from .config import CONFIG
from .gate import wilson
from .store import read_jsonl, write_json

KOKO = Path(os.environ.get("RSI_KOKOPELLI", Path.home() / "Desktop" / "kokopelli"))
BASE = Path(os.environ.get("RSI_QWEN_WEIGHTS", KOKO / "models" / "Qwen3-VL-8B-Instruct-4bit"))
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
SYSTEM = "You are a careful Python engineer. Follow every rule in the ticket exactly."
LORA = {"rank": 16, "alpha": 32.0, "dropout": 0.0, "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "learning_rate": 5e-5, "batch_size": 1, "max_seq_length": 6144, "epochs": 4, "max_rows": 400, "seed": 0}
DEV_N = 16  # greedy dev tasks per family behind the choice of epoch
EXPLORE = {"tasks": 8, "samples": 8}  # extra sampling per rule that still has no kept answer
# the bar this repo sets for "post-training worked"; not a lab standard
TARGET_GAIN, TARGET_P = 0.20, 0.05
CONFIRM = 20  # tasks behind a calibration level before it can be chosen
HEADLINE_TARGET, HARDER_TARGET, TOLERANCE = 0.50, 0.25, 0.10  # where the BEFORE pass rates should sit
FIX = ("Running your function on the sample tables raised:\n\n{trace}\n\n"
       "Fix it. Reply with the complete corrected function in one ```python block.")
WRONG = ("Your function ran on the sample tables, but its result was graded wrong. That is all the grader says.\n\n"
         "Go through the ticket's rules one by one against your code, find what your code does not do exactly as written, "
         "and reply with the complete corrected function in one ```python block.")


class Qwen:
    def __init__(self, base: Path = BASE, adapter: Path | None = None):
        from mlx_vlm import load
        self.base, self.adapter = base, adapter
        self.model, self.processor = load(str(base), adapter_path=str(adapter) if adapter else None)

    def describe(self) -> dict:
        q = json.loads((self.base / "config.json").read_text()).get("quantization")
        return {"id": MODEL_ID, "weights": str(self.base), "quantization": q,
                "adapter": str(self.adapter) if self.adapter else None}

    def _fmt(self, chat: str | list[dict]) -> str:
        from mlx_vlm.prompt_utils import apply_chat_template
        turns = [{"role": "user", "content": chat}] if isinstance(chat, str) else chat
        return apply_chat_template(self.processor, self.model.config, [{"role": "system", "content": SYSTEM}, *turns], num_images=0)

    def generate(self, prompts: list[str | list[dict]], temperature: float, max_tokens: int = 2000, batch: int = 8):
        """Yields (index, text) as chunks finish. A prompt is the user's text or a list of turns."""
        import mlx.core as mx
        from mlx_vlm import batch_generate, generate
        from mlx_vlm.sample_utils import make_sampler
        sampling = {"sampler": make_sampler(temperature)} if temperature > 0 else {}
        for i in range(0, len(prompts), batch):
            chunk = [self._fmt(p) for p in prompts[i:i + batch]]
            try:
                if getattr(self, "_no_batch", False):
                    raise RuntimeError("batch path off")
                texts = list(batch_generate(self.model, self.processor, prompts=chunk, max_tokens=max_tokens,
                                            verbose=False, **sampling).texts)
            except Exception as e:  # the batch path is newer; fall back to one at a time for good
                if not getattr(self, "_no_batch", False):
                    print(f"batch_generate failed ({type(e).__name__}: {e}); generating one by one", flush=True)
                self._no_batch = True
                texts = [generate(self.model, self.processor, c, max_tokens=max_tokens,
                                  temperature=temperature, verbose=False).text for c in chunk]
            mx.clear_cache()
            for j, t in enumerate(texts):
                yield i + j, t


class Ctx:
    """Families with the data slices each is graded on (the control family has its own dataset)."""

    def __init__(self):
        self.fams: dict[str, F.Family] = {}
        self.dbs: dict[str, list[dict]] = {}

    def add(self, fam: F.Family, dbs: list[dict]):
        self.fams[fam.id], self.dbs[fam.id] = fam, dbs


def attempt(qwen: Qwen, ctx: Ctx, tasks: list[F.Task], *, samples: int, temperature: float, stage: str, log: Path) -> list[dict]:
    """Generate and grade; rows already in `log` are reused, so a stopped stage resumes."""
    have = {(r["id"], r["sample"]): r for r in read_jsonl(log) if r.get("temperature") == temperature}
    todo = [(t, s) for t in tasks for s in range(samples) if (t.id, s) not in have]
    started, done = time.time(), 0
    if todo:
        prompts = [F.prompt(ctx.fams[t.family], t, ctx.dbs[t.family][0]) for t, _ in todo]
        with log.open("a") as f:
            for i, text in qwen.generate(prompts, temperature):
                t, s = todo[i]
                fam = ctx.fams[t.family]
                g = F.score(fam, t, text, ctx.dbs[t.family])
                row = {**t.as_dict(fam), "sample": s, "temperature": temperature, **g, "answer": text}
                f.write(json.dumps(row) + "\n")
                f.flush()
                have[(t.id, s)] = row
                done += 1
                events.emit("work", stage=stage, agent=f"qwen-{t.id}-{s}", role="qwen",
                            msg=f"{t.id} {'SOLVED' if g['solved'] else 'fail'} ({g['key_share']:.0%} of keys"
                                + (f"; {g['error'][:60]}" if g["error"] else "") + ")",
                            level=t.level, family=t.family, solved=g["solved"])
                if done % 8 == 0 or done == len(todo):
                    print(f"  {stage}: {done}/{len(todo)} generated, {time.time() - started:.0f} s", flush=True)
    rows = [have[(t.id, s)] for t in tasks for s in range(samples)]
    print(f"{stage}: {sum(r['solved'] for r in rows)}/{len(rows)} solved"
          + (f" ({len(todo)} new in {time.time() - started:.0f} s)" if todo else " (all reused)"), flush=True)
    return rows


def repair(qwen: Qwen, ctx: Ctx, rows: list[dict], *, turns: int, temperature: float, stage: str, log: Path) -> list[dict]:
    """Give each failed answer its reward and another go; returns the new rows.

    A crashed answer gets its own traceback: whatever running it on the shown slice raises
    (family.crash_on_shown). An answer that runs but is graded wrong is told only that. How it is
    wrong, or what was expected, never reaches the model: that would leak the oracle."""
    have = {(r["id"], r["sample"], r["turn"]): r for r in read_jsonl(log)}
    out, cur = [], rows
    for turn in range(1, turns + 1):
        todo, nxt = [], []
        for r in cur:
            if r["solved"]:
                continue
            if (r["id"], r["sample"], turn) in have:
                nxt.append(have[(r["id"], r["sample"], turn)])
                continue
            task = F.Task(r["family"], tuple(r["rules"]), r["params"])
            todo.append((r, task, F.crash_on_shown(task, r["answer"], ctx.dbs[r["family"]][0])))
        started = time.time()
        if todo:
            chats = [[{"role": "user", "content": F.prompt(ctx.fams[t.family], t, ctx.dbs[t.family][0])},
                      {"role": "assistant", "content": r["answer"]},
                      {"role": "user", "content": FIX.format(trace=trace) if trace else WRONG}] for r, t, trace in todo]
            with log.open("a") as f:
                for i, text in qwen.generate(chats, temperature):
                    r, t, trace = todo[i]
                    fam = ctx.fams[t.family]
                    g = F.score(fam, t, text, ctx.dbs[t.family])
                    row = {**t.as_dict(fam), "sample": r["sample"], "turn": turn, "temperature": temperature,
                           "crash": trace.splitlines()[-1][:200] if trace else "", **g, "answer": text}
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    nxt.append(row)
                    events.emit("work", stage=stage, agent=f"qwen-{t.id}-{r['sample']}-fix{turn}", role="qwen",
                                msg=f"{t.id} fix {turn} after {row['crash'][:50] or 'a wrong answer'}: {'SOLVED' if g['solved'] else 'fail'}",
                                level=t.level, family=t.family, solved=g["solved"])
        print(f"{stage} fix {turn}: {sum(r['solved'] for r in nxt)}/{len(nxt)} failed answers now solved"
              + (f" ({len(todo)} new in {time.time() - started:.0f} s)" if todo else ""), flush=True)
        out += nxt
        cur = nxt
    return out


def rate(rows: list[dict]) -> dict:
    k, n = sum(r["solved"] for r in rows), len(rows)
    lo, hi = wilson(k, n)
    return {"solved": k, "n": n, "rate": round(k / n, 3) if n else None, "wilson95": [round(lo, 3), round(hi, 3)]}


def mcnemar(before: list[dict], after: list[dict]) -> dict:
    """Paired, exact: of the tasks whose result changed, how many improved vs regressed."""
    b = {r["id"]: r["solved"] for r in before}
    gained = sum(1 for r in after if r["solved"] and not b[r["id"]])
    lost = sum(1 for r in after if not r["solved"] and b[r["id"]])
    n = gained + lost
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(gained, lost) + 1)) / 2 ** n)
    return {"gained": gained, "lost": lost, "p": round(p, 5)}


def by_combination(before: list[dict], after: list[dict], flips: int = 20000) -> dict:
    """The same paired question with the rule combination as the unit. Tasks that share a combination
    differ only in a parameter value, so they rise and fall together and are not independent draws.
    Sign-flip test on each combination's change in tasks solved: exact up to 16 changed combinations,
    sampled beyond that."""
    b = {r["id"]: r["solved"] for r in before}
    change = defaultdict(int)
    for r in after:
        change[r.get("combo", r["id"])] += int(r["solved"]) - int(b[r["id"]])
    d = [v for v in change.values() if v]
    seen = abs(sum(d))
    if not d:
        p = 1.0
    elif len(d) <= 16:
        p = sum(abs(sum(v * s for v, s in zip(d, signs))) >= seen for signs in itertools.product((1, -1), repeat=len(d))) / 2 ** len(d)
    else:
        rng = random.Random(0)
        p = (1 + sum(abs(sum(v * rng.choice((1, -1)) for v in d)) >= seen for _ in range(flips))) / (flips + 1)
    return {"combinations": len(change), "up": sum(v > 0 for v in d), "down": sum(v < 0 for v in d), "p_by_combination": round(p, 5)}


# ---------------------------------------------------------------- calibrate

def _confirmed(level: int, m: dict) -> bool:
    """Enough tasks behind a level to choose it. Level 0 has as many tasks as there are params."""
    return m["n"] >= CONFIRM or (level == 0 and m["n"] >= 5)


def calibrate(qwen: Qwen, ctx: Ctx, fam: F.Family, out: Path, n: int = 5, confirm: int = CONFIRM, stop_at: float = HEADLINE_TARGET) -> dict:
    """Come down from the top level until the pass rate reaches the headline target. Returns the
    curve; `reach` is False when no confirmed level has a solve."""
    db = ctx.dbs[fam.id][0]
    top = len(fam.rules)
    measured: dict[int, dict] = {}

    def measure(level: int, count: int) -> dict:
        pool = "train" if F.split_combos(fam)[level]["train"] else "bench"
        tasks = F.make_tasks(fam, level, count, pool, 9000, db)
        rows = attempt(qwen, ctx, tasks, samples=1, temperature=0.0, stage="calibrate", log=out / "calibrate.jsonl")
        measured[level] = rate(rows)
        events.emit("metric", stage="calibrate", name="calibrate", family=fam.id, level=level,
                    value=measured[level]["rate"], n=len(rows), msg=f"{fam.id} level {level}: {measured[level]['solved']}/{len(rows)}")
        return measured[level]

    seen = False  # screening jumps two levels at a time near the top; after the first solve every level is confirmed
    for level in [l for l in dict.fromkeys([top, top - 2, top - 4, *range(top - 5, -1, -1)]) if l >= 0]:
        m = measure(level, confirm if seen else n)
        if not seen and m["solved"]:
            seen = True
            m = measure(level, confirm)
            if level + 1 <= top:
                measure(level + 1, confirm)  # the level jumped over may be the harder level
        if seen and m["rate"] >= stop_at:
            break
    reach = any(_confirmed(l, m) and m["solved"] for l, m in measured.items())
    return {"family": fam.id, "reach": reach,
            "note": "" if reach else "out of reach: Qwen solved nothing, or too little, even with every rule off",
            "curve": [{"level": l, **m} for l, m in sorted(measured.items(), reverse=True)]}


def choose_levels(rates: dict[str, dict[int, tuple[int, int]]], targets: tuple[float, float] = (HEADLINE_TARGET, HARDER_TARGET)) -> dict[str, tuple[int, int]]:
    """Per family a (headline, harder) pair of levels, harder above headline, so the pass rates
    pooled over families sit nearest the targets. rates: family -> level -> (solved, n). A small
    extra cost keeps each family near the targets too, so one family is not all of the headline."""
    fams = [f for f in rates if len(rates[f]) >= 2]
    if not fams:
        return {}
    options = [[(h, g) for h in sorted(rates[f]) for g in sorted(rates[f]) if g > h] for f in fams]
    best, best_cost = {}, None
    for combo in itertools.product(*options):
        cost = 0.0
        for which, target in enumerate(targets):
            pairs = [rates[f][lv[which]] for f, lv in zip(fams, combo)]
            cost += abs(sum(k for k, _ in pairs) / sum(n for _, n in pairs) - target)
            cost += 0.25 * sum(abs(k / n - target) for k, n in pairs) / len(pairs)
        cost -= 1e-6 * sum(h + g for h, g in combo)  # ties go to the harder levels
        if best_cost is None or cost < best_cost:
            best, best_cost = dict(zip(fams, combo)), cost
    return best


def settle(qwen: Qwen, ctx: Ctx, fams: list[F.Family], cal: dict[str, dict], n: int, out: Path,
           targets: tuple[float, float] = (HEADLINE_TARGET, HARDER_TARGET)) -> dict:
    """Adjust the benchmark levels until the BEFORE rates on the held-out tasks sit on the targets.

    Start from the calibration curve (training-pool tasks), measure the levels it points to on the
    benchmark tasks, choose again with the measurements in place of the estimates, and repeat until
    every chosen level is one that was measured."""
    events.emit("stage_start", stage="settle", msg=f"adjust levels until before is about {targets[0]:.0%} (headline) and {targets[1]:.0%} (harder)")
    guess = {fid: {x["level"]: (x["solved"], x["n"]) for x in c["curve"] if _confirmed(x["level"], x)} for fid, c in cal.items() if c["reach"]}
    by_id = {f.id: f for f in fams}
    measured: dict[str, dict[int, tuple[int, int]]] = {fid: {} for fid in guess}
    steps = []
    while True:
        rates = {fid: {**guess[fid], **measured[fid]} for fid in guess}
        levels = choose_levels(rates, targets)
        new = [(fid, l) for fid, pair in levels.items() for l in pair if l not in measured[fid]]
        pooled = [sum(rates[f][lv[i]][0] for f, lv in levels.items()) / max(1, sum(rates[f][lv[i]][1] for f, lv in levels.items())) for i in (0, 1)]
        steps.append({"levels": {f: list(lv) for f, lv in levels.items()}, "headline": round(pooled[0], 3), "harder": round(pooled[1], 3),
                      "measured": not new})
        events.emit("metric", stage="settle", name="settle", value=pooled[0], harder=pooled[1], levels=steps[-1]["levels"], final=not new,
                    msg=("settled: " if not new else "trying: ") + ", ".join(f"{f} L{h}/L{g}" for f, (h, g) in levels.items())
                        + f" -> headline {pooled[0]:.0%}, harder {pooled[1]:.0%}" + ("" if not new else " (part estimated)"))
        if not levels:
            break
        if not new:
            # measured and settled. If it missed a target, widen the search: the level next to each
            # chosen one, on the side the miss points to, and choose again with it measured.
            miss = [pooled[i] - targets[i] for i in (0, 1)]
            for fid, pair in levels.items():
                for i in (0, 1):
                    step = 0 if abs(miss[i]) <= TOLERANCE else (-1 if miss[i] < 0 else 1)  # too few solved: an easier level
                    lvl = pair[i] + step
                    if step and 0 <= lvl <= len(by_id[fid].rules) and lvl not in measured[fid] and (fid, lvl) not in new:
                        new.append((fid, lvl))
            if not new:
                break
            steps[-1]["widened_to"] = [f"{f} L{l}" for f, l in new]
            events.emit("work", stage="settle", agent="settle", role="qwen", msg="off target; also measuring " + ", ".join(steps[-1]["widened_to"]))
        for fid, lvl in new:
            fam = by_id[fid]
            rows = attempt(qwen, ctx, F.make_tasks(fam, lvl, n, "bench", 1000, ctx.dbs[fid][0]), samples=1, temperature=0.0,
                           stage="bench_before", log=out / "bench_before.jsonl")
            measured[fid][lvl] = (sum(r["solved"] for r in rows), len(rows))
    reached = bool(levels) and abs(pooled[0] - targets[0]) <= TOLERANCE and abs(pooled[1] - targets[1]) <= TOLERANCE
    res = {"targets": {"headline": targets[0], "harder": targets[1], "tolerance": TOLERANCE}, "reached": reached,
           "levels": {f: {"headline": h, "harder": g} for f, (h, g) in levels.items()}, "steps": steps,
           "before_by_level": {f: {str(l): {"solved": k, "n": m} for l, (k, m) in sorted(v.items())} for f, v in measured.items()}}
    events.emit("stage_end", stage="settle", levels={f: v["headline"] for f, v in res["levels"].items()},
                harder_levels={f: v["harder"] for f, v in res["levels"].items()},
                msg=f"before: headline {pooled[0]:.0%}, harder {pooled[1]:.0%}" + ("" if reached else " (targets not reached with the levels there are)"))
    return res


# ---------------------------------------------------------------- collect and train

def dev_combos(fam: F.Family, harder: int) -> dict[int, list[list[str]]]:
    """Per training level, the training rule combinations set aside as dev tasks: about one in six."""
    out = {}
    for lvl in range(1, harder + 1):
        train = F.split_combos(fam)[lvl]["train"]
        if len(train) >= 3:
            out[lvl] = [list(c) for c in train[-max(1, len(train) // 6):]]
    return out


def dev_tasks(ctx: Ctx, plan: dict[str, dict], n: int = DEV_N) -> list[F.Task]:
    """n tasks per family on its set-aside combinations, spread evenly over the levels."""
    tasks = {}
    for fid, want in plan.items():
        by_level = [cs for _, cs in sorted(want.get("dev", {}).items()) if cs]
        if not by_level:
            continue
        ps = list(ctx.fams[fid].params_choices(ctx.dbs[fid][0]))
        random.Random(f"dev-{fid}").shuffle(ps)
        for i in range(n):
            cs, j = by_level[i % len(by_level)], i // len(by_level)
            t = F.Task(fid, tuple(cs[j % len(cs)]), ps[j % len(ps)])
            tasks[t.id] = t
    return list(tasks.values())


def dev_check(qwen: Qwen, ctx: Ctx, tasks: list[F.Task], log: Path) -> dict:
    rows = attempt(qwen, ctx, tasks, samples=1, temperature=0.0, stage="dev", log=log)
    return {**rate(rows), "crashed": sum(bool(r["error"]) for r in rows)}


def collect_tasks(ctx: Ctx, plan: dict[str, dict], rnd: int) -> list[F.Task]:
    tasks = []
    for fid, want in plan.items():
        fam = ctx.fams[fid]
        dev = {tuple(c) for cs in want.get("dev", {}).values() for c in cs}
        for lvl in want["levels"]:
            pool = "train" if F.split_combos(fam)[lvl]["train"] else "bench"  # level 0 has one combination; the plan only lists it when no benchmark uses it
            more = F.make_tasks(fam, lvl, 3 * want["n"], pool, 3000 + rnd, ctx.dbs[fid][0])
            tasks += [t for t in more if tuple(t.rules) not in dev][:want["n"]]
    return tasks


def explore_tasks(ctx: Ctx, plan: dict[str, dict], missing: dict[str, set[str]], rnd: int) -> dict[tuple[str, str], list[F.Task]]:
    """For each rule with no kept answer: training tasks that include it, at the lowest level that has any."""
    out = {}
    for fid, rules in missing.items():
        fam, want = ctx.fams[fid], plan[fid]
        dev = {tuple(c) for cs in want.get("dev", {}).values() for c in cs}
        for rule in sorted(rules):
            for lvl in sorted(l for l in want["levels"] if l):
                every = F.make_tasks(fam, lvl, 10 ** 4, "train", 5000 + rnd, ctx.dbs[fid][0])
                tasks = [t for t in every if rule in t.rules and tuple(t.rules) not in dev][:EXPLORE["tasks"]]
                if tasks:
                    out[(fid, rule)] = tasks
                    break
    return out


def collect(qwen: Qwen, ctx: Ctx, plan: dict[str, dict], out: Path, rnd: int, k: int, fixes: int = 2,
            have: dict[str, set[str]] | None = None) -> list[dict]:
    """plan: family -> {"levels": [...], "n": tasks per level}. Returns the graded-correct answers as training rows.
    `have`: per family, rules an earlier round already kept an answer for. A rule of a calibrated family that
    still has none after this round's sampling gets extra samples: reward is sparse there, so look harder."""
    events.emit("stage_start", stage="collect", msg=f"round {rnd}: {k} samples per training task, failed answers get {fixes} more goes, keep graded-correct answers")
    tasks = collect_tasks(ctx, plan, rnd)
    first = attempt(qwen, ctx, tasks, samples=k, temperature=0.8, stage="collect", log=out / f"collect_r{rnd}.jsonl")
    fixed = repair(qwen, ctx, first, turns=fixes, temperature=0.8, stage="collect", log=out / f"collect_r{rnd}_fix.jsonl")
    covered = defaultdict(set, {fid: set(rules) for fid, rules in (have or {}).items()})
    for r in first + fixed:
        if r["solved"]:
            covered[r["family"]] |= set(r["rules"])
    missing = {fid: {x["id"] for x in ctx.fams[fid].rules} - covered[fid] for fid, want in plan.items() if want.get("dev")}
    extra, explored = [], {}
    for (fid, rule), tasks in explore_tasks(ctx, plan, missing, rnd).items():
        rows = attempt(qwen, ctx, tasks, samples=EXPLORE["samples"], temperature=0.8, stage="collect", log=out / f"collect_r{rnd}_explore.jsonl")
        explored[f"{fid}: {rule}"] = f"{sum(r['solved'] for r in rows)}/{len(rows)}"
        extra += rows
    kept, per_task = [], defaultdict(set)
    for r in first + fixed + extra:
        code = F.extract_code(r["answer"]).strip()
        if r["solved"] and code not in per_task[r["id"]] and len(per_task[r["id"]]) < 3:
            per_task[r["id"]].add(code)
            fam = ctx.fams[r["family"]]
            task = F.Task(fam.id, tuple(r["rules"]), r["params"])
            kept.append({"messages": [{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": F.prompt(fam, task, ctx.dbs[fam.id][0])},
                                      {"role": "assistant", "content": f"```python\n{code}\n```"}],
                         "task": r["id"], "family": fam.id, "level": r["level"], "rules": list(r["rules"]), "fixed": bool(r.get("turn"))})
    by = defaultdict(list)
    for r in first:
        by[f"{r['family']} L{r['level']}"].append(r)
    covered = defaultdict(set)
    for row in kept:
        covered[row["family"]] |= set(row["rules"])
    summary = {"model": qwen.describe(), "plan": plan, "sampled": len(first), "solved": sum(r["solved"] for r in first),
               "fix_turns": len(fixed), "solved_after_fix": sum(r["solved"] for r in fixed), "explored": explored,
               "after_a_traceback": f"{sum(r['solved'] for r in fixed if r['crash'])}/{sum(bool(r['crash']) for r in fixed)}",
               "after_graded_wrong": f"{sum(r['solved'] for r in fixed if not r['crash'])}/{sum(not r['crash'] for r in fixed)}",
               "kept": len(kept), "tasks": len(tasks), "tasks_with_a_solve": len(per_task),
               "first_try_by_family_level": {key: rate(v) for key, v in sorted(by.items())},
               "rules_with_a_kept_answer": {fid: f"{len(covered[fid])}/{len(ctx.fams[fid].rules)}" for fid in plan}}
    write_json(out / f"collect_r{rnd}.json", summary)
    write_json(out / f"kept_r{rnd}.json", kept)
    events.emit("metric", stage="collect", name="collect", round=rnd, value=len(kept), sampled=len(first), solved=summary["solved"],
                fixed=summary["solved_after_fix"], msg=f"round {rnd}: {summary['solved']}/{len(first)} first tries right, {summary['solved_after_fix']} more after a fix")
    events.emit("stage_end", stage="collect", msg=f"{len(kept)} graded-correct answers kept from {len(first)} samples and {len(fixed)} fixes", kept=len(kept))
    return kept


def train(qwen: Qwen, rows: list[dict], adapter_dir: Path, dev=None) -> dict:
    """One pass over the answers per epoch. `dev(qwen, log)` has the model answer the dev tasks after
    each epoch: the epoch that solves most of them is the adapter, and training stops once an epoch
    does worse than an earlier one. Without `dev` the last epoch is kept."""
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_vlm.trainer.api import Trainer
    from mlx_vlm.trainer.common.model import prepare_model_for_training
    from mlx_vlm.trainer.vlm.config import VLMTrainingArgs
    rng = random.Random(LORA["seed"])
    if len(rows) > LORA["max_rows"]:
        rows = rng.sample(rows, LORA["max_rows"])
    events.emit("stage_start", stage="train", msg=f"LoRA r{LORA['rank']} on {len(rows)} answers, up to {LORA['epochs']} epochs")
    if adapter_dir.exists():  # an attempt that never finished: kept, out of the way
        adapter_dir.rename(adapter_dir.with_name(f"{adapter_dir.name}.unfinished-{int(time.time())}"))
    adapter_dir.mkdir(parents=True)
    mx.set_cache_limit(2 * 1024 ** 3)  # the trainer's buffer cache grew to 32 GB on a 36 GB Mac without this
    mx.random.seed(LORA["seed"])
    prepare_model_for_training(qwen.model, train_type="lora", lora_rank=LORA["rank"], lora_alpha=LORA["alpha"],
                               lora_dropout=LORA["dropout"], target_modules=LORA["targets"])
    opt = optim.AdamW(learning_rate=LORA["learning_rate"])
    started, epochs, peak = time.time(), [], 0.0
    for ep in range(1, LORA["epochs"] + 1):
        data = [{"messages": r["messages"]} for r in rows]
        rng.shuffle(data)
        folder = adapter_dir / f"epoch_{ep}"
        args = VLMTrainingArgs(batch_size=LORA["batch_size"], iters=len(data), steps_per_report=5, steps_per_eval=None,
                               steps_per_save=10 ** 9, val_batches=-1, adapter_file=str(folder / "adapters.safetensors"),
                               max_seq_length=LORA["max_seq_length"], train_on_completions=True)
        trainer = Trainer(qwen.model, args, train_dataset=data, processing_class=qwen.processor, optimizer=opt, seed=LORA["seed"] + ep)
        trainer.train()
        trainer.save_adapter(str(folder / "adapters.safetensors"))
        hist = [h["train_loss"] for h in trainer.log_history if "train_loss" in h]
        for v in hist[:: max(1, len(hist) // 10)]:
            events.emit("metric", stage="train", msg=f"epoch {ep} loss {v:.3f}", name="train_loss", value=v)
        peak = max([peak] + [h.get("peak_memory", 0) for h in trainer.log_history])
        row = {"epoch": ep, "steps": len(data), "loss": round(sum(hist) / len(hist), 4) if hist else None}
        if dev:
            qwen.model.eval()
            mx.clear_cache()
            row["dev"] = dev(qwen, folder / "dev.jsonl")
            events.emit("metric", stage="train", name="dev", value=row["dev"]["rate"], epoch=ep,
                        msg=f"epoch {ep}: {row['dev']['solved']}/{row['dev']['n']} dev tasks solved, {row['dev']['crashed']} crashed")
        epochs.append(row)
        if dev and row["dev"]["solved"] < max(e["dev"]["solved"] for e in epochs):
            break
    best = max(epochs, key=lambda e: (e["dev"]["solved"], -e["dev"]["crashed"], -e["epoch"])) if dev else epochs[-1]
    for name in ("adapters.safetensors", "adapter_config.json"):
        shutil.copy(adapter_dir / f"epoch_{best['epoch']}" / name, adapter_dir / name)
    by_fam = defaultdict(int)
    for r in rows:
        by_fam[r["family"]] += 1
    res = {"base": qwen.describe(), "settings": LORA, "rows": len(rows), "rows_by_family": dict(by_fam),
           "steps": sum(e["steps"] for e in epochs), "epochs": epochs, "chosen_epoch": best["epoch"],
           "seconds": round(time.time() - started, 1), "first_loss": epochs[0]["loss"], "last_loss": best["loss"],
           "peak_memory_gb": round(peak, 2),
           "trainer": "mlx-vlm SFT (mlx_vlm.trainer), loss on the answer only", "finished": True}
    write_json(adapter_dir / "run.json", res)
    events.emit("stage_end", stage="train", msg=f"epoch {best['epoch']} of {len(epochs)} kept"
                + (f" ({best['dev']['solved']}/{best['dev']['n']} dev tasks)" if dev else "") + f", {res['seconds'] / 60:.0f} min")
    return res


# ---------------------------------------------------------------- the run

def control_ctx(run_dir: Path, ctx: Ctx) -> F.Family:
    """The shoe-store family on its own synthetic dataset: nothing in the run trains on it."""
    base = run_dir / "control"
    if not (base / "business" / "catalog.json").exists():
        business_gen.write_csv(business_gen.generate(7), base / "raw")
        log, events._path = events._path, None  # the control's intake is not part of the story
        intake.run(base / "raw", base, stewards=False, rows=800)
        events._path = log
    fam = F.load(F.BUILTIN / "shoe_store.py")
    ctx.add(fam, F.load_dbs(base / "business"))
    return fam


SETS = ("headline", "harder", "out_of_reach", "control")


def bench_sets(ctx: Ctx, fams: list[F.Family], control: F.Family, levels: dict[str, dict], n: int) -> dict[str, list[F.Task]]:
    """levels: family -> {"headline": l, "harder": l}; a family not in it is out of reach."""
    sets: dict[str, list[F.Task]] = {name: [] for name in SETS}
    for fam in fams:
        db = ctx.dbs[fam.id][0]
        if fam.id not in levels:
            sets["out_of_reach"] += F.make_tasks(fam, 0, max(4, n // 2), "bench", 1000, db)
            continue
        for name in ("headline", "harder"):
            sets[name] += F.make_tasks(fam, levels[fam.id][name], n, "bench", 1000, db)
    for lvl in (1, 2):
        sets["control"] += F.make_tasks(control, lvl, 8, "bench", 1000, ctx.dbs[control.id][0])
    return sets


def train_plan(fams: list[F.Family], levels: dict[str, dict], n: int) -> dict[str, dict]:
    """Training levels per family: from the plain job up to the harder benchmark level, never a
    rule combination a benchmark uses, and the combinations set aside as dev tasks ("dev"). An out-of-reach family gets a few level-1 tasks, in case
    what the model learns on the others carries over."""
    plan = {}
    for fam in fams:
        split = F.split_combos(fam)
        if fam.id not in levels:
            plan[fam.id] = {"levels": [1], "n": max(2, n // 2)}
            continue
        lv = levels[fam.id]
        want = [l for l in range(1, lv["harder"] + 1) if split[l]["train"]]
        plan[fam.id] = {"levels": ([0] if lv["headline"] >= 1 else []) + (want or [1, 2]), "n": n, "dev": dev_combos(fam, lv["harder"])}
    return plan


def benchmark(qwen: Qwen, ctx: Ctx, sets: dict[str, list[F.Task]], tag: str, out: Path) -> dict:
    events.emit("stage_start", stage=f"bench_{tag}", msg=f"held-out benchmark, greedy: " + ", ".join(f"{k} {len(v)}" for k, v in sets.items()))
    res = {"model": qwen.describe(), "sets": {}, "rows": {}}
    for name, tasks in sets.items():
        rows = attempt(qwen, ctx, tasks, samples=1, temperature=0.0, stage=f"bench_{tag}", log=out / f"bench_{tag}.jsonl") if tasks else []
        by_fam = defaultdict(list)
        for r in rows:
            by_fam[f"{r['family']} L{r['level']}"].append(r)
        errors = defaultdict(int)
        for r in rows:
            if not r["solved"]:
                errors[(r["error"].split(":")[0] or "wrong answer")[:40]] += 1
        res["sets"][name] = {**rate(rows), "combinations": len({(t.family, tuple(t.rules)) for t in tasks}), "mean_key_share": round(sum(r["key_share"] for r in rows) / len(rows), 3) if rows else None,
                             "by_family": {k: rate(v) for k, v in sorted(by_fam.items())},
                             "failures": dict(sorted(errors.items(), key=lambda kv: -kv[1]))}
        res["rows"][name] = [{"id": r["id"], "solved": r["solved"], "combo": f"{r['family']}|{','.join(r['rules'])}"} for r in rows]
    write_json(out / f"bench_{tag}.json", res)
    h = res["sets"]["headline"]
    events.emit("metric", stage=f"bench_{tag}", name=f"bench_{tag}", value=h["rate"], sets={k: {x: v[x] for x in ("solved", "n", "rate", "wilson95")} for k, v in res["sets"].items()},
                by_family=h["by_family"], msg=f"{tag}: headline {h['solved']}/{h['n']}")
    events.emit("stage_end", stage=f"bench_{tag}", msg=f"headline {h['solved']}/{h['n']}" + (f" ({h['rate']:.0%})" if h["n"] else ""))
    return res


def compare(before: dict, after: dict) -> dict:
    out = {}
    for name in before["sets"]:
        b, a = before["sets"][name], after["sets"][name]
        m = mcnemar(before["rows"][name], after["rows"][name])
        out[name] = {"before": b["rate"], "after": a["rate"], "gain": round(a["rate"] - b["rate"], 3) if b["n"] else None, **m,
                     **by_combination(before["rows"][name], after["rows"][name])}
    # every calibrated task, headline and harder together: the set the significance test is run on,
    # because the headline alone rests on too few rule combinations for one
    bb, aa = (x["rows"]["headline"] + x["rows"].get("harder", []) for x in (before, after))
    rb, ra = rate(bb), rate(aa)
    out["calibrated"] = {"before": rb["rate"], "after": ra["rate"], "gain": round(ra["rate"] - rb["rate"], 3) if bb else None,
                         "n": len(bb), "solved_before": rb["solved"], "solved_after": ra["solved"], **mcnemar(bb, aa), **by_combination(bb, aa)}
    h, c = out["headline"], out["calibrated"]
    out["target"] = {"gain_at_least": TARGET_GAIN, "p_below": TARGET_P,
                     "met": bool(h["gain"] is not None and h["gain"] >= TARGET_GAIN and c["p_by_combination"] < TARGET_P),
                     "note": "this repo's bar for 'post-training worked', not a lab standard: the headline gain, and the "
                             "by-combination paired test on all calibrated tasks (headline + harder)"}
    return out


def run(run_dir: Path, *, bench_n: int, cal_n: int, collect_n: int, k: int, rounds: int,
        targets: tuple[float, float] = (HEADLINE_TARGET, HARDER_TARGET)) -> dict:
    import mlx.core as mx
    out = run_dir / "qwen"
    out.mkdir(parents=True, exist_ok=True)
    events.open_log(run_dir)
    fams = F.load_run(run_dir)
    if not fams:
        raise SystemExit(f"no accepted families in {run_dir / 'families.json'}")
    ctx = Ctx()
    dbs = F.load_dbs(run_dir / "business")
    for fam in fams:
        ctx.add(fam, dbs)
    control = control_ctx(run_dir, ctx)
    report: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dataset": json.loads((run_dir / "business" / "catalog.json").read_text())["source"],
                    "slices": len(dbs), "families": [f.id for f in fams], "control": control.id,
                    "settings": {"bench_n": bench_n, "cal_n": cal_n, "confirm_n": CONFIRM, "collect_n": collect_n, "k": k, "rounds": rounds,
                                 "headline_target": targets[0], "harder_target": targets[1]}}
    events.emit("stage_start", stage="qwen", msg=f"Qwen leg on {len(fams)} families: " + ", ".join(f.id for f in fams))

    def save():
        write_json(out / "report.json", report)
        (out / "REPORT.md").write_text(markdown(report))

    qwen = Qwen()
    report["model"] = qwen.describe()
    events.emit("stage_start", stage="calibrate", msg="start impossible (every rule on), come down until the headline target")
    report["calibrate"] = {fam.id: calibrate(qwen, ctx, fam, out, cal_n, stop_at=targets[0]) for fam in fams}
    write_json(out / "calibrate.json", report["calibrate"])
    events.emit("stage_end", stage="calibrate", msg="in reach: " + (", ".join(f for f, c in report["calibrate"].items() if c["reach"]) or "no family")
                + "; out of reach: " + (", ".join(f for f, c in report["calibrate"].items() if not c["reach"]) or "none"))
    report["settle"] = settle(qwen, ctx, fams, report["calibrate"], bench_n, out, targets)
    write_json(out / "settle.json", report["settle"])
    levels = report["settle"]["levels"]

    sets = bench_sets(ctx, fams, control, levels, bench_n)
    report["before"] = benchmark(qwen, ctx, sets, "before", out)
    save()
    if not sets["headline"]:
        report["stopped"] = "no family is in reach, so there is no headline to train for"
        rounds = 0
    plan = train_plan(fams, levels, collect_n)
    dev = dev_tasks(ctx, plan)
    if rounds and dev:
        report["dev_base"] = dev_check(qwen, ctx, dev, out / "dev_base.jsonl")
        events.emit("metric", stage="bench_before", name="dev", value=report["dev_base"]["rate"], epoch=0,
                    msg=f"untrained: {report['dev_base']['solved']}/{report['dev_base']['n']} dev tasks solved")
    if (out / "NOTES.md").exists():
        report["notes"] = (out / "NOTES.md").read_text()

    kept: list[dict] = []
    for rnd in range(1, rounds + 1):
        have = defaultdict(set)
        for row in kept:
            have[row["family"]] |= set(row["rules"])
        kept += collect(qwen, ctx, plan, out, rnd, k, have=have)
        report[f"collect_r{rnd}"] = json.loads((out / f"collect_r{rnd}.json").read_text())
        if not kept:
            report["stopped"] = "no graded-correct answers to train on"
            events.emit("fail", stage="train", agent="trainer", role="trainer", msg=report["stopped"])
            break
        adapter = out / f"adapter_r{rnd}"
        del qwen
        mx.clear_cache()
        if (adapter / "run.json").exists() and json.loads((adapter / "run.json").read_text()).get("finished"):
            report[f"train_r{rnd}"] = json.loads((adapter / "run.json").read_text())
        else:
            qwen = Qwen()  # every round trains from the base weights on all answers kept so far
            report[f"train_r{rnd}"] = train(qwen, kept, adapter, dev=(lambda q, log: dev_check(q, ctx, dev, log)) if dev else None)
            del qwen
            mx.clear_cache()
        qwen = Qwen(adapter=adapter)
        report[f"after_r{rnd}"] = benchmark(qwen, ctx, sets, f"after_r{rnd}", out)
        report[f"compare_r{rnd}"] = compare(report["before"], report[f"after_r{rnd}"])
        report["last_round"] = rnd
        save()

    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()
    last = report.get("last_round")
    if last:
        c = report[f"compare_r{last}"]["headline"]
        events.emit("metric", stage="qwen", name="result", value=c["after"], before=c["before"], after=c["after"], p=c["p"],
                    msg=f"headline {c['before']:.0%} -> {c['after']:.0%} (p = {c['p']})")
    events.emit("stage_end", stage="qwen", msg="Qwen leg done" if last else "Qwen leg stopped before training")
    print((out / "REPORT.md").read_text())
    return report


def markdown(r: dict) -> str:
    m = r["model"]
    last = r.get("last_round")
    pct = lambda v: f"{v['solved']}/{v['n']} ({v['rate']:.0%}, 95% {v['wilson95'][0]:.0%}-{v['wilson95'][1]:.0%})"
    lines = ["# Qwen post-training on business-data tasks", "",
             f"Model: {m['id']}, weights `{m['weights']}` (quantization {m['quantization']}).",
             f"Dataset: `{r['dataset']}`, {r['slices']} disjoint slices; answers are graded on all of them.",
             f"Task families: {', '.join(r['families'])}. Control (different dataset, never trained on): {r['control']}.", "",
             "## Where the benchmark sits", "",
             "Coming down from the top level (every rule on), on training-pool tasks:", ""]
    for fid, c in r["calibrate"].items():
        lines.append(f"- {fid}: " + ", ".join(f"L{x['level']} {x['solved']}/{x['n']}" for x in c["curve"]) + (f" ({c['note']})" if c["note"] else ""))
    st = r["settle"]
    lines += ["", f"Levels were then adjusted on the held-out benchmark tasks until the before rates sat nearest "
              f"{st['targets']['headline']:.0%} (headline) and {st['targets']['harder']:.0%} (harder):", ""]
    for i, step in enumerate(st["steps"], 1):
        lines.append(f"{i}. " + ", ".join(f"{f} L{h}/L{g}" for f, (h, g) in step["levels"].items())
                     + f": headline {step['headline']:.0%}, harder {step['harder']:.0%}"
                     + (" (part estimated; measured next)" if not step["measured"] else
                        f" (measured; off target, so {', '.join(step['widened_to'])} measured too)" if step.get("widened_to") else " (measured; settled)"))
    lines += ["", "Before, by level, on the benchmark tasks: " + "; ".join(
        f"{f}: " + ", ".join(f"L{l} {v['solved']}/{v['n']}" for l, v in lv.items()) for f, lv in st["before_by_level"].items()) + ".",
        "Targets " + ("reached" if st["reached"] else "NOT reached") + f" within {st['targets']['tolerance']:.0%} points."]
    if not last:
        b = r["before"]["sets"]
        lines += ["", "## Before", ""] + [f"- {k}: {pct(v)}" for k, v in b.items() if v["n"]]
        lines += ["", f"Training has not finished: {r.get('stopped', 'still running')}."]
        return "\n".join(lines) + "\n"
    a, b, c = r[f"after_r{last}"], r["before"], r[f"compare_r{last}"]
    lines += ["", f"## Result after round {last}", "",
              f"After = the same weights + LoRA `{a['model']['adapter']}`. Held-out rule combinations, greedy decoding, all-or-nothing.", "",
              "| set | before | after | tasks gained | tasks lost | rule combinations: up / down of all | p by combination |", "|---|---|---|---|---|---|---|"]
    for name in SETS:
        bb, aa, cc = b["sets"][name], a["sets"][name], c[name]
        if bb["n"]:
            lines.append(f"| {name} | {pct(bb)} | {pct(aa)} | {cc['gained']} | {cc['lost']} | {cc['up']} / {cc['down']} of {cc['combinations']} | {cc['p_by_combination']} |")
    cc, t = c["calibrated"], c["target"]
    lines.append(f"| headline + harder | {cc['solved_before']}/{cc['n']} ({cc['before']:.0%}) | {cc['solved_after']}/{cc['n']} ({cc['after']:.0%}) | {cc['gained']} | {cc['lost']} | "
                 f"{cc['up']} / {cc['down']} of {cc['combinations']} | {cc['p_by_combination']} |")
    lines += ["", "Tasks that share a rule combination differ only in a parameter value, so they are not independent. The p-value therefore "
              "treats each rule combination as one unit (paired sign-flip test on its change in tasks solved). The headline alone rests on "
              f"{c['headline']['combinations']} combinations, too few for any such test to reach 0.05, so significance is judged on headline + harder together.",
              "", f"Bar set by this repo (not a lab standard): headline gain of at least {t['gain_at_least']:.0%} points, and p by combination "
              f"below {t['p_below']} on headline + harder: " + ("**met**." if t["met"] else "**not met**."),
              "", "By family (headline):", ""]
    for fam_level, bb in b["sets"]["headline"]["by_family"].items():
        aa = a["sets"]["headline"]["by_family"].get(fam_level, {})
        lines.append(f"- {fam_level}: {bb['solved']}/{bb['n']} -> {aa.get('solved', '-')}/{aa.get('n', '-')}")
    lines += ["", "Why answers failed (headline), before -> after:", "",
              f"- before: {b['sets']['headline']['failures']}", f"- after: {a['sets']['headline']['failures']}", "",
              "## Round by round", "", "| round | first tries right | right after feedback | answers trained on | rules covered | headline | harder | control |", "|---|---|---|---|---|---|---|---|",
              f"| before | | | | | {pct(b['sets']['headline'])} | " + " | ".join(pct(b["sets"][x]) if b["sets"][x]["n"] else "-" for x in ("harder", "control")) + " |"]
    for rnd in range(1, last + 1):
        col, t, s = r[f"collect_r{rnd}"], r[f"train_r{rnd}"], r[f"after_r{rnd}"]["sets"]
        lines.append(f"| {rnd} | {col['solved']}/{col['sampled']} | {col['solved_after_fix']}/{col['fix_turns']} | {t['rows']} | "
                     + ", ".join(f"{v}" for v in col["rules_with_a_kept_answer"].values()) + f" | {pct(s['headline'])} | "
                     + " | ".join(pct(s[x]) if s[x]["n"] else "-" for x in ("harder", "control")) + " |")
    lines += ["", "Each round samples with the previous round's adapter and trains from the base weights on every answer kept so far."]
    for rnd in range(1, last + 1):
        col = r[f"collect_r{rnd}"]
        if col.get("after_a_traceback"):
            lines.append(f"Round {rnd}, second and third goes (solved/tried): {col['after_a_traceback']} after the answer's own traceback, "
                         f"{col['after_graded_wrong']} after being told only that it was graded wrong.")
        if r[f"collect_r{rnd}"].get("explored"):
            lines.append(f"Round {rnd}, extra samples for rules that had no kept answer (solved/sampled): "
                         + "; ".join(f"{k} {v}" for k, v in r[f"collect_r{rnd}"]["explored"].items()) + ".")
    lines += ["", "## Training", ""]
    if r.get("dev_base"):
        d = r["dev_base"]
        lines += [f"How long to train is decided on {d['n']} dev tasks: training rule combinations set aside, never trained on and never in the "
                  f"benchmark. The untrained model solves {d['solved']}/{d['n']} of them ({d['crashed']} crash). After each epoch the model answers "
                  "them greedily; the epoch that solves most is kept, and training stops once an epoch does worse than an earlier one.", ""]
    for rnd in range(1, last + 1):
        t = r[f"train_r{rnd}"]
        line = (f"- round {rnd}: {t['rows']} answers ({t['rows_by_family']}), learning rate {t.get('settings', {}).get('learning_rate', '?')}, "
                f"{t['steps']} steps, loss {t['first_loss']} -> {t['last_loss']}, {t['seconds'] / 60:.0f} min, peak {t['peak_memory_gb']} GB")
        if any("dev" in e for e in t.get("epochs", [])):
            line += ("; dev tasks solved after each epoch: " + ", ".join(f"{e['dev']['solved']}/{e['dev']['n']} ({e['dev']['crashed']} crash)" for e in t["epochs"])
                     + f"; epoch {t['chosen_epoch']} kept")
        lines.append(line)
    if r.get("notes"):
        lines += ["", "## Earlier attempts", "", r["notes"].strip()]
    return "\n".join(lines) + "\n"


def smoke(run_dir: Path) -> None:
    """A few answers per family at the easy end, batch and fix paths included, to see where Qwen stands."""
    out = run_dir / "qwen_smoke"
    out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx()
    dbs = F.load_dbs(run_dir / "business")
    fams = F.load_run(run_dir)
    for fam in fams:
        ctx.add(fam, dbs)
    qwen = Qwen()
    tasks = [t for fam in fams for lvl in (0, 1) for t in F.make_tasks(fam, lvl, 1, "bench", 5, dbs[0])]
    rows = attempt(qwen, ctx, tasks, samples=1, temperature=0.0, stage="smoke", log=out / "smoke.jsonl")
    for r in rows:
        print(f"{r['id']}: {'SOLVED' if r['solved'] else 'fail'} keys {r['key_share']} {r['error'][:90]}", flush=True)
    rows = attempt(qwen, ctx, tasks[:2], samples=4, temperature=0.8, stage="smoke", log=out / "smoke_sampled.jsonl")
    print("sampled:", [r["solved"] for r in rows], flush=True)
    for r in repair(qwen, ctx, rows, turns=1, temperature=0.8, stage="smoke", log=out / "smoke_fix.jsonl"):
        print(f"fix of {r['id']} after {r['crash'][:60]}: {'SOLVED' if r['solved'] else 'fail'} {r['error'][:80]}", flush=True)


def smoke_train(run_dir: Path) -> None:
    """The training path end to end in a few minutes: two epochs on four kept answers with the dev check
    between them, then the saved adapter loaded from disk, which must answer as the trained model did."""
    import mlx.core as mx
    out = run_dir / "qwen_smoke"
    out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx()
    dbs = F.load_dbs(run_dir / "business")
    fams = F.load_run(run_dir)
    for fam in fams:
        ctx.add(fam, dbs)
    kept = json.loads(sorted((run_dir / "qwen").rglob("kept_r*.json"))[0].read_text())[:4]
    tasks = F.make_tasks(ctx.fams[kept[0]["family"]], 1, 2, "train", 9, dbs[0])
    LORA["epochs"] = 2
    qwen = Qwen()
    res = train(qwen, kept, out / "adapter_smoke", dev=lambda q, log: dev_check(q, ctx, tasks, log))
    print(json.dumps({k: res[k] for k in ("epochs", "chosen_epoch", "seconds", "peak_memory_gb")}), flush=True)
    del qwen
    mx.clear_cache()
    qwen = Qwen(adapter=out / "adapter_smoke")
    again = attempt(qwen, ctx, tasks, samples=1, temperature=0.0, stage="smoke", log=out / "adapter_smoke" / "reload.jsonl")
    then = {r["id"]: r["answer"] for r in read_jsonl(out / "adapter_smoke" / f"epoch_{res['chosen_epoch']}" / "dev.jsonl")}
    print("reloaded adapter answers as the trained model did:", [r["answer"] == then[r["id"]] for r in again], flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder under runs/ holding business/ and families.json")
    ap.add_argument("--bench-n", type=int, default=20, help="held-out headline tasks per family")
    ap.add_argument("--cal-n", type=int, default=5, help="screening tasks per level; from the first solve down every level gets 20")
    ap.add_argument("--collect-n", type=int, default=6, help="training tasks per level per family per round")
    ap.add_argument("--k", type=int, default=4, help="samples per training task")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--headline-target", type=float, default=HEADLINE_TARGET, help="before pass rate the headline set should sit at")
    ap.add_argument("--harder-target", type=float, default=HARDER_TARGET, help="before pass rate the harder set should sit at")
    ap.add_argument("--smoke", action="store_true", help="a few answers per family and exit")
    ap.add_argument("--smoke-train", action="store_true", help="a two-epoch training on four kept answers, dev check and reload, and exit")
    a = ap.parse_args()
    if a.smoke_train:
        return smoke_train(CONFIG.runs_dir / a.run)
    if a.smoke:
        return smoke(CONFIG.runs_dir / a.run)
    run(CONFIG.runs_dir / a.run, bench_n=a.bench_n, cal_n=a.cal_n, collect_n=a.collect_n, k=a.k, rounds=a.rounds,
        targets=(a.headline_target, a.harder_target))


if __name__ == "__main__":
    main()
