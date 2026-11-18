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
                   back and another go (twice at most). Only answers the oracle grades fully
                   right are kept. The reward is the only teacher: no reference solution and no
                   expected value is ever shown to the model.
    train          LoRA on every answer kept so far, each as ticket -> final correct code
                   (mlx-vlm SFT trainer, loss on the answer only), from the base weights
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
        "learning_rate": 1e-4, "batch_size": 1, "max_seq_length": 6144, "epochs": 3, "max_rows": 400, "seed": 0}
# the bar this repo sets for "post-training worked"; not a lab standard
TARGET_GAIN, TARGET_P = 0.20, 0.05
CONFIRM = 20  # tasks behind a calibration level before it can be chosen
HEADLINE_TARGET, HARDER_TARGET, TOLERANCE = 0.50, 0.25, 0.10  # where the BEFORE pass rates should sit
FIX = ("Running your function on the sample tables raised:\n\n{trace}\n\n"
       "Fix it. Reply with the complete corrected function in one ```python block.")


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
    """Give each crashed answer its own traceback and another go; returns the new rows.

    A crash is whatever running the answer on the shown slice raises (family.crash_on_shown).
    Answers that run but are wrong get nothing back: telling the model how would leak the oracle."""
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
            trace = F.crash_on_shown(task, r["answer"], ctx.dbs[r["family"]][0])
            if trace:
                todo.append((r, task, trace))
        started = time.time()
        if todo:
            chats = [[{"role": "user", "content": F.prompt(ctx.fams[t.family], t, ctx.dbs[t.family][0])},
                      {"role": "assistant", "content": r["answer"]},
                      {"role": "user", "content": FIX.format(trace=trace)}] for r, t, trace in todo]
            with log.open("a") as f:
                for i, text in qwen.generate(chats, temperature):
                    r, t, trace = todo[i]
                    fam = ctx.fams[t.family]
                    g = F.score(fam, t, text, ctx.dbs[t.family])
                    row = {**t.as_dict(fam), "sample": r["sample"], "turn": turn, "temperature": temperature,
                           "crash": trace.splitlines()[-1][:200], **g, "answer": text}
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    nxt.append(row)
                    events.emit("work", stage=stage, agent=f"qwen-{t.id}-{r['sample']}-fix{turn}", role="qwen",
                                msg=f"{t.id} fix {turn} after {row['crash'][:50]}: {'SOLVED' if g['solved'] else 'fail'}",
                                level=t.level, family=t.family, solved=g["solved"])
        print(f"{stage} fix {turn}: {sum(r['solved'] for r in nxt)}/{len(nxt)} crashed answers now solved"
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

def collect(qwen: Qwen, ctx: Ctx, plan: dict[str, dict], out: Path, rnd: int, k: int, fixes: int = 2) -> list[dict]:
    """plan: family -> {"levels": [...], "n": tasks per level}. Returns the graded-correct answers as training rows."""
    events.emit("stage_start", stage="collect", msg=f"round {rnd}: {k} samples per training task, crashed answers get {fixes} fixes, keep graded-correct answers")
    tasks = []
    for fid, want in plan.items():
        fam = ctx.fams[fid]
        for lvl in want["levels"]:
            pool = "train" if F.split_combos(fam)[lvl]["train"] else "bench"  # level 0 has one combination; the plan only lists it when no benchmark uses it
            tasks += F.make_tasks(fam, lvl, want["n"], pool, 3000 + rnd, ctx.dbs[fid][0])
    first = attempt(qwen, ctx, tasks, samples=k, temperature=0.8, stage="collect", log=out / f"collect_r{rnd}.jsonl")
    fixed = repair(qwen, ctx, first, turns=fixes, temperature=0.8, stage="collect", log=out / f"collect_r{rnd}_fix.jsonl")
    kept, per_task = [], defaultdict(set)
    for r in first + fixed:
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
               "fix_turns": len(fixed), "solved_after_fix": sum(r["solved"] for r in fixed),
               "kept": len(kept), "tasks": len(tasks), "tasks_with_a_solve": len(per_task),
               "first_try_by_family_level": {key: rate(v) for key, v in sorted(by.items())},
               "rules_with_a_kept_answer": {fid: f"{len(covered[fid])}/{len(ctx.fams[fid].rules)}" for fid in plan}}
    write_json(out / f"collect_r{rnd}.json", summary)
    write_json(out / f"kept_r{rnd}.json", kept)
    events.emit("metric", stage="collect", name="collect", round=rnd, value=len(kept), sampled=len(first), solved=summary["solved"],
                fixed=summary["solved_after_fix"], msg=f"round {rnd}: {summary['solved']}/{len(first)} first tries right, {summary['solved_after_fix']} more after a fix")
    events.emit("stage_end", stage="collect", msg=f"{len(kept)} graded-correct answers kept from {len(first)} samples and {len(fixed)} fixes", kept=len(kept))
    return kept


def train(qwen: Qwen, rows: list[dict], adapter_dir: Path) -> dict:
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_vlm.trainer.api import Trainer
    from mlx_vlm.trainer.common.model import prepare_model_for_training
    from mlx_vlm.trainer.vlm.config import VLMTrainingArgs
    rng = random.Random(LORA["seed"])
    if len(rows) > LORA["max_rows"]:
        rows = rng.sample(rows, LORA["max_rows"])
    events.emit("stage_start", stage="train", msg=f"LoRA r{LORA['rank']} on {len(rows)} answers x {LORA['epochs']} epochs")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(2 * 1024 ** 3)  # the trainer's buffer cache grew to 32 GB on a 36 GB Mac without this
    mx.random.seed(LORA["seed"])
    data = [{"messages": r["messages"]} for r in rows] * LORA["epochs"]
    rng.shuffle(data)
    prepare_model_for_training(qwen.model, train_type="lora", lora_rank=LORA["rank"], lora_alpha=LORA["alpha"],
                               lora_dropout=LORA["dropout"], target_modules=LORA["targets"])
    args = VLMTrainingArgs(batch_size=LORA["batch_size"], iters=len(data), steps_per_report=5, steps_per_eval=None,
                           steps_per_save=10 ** 9, val_batches=-1, adapter_file=str(adapter_dir / "adapters.safetensors"),
                           max_seq_length=LORA["max_seq_length"], train_on_completions=True)
    trainer = Trainer(qwen.model, args, train_dataset=data, processing_class=qwen.processor,
                      optimizer=optim.AdamW(learning_rate=LORA["learning_rate"]), seed=LORA["seed"])
    started = time.time()
    trainer.train()
    trainer.save_adapter(str(adapter_dir / "adapters.safetensors"))
    hist = [h for h in trainer.log_history if "train_loss" in h]
    for h in hist[:: max(1, len(hist) // 30)]:
        events.emit("metric", stage="train", msg=f"loss {h['train_loss']:.3f}", name="train_loss", value=h["train_loss"])
    by_fam = defaultdict(int)
    for r in rows:
        by_fam[r["family"]] += 1
    res = {"base": qwen.describe(), "settings": LORA, "rows": len(rows), "rows_by_family": dict(by_fam), "steps": len(data),
           "seconds": round(time.time() - started, 1),
           "first_loss": round(hist[0]["train_loss"], 4) if hist else None,
           "last_loss": round(sum(h["train_loss"] for h in hist[-3:]) / len(hist[-3:]), 4) if hist else None,
           "peak_memory_gb": round(max((h.get("peak_memory", 0) for h in trainer.log_history), default=0), 2),
           "trainer": "mlx-vlm SFT (mlx_vlm.trainer), loss on the answer only", "finished": True}
    write_json(adapter_dir / "run.json", res)
    events.emit("stage_end", stage="train", msg=f"loss {res['first_loss']} -> {res['last_loss']} in {res['seconds'] / 60:.0f} min")
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
    rule combination a benchmark uses. An out-of-reach family gets a few level-1 tasks, in case
    what the model learns on the others carries over."""
    plan = {}
    for fam in fams:
        split = F.split_combos(fam)
        if fam.id not in levels:
            plan[fam.id] = {"levels": [1], "n": max(2, n // 2)}
            continue
        lv = levels[fam.id]
        want = [l for l in range(1, lv["harder"] + 1) if split[l]["train"]]
        plan[fam.id] = {"levels": ([0] if lv["headline"] >= 1 else []) + (want or [1, 2]), "n": n}
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
        res["sets"][name] = {**rate(rows), "mean_key_share": round(sum(r["key_share"] for r in rows) / len(rows), 3) if rows else None,
                             "by_family": {k: rate(v) for k, v in sorted(by_fam.items())},
                             "failures": dict(sorted(errors.items(), key=lambda kv: -kv[1]))}
        res["rows"][name] = [{"id": r["id"], "solved": r["solved"]} for r in rows]
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
        out[name] = {"before": b["rate"], "after": a["rate"], "gain": round(a["rate"] - b["rate"], 3) if b["n"] else None, **m}
    h = out["headline"]
    out["target"] = {"gain_at_least": TARGET_GAIN, "p_below": TARGET_P,
                     "met": bool(h["gain"] is not None and h["gain"] >= TARGET_GAIN and h["p"] < TARGET_P),
                     "note": "this repo's bar for 'post-training worked', not a lab standard"}
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

    kept: list[dict] = []
    for rnd in range(1, rounds + 1):
        kept += collect(qwen, ctx, plan, out, rnd, k)
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
            report[f"train_r{rnd}"] = train(qwen, kept, adapter)
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
              "| set | before | after | gained | lost | paired p |", "|---|---|---|---|---|---|"]
    for name in SETS:
        bb, aa, cc = b["sets"][name], a["sets"][name], c[name]
        if bb["n"]:
            lines.append(f"| {name} | {pct(bb)} | {pct(aa)} | {cc['gained']} | {cc['lost']} | {cc['p']} |")
    t = c["target"]
    lines += ["", f"Bar set by this repo (not a lab standard): headline gain of at least {t['gain_at_least']:.0%} points with paired p below {t['p_below']}: "
              + ("**met**." if t["met"] else "**not met**."), "", "By family (headline):", ""]
    for fam_level, bb in b["sets"]["headline"]["by_family"].items():
        aa = a["sets"]["headline"]["by_family"].get(fam_level, {})
        lines.append(f"- {fam_level}: {bb['solved']}/{bb['n']} -> {aa.get('solved', '-')}/{aa.get('n', '-')}")
    lines += ["", "Why answers failed (headline), before -> after:", "",
              f"- before: {b['sets']['headline']['failures']}", f"- after: {a['sets']['headline']['failures']}", "",
              "## Round by round", "", "| round | first tries right | right after a fix | answers trained on | rules covered | headline | harder | control |", "|---|---|---|---|---|---|---|---|",
              f"| before | | | | | {pct(b['sets']['headline'])} | " + " | ".join(pct(b["sets"][x]) if b["sets"][x]["n"] else "-" for x in ("harder", "control")) + " |"]
    for rnd in range(1, last + 1):
        col, t, s = r[f"collect_r{rnd}"], r[f"train_r{rnd}"], r[f"after_r{rnd}"]["sets"]
        lines.append(f"| {rnd} | {col['solved']}/{col['sampled']} | {col['solved_after_fix']}/{col['fix_turns']} | {t['rows']} | "
                     + ", ".join(f"{v}" for v in col["rules_with_a_kept_answer"].values()) + f" | {pct(s['headline'])} | "
                     + " | ".join(pct(s[x]) if s[x]["n"] else "-" for x in ("harder", "control")) + " |")
    lines += ["", "Each round samples with the previous round's adapter and trains from the base weights on every answer kept so far.", "", "## Training", ""]
    for rnd in range(1, last + 1):
        t = r[f"train_r{rnd}"]
        lines.append(f"- round {rnd}: {t['rows']} answers ({t['rows_by_family']}), {t['steps']} steps, loss {t['first_loss']} -> {t['last_loss']}, "
                     f"{t['seconds'] / 60:.0f} min, peak {t['peak_memory_gb']} GB")
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
    a = ap.parse_args()
    if a.smoke:
        return smoke(CONFIG.runs_dir / a.run)
    run(CONFIG.runs_dir / a.run, bench_n=a.bench_n, cal_n=a.cal_n, collect_n=a.collect_n, k=a.k, rounds=a.rounds,
        targets=(a.headline_target, a.harder_target))


if __name__ == "__main__":
    main()
