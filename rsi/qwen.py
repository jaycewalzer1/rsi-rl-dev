"""Post-train a local Qwen on a run's task families and measure it before and after.

    calibrate      per family, start at the top level (every rule on, "impossible") and come
                   down until Qwen's pass rate reaches the band (RSI_BAND_LOW..HIGH, target 25%).
                   That level, L*, is where the benchmark sits.
    bench before   held-out tasks, greedy decoding, all-or-nothing on every data slice:
                     headline  L* in every family            (the ~25% starting point)
                     harder    L*+1 in every family          (does the curve move, not one point)
                     control   a family from a DIFFERENT dataset nothing here trains on
    collect        rejection sampling on TRAINING rule combinations at L*-1..L*+1: k samples
                   per task, keep only answers the oracle grades fully right. The reward is the
                   only teacher; no reference solution is ever shown to the model.
    train          LoRA on the kept answers (mlx-vlm SFT trainer, loss on the answer only)
    bench after    the same held-out tasks with the adapter; paired test against before

Benchmark rule combinations are never trained on (family.split_combos), and every answer is
graded on slices of the business's data the model was never shown rows from.

Every stage writes its rows as it goes and picks up where it left off, so a stopped job
resumes. It loads Qwen, so it runs with kokopelli's venv (mlx-vlm), only through its GPU queue:

    uv run rsi qwen --run NAME            # queues it; see cli.py for the exact command

Every result file names the model and weights it used.
"""

from __future__ import annotations

import argparse
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
        "learning_rate": 1e-4, "batch_size": 1, "max_seq_length": 6144, "epochs": 2, "max_rows": 320, "seed": 0}
# the bar this repo sets for "post-training worked"; not a lab standard
TARGET_GAIN, TARGET_P = 0.20, 0.05


class Qwen:
    def __init__(self, base: Path = BASE, adapter: Path | None = None):
        from mlx_vlm import load
        self.base, self.adapter = base, adapter
        self.model, self.processor = load(str(base), adapter_path=str(adapter) if adapter else None)

    def describe(self) -> dict:
        q = json.loads((self.base / "config.json").read_text()).get("quantization")
        return {"id": MODEL_ID, "weights": str(self.base), "quantization": q,
                "adapter": str(self.adapter) if self.adapter else None}

    def _fmt(self, user: str) -> str:
        from mlx_vlm.prompt_utils import apply_chat_template
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        return apply_chat_template(self.processor, self.model.config, msgs, num_images=0)

    def generate(self, prompts: list[str], temperature: float, max_tokens: int = 2000, batch: int = 8):
        """Yields (index, text) as chunks finish."""
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

def calibrate(qwen: Qwen, ctx: Ctx, fam: F.Family, out: Path, n: int = 5) -> dict:
    """Come down from the top level until the pass rate reaches the band; return L*."""
    db = ctx.dbs[fam.id][0]
    top = len(fam.rules)
    order = [l for l in dict.fromkeys([top, top - 2, top - 4, *range(top - 5, -1, -1)]) if l >= 0]
    measured: dict[int, dict] = {}

    def measure(level: int, count: int) -> dict:
        pool = "train" if F.split_combos(fam)[level]["train"] else "bench"
        tasks = F.make_tasks(fam, level, count, pool, 9000, db)
        rows = attempt(qwen, ctx, tasks, samples=1, temperature=0.0, stage="calibrate", log=out / "calibrate.jsonl")
        measured[level] = rate(rows)
        events.emit("metric", stage="calibrate", name="calibrate", family=fam.id, level=level,
                    value=measured[level]["rate"], n=len(rows), msg=f"{fam.id} level {level}: {measured[level]['solved']}/{len(rows)}")
        return measured[level]

    hit = None
    for level in order:
        m = measure(level, n)
        if m["solved"] >= 1:
            m = measure(level, 2 * n)
            if m["rate"] >= CONFIG.band_low:
                hit = level
                break
    if hit is not None and hit + 1 <= top and (hit + 1 not in measured or measured[hit + 1]["n"] < 2 * n):
        measure(hit + 1, 2 * n)  # a skipped level may be the one nearer the target
    solid = {l: m for l, m in measured.items() if m["n"] >= 2 * n and m["solved"] >= 1}
    if solid:
        level = min(solid, key=lambda l: (abs(solid[l]["rate"] - CONFIG.target_rate), -l))
        note = "level whose measured pass rate is nearest the target"
    else:
        level, note = 0, "no level reached the band: Qwen solved nothing, or too little, even with every rule off"
    return {"family": fam.id, "level": level, "note": note,
            "curve": [{"level": l, **m} for l, m in sorted(measured.items(), reverse=True)]}


# ---------------------------------------------------------------- collect and train

def collect(qwen: Qwen, ctx: Ctx, plan: dict[str, list[int]], out: Path, rnd: int, n: int, k: int) -> list[dict]:
    events.emit("stage_start", stage="collect", msg=f"round {rnd}: {k} samples per training task, keep graded-correct answers")
    tasks = [t for fid, levels in plan.items() for lvl in levels
             for t in F.make_tasks(ctx.fams[fid], lvl, n, "train", 3000 + rnd, ctx.dbs[fid][0])]
    rows = attempt(qwen, ctx, tasks, samples=k, temperature=0.8, stage="collect", log=out / f"collect_r{rnd}.jsonl")
    kept, per_task = [], defaultdict(set)
    for r in rows:
        code = F.extract_code(r["answer"]).strip()
        if r["solved"] and code not in per_task[r["id"]] and len(per_task[r["id"]]) < 2:
            per_task[r["id"]].add(code)
            fam = ctx.fams[r["family"]]
            task = F.Task(fam.id, tuple(r["rules"]), r["params"])
            kept.append({"messages": [{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": F.prompt(fam, task, ctx.dbs[fam.id][0])},
                                      {"role": "assistant", "content": f"```python\n{code}\n```"}],
                         "task": r["id"], "family": fam.id, "level": r["level"]})
    by = defaultdict(list)
    for r in rows:
        by[f"{r['family']} L{r['level']}"].append(r)
    summary = {"model": qwen.describe(), "plan": plan, "sampled": len(rows), "solved": sum(r["solved"] for r in rows),
               "kept": len(kept), "tasks": len(tasks), "tasks_with_a_solve": len(per_task),
               "by_family_level": {key: rate(v) for key, v in sorted(by.items())}}
    write_json(out / f"collect_r{rnd}.json", summary)
    events.emit("stage_end", stage="collect", msg=f"{len(kept)} graded-correct answers kept from {len(rows)} samples", kept=len(kept))
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


def bench_sets(ctx: Ctx, fams: list[F.Family], control: F.Family, levels: dict[str, int], n: int) -> dict[str, list[F.Task]]:
    sets: dict[str, list[F.Task]] = {"headline": [], "harder": [], "control": []}
    for fam in fams:
        db, top, lvl = ctx.dbs[fam.id][0], len(fam.rules), levels[fam.id]
        sets["headline"] += F.make_tasks(fam, lvl, n, "bench", 1000, db)
        if lvl + 1 <= top:
            sets["harder"] += F.make_tasks(fam, lvl + 1, max(4, n // 2), "bench", 1000, db)
    for lvl in (1, 2):
        sets["control"] += F.make_tasks(control, lvl, 8, "bench", 1000, ctx.dbs[control.id][0])
    return sets


def benchmark(qwen: Qwen, ctx: Ctx, sets: dict[str, list[F.Task]], tag: str, out: Path) -> dict:
    events.emit("stage_start", stage=f"bench_{tag}", msg=f"held-out benchmark, greedy: " + ", ".join(f"{k} {len(v)}" for k, v in sets.items()))
    res = {"model": qwen.describe(), "sets": {}, "rows": {}}
    for name, tasks in sets.items():
        rows = attempt(qwen, ctx, tasks, samples=1, temperature=0.0, stage=f"bench_{tag}", log=out / f"bench_{tag}.jsonl")
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
    events.emit("stage_end", stage=f"bench_{tag}", msg=f"headline {h['rate']:.0%} ({h['solved']}/{h['n']})")
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


def run(run_dir: Path, *, bench_n: int, cal_n: int, collect_n: int, k: int, rounds: int) -> dict:
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
                    "settings": {"bench_n": bench_n, "cal_n": cal_n, "collect_n": collect_n, "k": k, "rounds": rounds,
                                 "target_rate": CONFIG.target_rate, "band": [CONFIG.band_low, CONFIG.band_high]}}
    events.emit("stage_start", stage="qwen", msg=f"Qwen leg on {len(fams)} families: " + ", ".join(f.id for f in fams))

    qwen = Qwen()
    report["model"] = qwen.describe()
    events.emit("stage_start", stage="calibrate", msg="start impossible (every rule on), come down until the band")
    report["calibrate"] = {fam.id: calibrate(qwen, ctx, fam, out, cal_n) for fam in fams}
    write_json(out / "calibrate.json", report["calibrate"])
    levels = {fid: c["level"] for fid, c in report["calibrate"].items()}
    events.emit("stage_end", stage="calibrate", msg="benchmark levels: " + ", ".join(f"{f} L{l}" for f, l in levels.items()), levels=levels)

    sets = bench_sets(ctx, fams, control, levels, bench_n)
    report["before"] = benchmark(qwen, ctx, sets, "before", out)
    plan = {}
    for fam in fams:
        trainable = [l for l in fam.levels if F.split_combos(fam)[l]["train"]]
        want = [l for l in (levels[fam.id] - 1, levels[fam.id], levels[fam.id] + 1) if l in trainable]
        plan[fam.id] = want or sorted(trainable)[:2]

    kept: list[dict] = []
    for rnd in range(1, rounds + 1):
        kept += collect(qwen, ctx, plan, out, rnd, collect_n, k)
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
            qwen = Qwen()
            report[f"train_r{rnd}"] = train(qwen, kept, adapter)
            del qwen
            mx.clear_cache()
        qwen = Qwen(adapter=adapter)
        report[f"after_r{rnd}"] = benchmark(qwen, ctx, sets, f"after_r{rnd}", out)
        report[f"compare_r{rnd}"] = compare(report["before"], report[f"after_r{rnd}"])
        report["last_round"] = rnd
        write_json(out / "report.json", report)
        (out / "REPORT.md").write_text(markdown(report))

    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(out / "report.json", report)
    (out / "REPORT.md").write_text(markdown(report))
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
    lines = ["# Qwen post-training on business-data tasks", "",
             f"Model: {m['id']}, weights `{m['weights']}` (quantization {m['quantization']}).",
             f"Dataset: `{r['dataset']}`, {r['slices']} disjoint slices; answers are graded on all of them.",
             f"Task families: {', '.join(r['families'])}. Control (different dataset, never trained on): {r['control']}.", "",
             "## Where the benchmark sits", "",
             "Coming down from the top level (every rule on) until the pass rate reaches the band:", ""]
    for fid, c in r["calibrate"].items():
        lines.append(f"- {fid}: " + ", ".join(f"L{x['level']} {x['solved']}/{x['n']}" for x in c["curve"]) + f" -> benchmark at level {c['level']} ({c['note']})")
    if not last:
        b = r["before"]["sets"]
        lines += ["", "## Before", ""] + [f"- {k}: {v['solved']}/{v['n']} ({v['rate']:.0%})" for k, v in b.items() if v["n"]]
        lines += ["", f"Training did not run: {r.get('stopped', 'not finished')}."]
        return "\n".join(lines) + "\n"
    a, b, c = r[f"after_r{last}"], r["before"], r[f"compare_r{last}"]
    lines += ["", f"## Result after round {last}", "",
              f"After = the same weights + LoRA `{a['model']['adapter']}`. Held-out rule combinations, greedy decoding, all-or-nothing.", "",
              "| set | before | after | gained | lost | paired p |", "|---|---|---|---|---|---|"]
    for name in ("headline", "harder", "control"):
        bb, aa, cc = b["sets"][name], a["sets"][name], c[name]
        if not bb["n"]:
            continue
        lines.append(f"| {name} | {bb['solved']}/{bb['n']} ({bb['rate']:.0%}, 95% {bb['wilson95'][0]:.0%}-{bb['wilson95'][1]:.0%}) "
                     f"| {aa['solved']}/{aa['n']} ({aa['rate']:.0%}, 95% {aa['wilson95'][0]:.0%}-{aa['wilson95'][1]:.0%}) "
                     f"| {cc['gained']} | {cc['lost']} | {cc['p']} |")
    t = c["target"]
    lines += ["", f"Bar set by this repo (not a lab standard): headline gain of at least {t['gain_at_least']:.0%} points with paired p below {t['p_below']}: "
              + ("**met**." if t["met"] else "**not met**."), "", "By family (headline):", ""]
    for fam_level, bb in b["sets"]["headline"]["by_family"].items():
        aa = a["sets"]["headline"]["by_family"].get(fam_level, {})
        lines.append(f"- {fam_level}: {bb['solved']}/{bb['n']} -> {aa.get('solved', '-')}/{aa.get('n', '-')}")
    lines += ["", "Why answers failed (headline), before -> after:", "",
              f"- before: {b['sets']['headline']['failures']}", f"- after: {a['sets']['headline']['failures']}", "", "## Training", ""]
    for rnd in range(1, last + 1):
        col, t = r[f"collect_r{rnd}"], r[f"train_r{rnd}"]
        lines.append(f"- round {rnd}: {col['sampled']} samples on {col['tasks']} training tasks, {col['solved']} graded correct, "
                     f"{t['rows']} kept for training ({t['rows_by_family']}); {t['steps']} steps, loss {t['first_loss']} -> {t['last_loss']}, "
                     f"{t['seconds'] / 60:.0f} min, peak {t['peak_memory_gb']} GB")
    return "\n".join(lines) + "\n"


def smoke(run_dir: Path) -> None:
    """A few answers per family at the easy end, batch path included, to see where Qwen stands."""
    out = run_dir / "qwen_smoke"
    out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx()
    dbs = F.load_dbs(run_dir / "business")
    fams = F.load_run(run_dir)
    for fam in fams:
        ctx.add(fam, dbs)
    qwen = Qwen()
    tasks = [t for fam in fams for lvl in (0, 1, 2) for t in F.make_tasks(fam, lvl, 2, "bench", 5, dbs[0])]
    rows = attempt(qwen, ctx, tasks, samples=1, temperature=0.0, stage="smoke", log=out / "smoke.jsonl")
    for r in rows:
        print(f"{r['id']}: {'SOLVED' if r['solved'] else 'fail'} keys {r['key_share']} {r['error'][:90]}", flush=True)
    rows = attempt(qwen, ctx, tasks[:2], samples=3, temperature=0.8, stage="smoke", log=out / "smoke_sampled.jsonl")
    print("sampled:", [r["solved"] for r in rows], flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder under runs/ holding business/ and families.json")
    ap.add_argument("--bench-n", type=int, default=20, help="held-out headline tasks per family")
    ap.add_argument("--cal-n", type=int, default=5, help="calibration tasks per level (doubled at a level with a solve)")
    ap.add_argument("--collect-n", type=int, default=6, help="training tasks per level per family per round")
    ap.add_argument("--k", type=int, default=4, help="samples per training task")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="a few answers per family and exit")
    a = ap.parse_args()
    if a.smoke:
        return smoke(CONFIG.runs_dir / a.run)
    run(CONFIG.runs_dir / a.run, bench_n=a.bench_n, cal_n=a.cal_n, collect_n=a.collect_n, k=a.k, rounds=a.rounds)


if __name__ == "__main__":
    main()
