"""Post-train a local Qwen on the business tasks: benchmark, anneal, train, benchmark again.

    bench before   held-out rule combinations at every level 8..1, greedy, one answer each
    anneal         start at level 8 (all rules, "impossible"), step down one rule at a time
                   until Qwen's sampled pass rate reaches the band (RSI_BAND_LOW, target 25%)
    collect        rejection sampling on TRAINING combinations around that level: k samples
                   each, keep only answers the oracle grades fully right (expert iteration)
    train          LoRA on the kept answers with mlx-vlm's SFT trainer (loss on the answer only)
    bench after    the same held-out benchmark, base weights + the adapter

Rounds of collect + train repeat with --rounds; each round samples from the adapter so far.
Benchmark combinations are never trained on (bizops.split_combos), so "after" measures
transfer to rule stacks the model never saw graded.

It loads Qwen, so it runs with kokopelli's venv (mlx-vlm) and only through its GPU queue:

    python3 ~/Desktop/kokopelli/kokopelli/ops/gpu.py run --name rsi_qwen_bizops --priority 50 --gb 18 \\
      --cwd ~/Desktop/rsi-rl-dev --about "..." --output runs/<run>/qwen -- \\
      ~/Desktop/kokopelli/.venv/bin/python -u -m rsi.qwen --run <run>

Every result file names the model and weights it used.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from . import bizops, events
from .config import CONFIG
from .store import write_json

KOKO = Path(os.environ.get("RSI_KOKOPELLI", Path.home() / "Desktop" / "kokopelli"))
BASE = Path(os.environ.get("RSI_QWEN_WEIGHTS", KOKO / "models" / "Qwen3-VL-8B-Instruct-4bit"))
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
SYSTEM = "You are a careful Python engineer. Follow every rule in the ticket exactly."
LORA = {"rank": 16, "alpha": 32.0, "dropout": 0.0, "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "learning_rate": 1e-4, "batch_size": 1, "max_seq_length": 4096, "epochs": 2, "seed": 0}


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

    def generate(self, prompts: list[str], temperature: float, max_tokens: int = 2200, batch: int = 6) -> list[str]:
        import mlx.core as mx
        from mlx_vlm import batch_generate, generate
        from mlx_vlm.sample_utils import make_sampler
        sampling = {"sampler": make_sampler(temperature)} if temperature > 0 else {}
        out = []
        for i in range(0, len(prompts), batch):
            chunk = [self._fmt(p) for p in prompts[i:i + batch]]
            try:
                r = batch_generate(self.model, self.processor, prompts=chunk, max_tokens=max_tokens,
                                   verbose=False, **sampling)
                out += list(r.texts)
            except Exception as e:  # batch path is newer; fall back to one at a time
                print(f"batch_generate failed ({type(e).__name__}: {e}); generating one by one", flush=True)
                for c in chunk:
                    out.append(generate(self.model, self.processor, c, max_tokens=max_tokens,
                                        temperature=temperature, verbose=False).text)
            mx.clear_cache()
            print(f"  generated {len(out)}/{len(prompts)}", flush=True)
        return out


def attempt(qwen: Qwen, tasks: list[bizops.Task], db: dict, dbs: list[dict], *, samples: int, temperature: float,
            stage: str, log: Path) -> list[dict]:
    prompts, owners = [], []
    for t in tasks:
        for s in range(samples):
            prompts.append(bizops.prompt(t, db))
            owners.append((t, s))
    started = time.time()
    texts = qwen.generate(prompts, temperature)
    rows = []
    for (t, s), text in zip(owners, texts):
        g = bizops.score(t, text, dbs)
        row = {**t.as_dict(), "sample": s, "temperature": temperature, **g, "answer": text}
        rows.append(row)
        events.emit("work", stage=stage, agent=f"qwen-{t.id}-{s}", role="qwen",
                    msg=f"{t.id} {'SOLVED' if g['solved'] else 'fail'} ({g['order_share']:.0%} orders)",
                    level=t.level, solved=g["solved"])
    with log.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{stage}: {sum(r['solved'] for r in rows)}/{len(rows)} solved in {time.time() - started:.0f} s", flush=True)
    return rows


def per_level(rows: list[dict]) -> dict:
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["level"]][0] += r["solved"]
        by[r["level"]][1] += 1
    return {lvl: {"solved": s, "n": n, "rate": round(s / n, 3)} for lvl, (s, n) in sorted(by.items(), reverse=True)}


def per_mode(rows: list[dict]) -> dict:
    """Pass rate of tasks that include each failure mode (a mode that drags is low)."""
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        for m in r["modes"]:
            by[m][0] += r["solved"]
            by[m][1] += 1
    return {m: {"solved": s, "n": n, "rate": round(s / n, 3)} for m, (s, n) in sorted(by.items())}


def benchmark(qwen: Qwen, tag: str, db, dbs, stores, out: Path, n: int) -> dict:
    events.emit("stage_start", stage=f"bench_{tag}", msg=f"held-out benchmark, {n} tasks per level, greedy")
    tasks = [t for lvl in bizops.LEVELS for t in bizops.make_tasks(lvl, n, "bench", 1000, stores)]
    rows = attempt(qwen, tasks, db, dbs, samples=1, temperature=0.0, stage=f"bench_{tag}", log=out / f"bench_{tag}.jsonl")
    res = {"model": qwen.describe(), "tasks": len(rows), "solved": sum(r["solved"] for r in rows),
           "rate": round(sum(r["solved"] for r in rows) / len(rows), 3),
           "mean_order_share": round(sum(r["order_share"] for r in rows) / len(rows), 3),
           "by_level": per_level(rows), "by_mode": per_mode(rows)}
    write_json(out / f"bench_{tag}.json", res)
    events.emit("metric", stage=f"bench_{tag}", msg=f"{tag}: {res['solved']}/{res['tasks']} solved",
                name=f"bench_{tag}", value=res["rate"], by_level=res["by_level"])
    events.emit("stage_end", stage=f"bench_{tag}", msg=f"{res['rate']:.0%} solved")
    return res


def anneal(qwen: Qwen, db, dbs, stores, out: Path, n: int) -> dict:
    events.emit("stage_start", stage="anneal", msg="start impossible (all 8 rules), lower until the band")
    curve, chosen = [], None
    ladder = {}
    if (out.parent / "design.json").exists():  # the swarm's ladder: which rule comes off at each step
        ladder = {s["level"]: tuple(s["rules"]) for s in json.loads((out.parent / "design.json").read_text())["ladder"]}
    for lvl in bizops.LEVELS:
        if lvl in ladder:
            tasks = [bizops.Task(ladder[lvl], {"store_id": stores[i % len(stores)]}) for i in range(n)]
        else:
            tasks = bizops.make_tasks(lvl, n, "train", 2000, stores)
        rows = attempt(qwen, tasks, db, dbs, samples=1, temperature=0.7, stage="anneal", log=out / "anneal.jsonl")
        rate = sum(r["solved"] for r in rows) / len(rows)
        curve.append({"level": lvl, "solved": sum(r["solved"] for r in rows), "n": len(rows), "rate": round(rate, 3)})
        events.emit("metric", stage="anneal", msg=f"level {lvl}: {rate:.0%}", name="anneal", level=lvl, value=round(rate, 3))
        if rate >= CONFIG.band_low:
            chosen = lvl
            break
    chosen = chosen or 1
    res = {"model": qwen.describe(), "target": CONFIG.target_rate, "band": [CONFIG.band_low, CONFIG.band_high],
           "curve": curve, "level": chosen,
           "ladder": {str(k): list(v) for k, v in ladder.items()},
           "note": "first level, going down the swarm's ladder from 8, whose sampled pass rate reached the band's floor"}
    write_json(out / "anneal.json", res)
    events.emit("stage_end", stage="anneal", msg=f"landed at level {chosen}", level=chosen)
    return res


def collect(qwen: Qwen, levels: list[int], db, dbs, stores, out: Path, rnd: int, n: int, k: int) -> list[dict]:
    events.emit("stage_start", stage="collect", msg=f"round {rnd}: rejection sampling at levels {levels}, k={k}")
    tasks = [t for lvl in levels for t in bizops.make_tasks(lvl, n, "train", 3000 + rnd, stores)]
    rows = attempt(qwen, tasks, db, dbs, samples=k, temperature=0.8, stage="collect", log=out / f"collect_r{rnd}.jsonl")
    kept, per_task = [], defaultdict(set)
    for r in rows:
        code = bizops.extract_code(r["answer"]).strip()
        if r["solved"] and code not in per_task[r["id"]] and len(per_task[r["id"]]) < 2:
            per_task[r["id"]].add(code)
            task = bizops.Task(tuple(r["rules"]), r["params"])
            kept.append({"messages": [{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": bizops.prompt(task, db)},
                                      {"role": "assistant", "content": f"```python\n{code}\n```"}],
                         "task": r["id"], "level": r["level"]})
    write_json(out / f"collect_r{rnd}.json", {"model": qwen.describe(), "levels": levels, "sampled": len(rows),
                                             "solved": sum(r["solved"] for r in rows), "kept": len(kept),
                                             "tasks_with_a_solve": len(per_task), "by_level": per_level(rows)})
    events.emit("stage_end", stage="collect", msg=f"{len(kept)} graded-correct answers kept of {len(rows)}", kept=len(kept))
    return kept


def train(qwen: Qwen, rows: list[dict], adapter_dir: Path) -> dict:
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_vlm.trainer.api import Trainer
    from mlx_vlm.trainer.common.model import prepare_model_for_training
    from mlx_vlm.trainer.vlm.config import VLMTrainingArgs
    events.emit("stage_start", stage="train", msg=f"LoRA r{LORA['rank']} on {len(rows)} rows x {LORA['epochs']} epochs")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    mx.random.seed(LORA["seed"])
    rng = random.Random(LORA["seed"])
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
    hist = list(trainer.log_history)
    for h in hist:
        if "train_loss" in h:
            events.emit("metric", stage="train", msg=f"loss {h['train_loss']:.3f}", name="train_loss", value=h["train_loss"])
    res = {"base": qwen.describe(), "settings": LORA, "rows": len(rows), "steps": len(data),
           "seconds": round(time.time() - started, 1),
           "first_loss": next((h["train_loss"] for h in hist if "train_loss" in h), None),
           "last_loss": next((h["train_loss"] for h in reversed(hist) if "train_loss" in h), None),
           "peak_memory_gb": round(max((h.get("peak_memory", 0) for h in hist), default=0), 2),
           "trainer": "mlx-vlm SFT (mlx_vlm.trainer), loss on the answer only"}
    write_json(adapter_dir / "run.json", res)
    events.emit("stage_end", stage="train", msg=f"loss {res['first_loss']} -> {res['last_loss']}")
    return res


def run(run_dir: Path, *, bench_n: int, anneal_n: int, collect_n: int, k: int, rounds: int, skip_before: bool) -> dict:
    import mlx.core as mx
    out = run_dir / "qwen"
    out.mkdir(parents=True, exist_ok=True)
    events.open_log(run_dir)
    db = bizops.load_db(run_dir / "business")
    dbs = [db] + bizops.twin_dbs()
    stores = [s["store_id"] for s in db["stores"]]
    report: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dataset": str(run_dir / "business"),
                    "settings": {"bench_n": bench_n, "anneal_n": anneal_n, "collect_n": collect_n, "k": k, "rounds": rounds}}

    qwen = Qwen()
    report["before"] = (json.loads((out / "bench_before.json").read_text()) if skip_before and (out / "bench_before.json").exists()
                        else benchmark(qwen, "before", db, dbs, stores, out, bench_n))
    report["anneal"] = anneal(qwen, db, dbs, stores, out, anneal_n)
    lvl = report["anneal"]["level"]
    trainable = [l for l in bizops.LEVELS if bizops.split_combos()[l]["train"]]
    levels = sorted({l for l in (lvl - 1, lvl, lvl + 1, lvl + 2) if l in trainable}, reverse=True)

    adapter, kept = None, []
    for rnd in range(1, rounds + 1):
        kept += collect(qwen, levels, db, dbs, stores, out, rnd, collect_n, k)
        if len(kept) < 8:
            print(f"only {len(kept)} solved answers; widening to easier levels", flush=True)
            easier = [l for l in trainable if l < min(levels)][:2]
            kept += collect(qwen, easier, db, dbs, stores, out, rnd * 10, collect_n, k)
        if not kept:
            print("no graded-correct answers at any level; nothing to train on, stopping before training", flush=True)
            events.emit("fail", stage="train", agent="trainer", role="trainer", msg="no correct samples to train on")
            break
        adapter = out / f"adapter_r{rnd}"
        del qwen
        mx.clear_cache()
        qwen = Qwen()
        report[f"train_r{rnd}"] = train(qwen, kept, adapter)
        del qwen
        mx.clear_cache()
        qwen = Qwen(adapter=adapter)

    if adapter is None:
        write_json(out / "report.json", report)
        return report
    report["after"] = benchmark(qwen, "after", db, dbs, stores, out, bench_n)
    b, a = report["before"], report["after"]
    report["delta"] = {"rate": round(a["rate"] - b["rate"], 3),
                       "by_level": {l: round(a["by_level"][l]["rate"] - b["by_level"][l]["rate"], 3)
                                    for l in a["by_level"] if l in b["by_level"]}}
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(out / "report.json", report)
    (out / "REPORT.md").write_text(markdown(report))
    events.emit("stage_end", stage="qwen", msg=f"before {b['rate']:.0%} -> after {a['rate']:.0%}")
    print((out / "REPORT.md").read_text())
    return report


def markdown(r: dict) -> str:
    b, a = r["before"], r["after"]
    m = b["model"]
    lines = [
        "# Qwen post-training on business-data tasks", "",
        f"Model: {m['id']}, weights `{m['weights']}` (quantization {m['quantization']}); "
        f"after = same weights + LoRA `{a['model']['adapter']}`.",
        f"Dataset: `{r['dataset']}` (synthetic shoe-store export, pruned by rsi intake).", "",
        f"Held-out benchmark: {b['tasks']} tasks, rule combinations never trained on, greedy decoding.", "",
        f"| | solved | rate | mean share of orders right |", "|---|---|---|---|",
        f"| before | {b['solved']}/{b['tasks']} | {b['rate']:.0%} | {b['mean_order_share']:.0%} |",
        f"| after | {a['solved']}/{a['tasks']} | {a['rate']:.0%} | {a['mean_order_share']:.0%} |", "",
        "| level (rules stacked) | before | after |", "|---|---|---|",
    ]
    for lvl in b["by_level"]:
        lb, la = b["by_level"][lvl], a["by_level"].get(lvl, {})
        lines.append(f"| {lvl} | {lb['solved']}/{lb['n']} | {la.get('solved', '-')}/{la.get('n', '-')} |")
    an = r["anneal"]
    lines += ["", "Anneal (start impossible, lower one rule at a time, sampled at T=0.7): "
              + ", ".join(f"L{c['level']} {c['solved']}/{c['n']}" for c in an["curve"]) + f" -> trained around level {an['level']}."]
    for key in sorted(k for k in r if k.startswith("train_r")):
        t = r[key]
        lines.append(f"{key}: {t['rows']} graded-correct answers, {t['steps']} steps, loss {t['first_loss']} -> {t['last_loss']}, "
                     f"{t['seconds']} s, peak {t['peak_memory_gb']} GB.")
    lines += ["", "Pass rate by failure mode (tasks containing it), before -> after:"]
    for mode in b["by_mode"]:
        lines.append(f"- {mode}: {b['by_mode'][mode]['rate']:.0%} -> {a['by_mode'].get(mode, {}).get('rate', 0):.0%}")
    return "\n".join(lines) + "\n"


def smoke(run_dir: Path) -> None:
    """Every step once on a few rows: batch generation, grading, two LoRA steps, reload with the adapter."""
    import mlx.core as mx
    out = run_dir / "qwen_smoke"
    out.mkdir(parents=True, exist_ok=True)
    db = bizops.load_db(run_dir / "business")
    stores = [s["store_id"] for s in db["stores"]]
    qwen = Qwen()
    tasks = bizops.make_tasks(1, 2, "train", 1, stores) + bizops.make_tasks(8, 1, "bench", 1, stores)
    t0 = time.time()
    rows = attempt(qwen, tasks, db, [db], samples=2, temperature=0.7, stage="smoke", log=out / "smoke.jsonl")
    print(f"smoke: {len(rows)} answers in {time.time() - t0:.0f} s; solved {[r['solved'] for r in rows]}", flush=True)
    t = tasks[0]
    fake = [{"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": bizops.prompt(t, db)},
                          {"role": "assistant", "content": "```python\ndef solve(db, params):\n    return {}\n```"}]}] * 2
    saved = LORA["epochs"]
    LORA["epochs"] = 1
    print(train(qwen, fake, out / "adapter"), flush=True)
    LORA["epochs"] = saved
    del qwen
    mx.clear_cache()
    qwen = Qwen(adapter=out / "adapter")
    print("with adapter:", qwen.generate([bizops.prompt(t, db)], 0.0, max_tokens=60)[0][:200], flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder under runs/ that holds business/ (rsi intake)")
    ap.add_argument("--bench-n", type=int, default=6, help="held-out tasks per level")
    ap.add_argument("--anneal-n", type=int, default=8)
    ap.add_argument("--collect-n", type=int, default=10, help="training tasks per level per round")
    ap.add_argument("--k", type=int, default=4, help="samples per training task")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--skip-before", action="store_true", help="reuse bench_before.json if present")
    ap.add_argument("--smoke", action="store_true", help="check the route on a few rows and exit")
    a = ap.parse_args()
    if a.smoke:
        return smoke(CONFIG.runs_dir / a.run)
    run(CONFIG.runs_dir / a.run, bench_n=a.bench_n, anneal_n=a.anneal_n, collect_n=a.collect_n, k=a.k,
        rounds=a.rounds, skip_before=a.skip_before)


if __name__ == "__main__":
    main()
