"""rsi: complaints + papers -> combined hard tasks -> Harbor-calibrated at ~25%.

  rsi loop     [--run NAME] [--tasks N]   whole pipeline, N tasks
  rsi scan     --run NAME                 stage 1, X complaints
  rsi papers   --run NAME                 stage 2, cross-reference with papers
  rsi combine  --run NAME                 stage 3, stack failure modes into a design
  rsi build    --run NAME                 stage 4, write the Harbor task
  rsi gate     TASK_DIR [--stages a,b]    stage 5, the testing suite
  rsi summary  TASK_DIR                   stage results, STALE if the task changed
  rsi calibrate TASK_DIR                  stage 6, hold the pass rate in the band
  rsi mine     TASK_DIR | --promote       read a task's failed attempts for the slips they made;
                                          --promote groups slips of a new kind into failure modes
  rsi pool                                the calibrated tasks: active and retired
  rsi recheck  [--task NAME] [--attempts N]
                                          measure the active tasks again with today's solvers;
                                          retire the ones that have become too easy

  rsi factory  --run NAME --business DIR [--modes-from RUN] [--families 3] [--only STAGES]
                                          business data in, proven task families and Harbor tasks out:
                                          intake, scan + papers, author agents, export, gate
  rsi qwen     --run NAME [--rounds N]    queue the Qwen leg on kokopelli's GPU queue:
                                          calibrate to the 25% band, benchmark, post-train, benchmark
  rsi demo     --run NAME [--port 8791]   the demo page, live or replay
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import archive, build, calibrate, combine, gate, mine, papers, scan_x
from .config import CONFIG


def run_dir(name: str | None) -> Path:
    d = CONFIG.runs_dir / (name or time.strftime("%Y%m%d-%H%M%S"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rsi", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["loop", "scan", "papers", "combine", "build"]:
        p = sub.add_parser(name)
        p.add_argument("--run", help="run folder name under runs/ (default: timestamp)")
        p.add_argument("--seed", type=int)
        p.add_argument("--tasks", type=int, default=1)
        p.add_argument("--skip-scan", action="store_true", help="use seed failure modes only")
    p = sub.add_parser("factory")
    p.add_argument("--run", required=True)
    p.add_argument("--business", type=Path, help="folder of CSV files, one table each (default: the synthetic shoe store)")
    p.add_argument("--modes-from", help="reuse the scan and paper check of an earlier run")
    p.add_argument("--families", type=int, default=3)
    p.add_argument("--rows", type=int, default=1500, help="root-table rows per data slice")
    p.add_argument("--gate", default="shape,audit,oracle,nop,mutations", help="gate stages for the exported tasks")
    p.add_argument("--only", help="comma list of stages to (re)run: intake,modes,author,export,gate")
    p = sub.add_parser("qwen")
    p.add_argument("--run", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--headline-target", type=float, default=0.50, help="before pass rate the headline set should sit at")
    p.add_argument("--harder-target", type=float, default=0.25, help="before pass rate the harder set should sit at")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--bench-n", type=int, default=20)
    p.add_argument("--collect-n", type=int, default=6)
    p.add_argument("--smoke", action="store_true", help="a few answers per family, to see where Qwen stands")
    p.add_argument("--dry", action="store_true", help="print the queue command only")
    p = sub.add_parser("demo")
    p.add_argument("--run", required=True)
    p.add_argument("--port", type=int, default=8791)
    p.add_argument("--export", type=Path, help="write a self-contained replay page instead of serving")
    for name in ["gate", "summary", "calibrate"]:
        p = sub.add_parser(name)
        p.add_argument("task_dir", type=Path)
        p.add_argument("--stages")
    p = sub.add_parser("mine")
    p.add_argument("task_dir", type=Path, nargs="?")
    p.add_argument("--promote", action="store_true")
    sub.add_parser("pool")
    p = sub.add_parser("recheck")
    p.add_argument("--task", action="append", help="only this task (repeatable); default: every active task")
    p.add_argument("--attempts", type=int)
    a = ap.parse_args(argv)

    print(f"models: author {CONFIG.author_model}, judge {CONFIG.judge_model}, "
          f"solvers {', '.join(CONFIG.solver_models)} via {CONFIG.solver_agent}")

    if a.cmd == "mine":
        if a.task_dir:
            mine.run(a.task_dir)
        if a.promote:
            mine.promote(papers.merge_seeds([]))
        return 0
    if a.cmd == "pool":
        print(archive.listing())
        return 0
    if a.cmd == "recheck":
        archive.recheck(a.task, a.attempts)
        print(archive.listing())
        return 0

    if a.cmd in ("gate", "summary", "calibrate"):
        if a.cmd == "gate":
            ok = gate.run(a.task_dir, gate.stage_list(a.stages))
            print(gate.summary(a.task_dir))
            return 0 if ok else 1
        if a.cmd == "summary":
            print(gate.summary(a.task_dir))
            return 0
        return 0 if calibrate.run(a.task_dir) else 1

    if a.cmd in ("factory", "qwen", "demo"):
        return _factory(a)

    out = run_dir(a.run)
    if a.cmd == "scan":
        scan_x.run(out)
    elif a.cmd == "papers":
        papers.run(out)
    elif a.cmd == "combine":
        combine.run(out, a.seed)
    elif a.cmd == "build":
        build.run(out / "design.json", out)
    elif a.cmd == "loop":
        if not a.skip_scan:
            scan_x.run(out)
        papers.run(out)
        results = {}
        for i in range(a.tasks):
            seed = None if a.seed is None else a.seed + i
            design = combine.run(out, seed)
            task = build.run(design, out)
            results[task.name] = calibrate.run(task, out / "failure_modes.json")
        for name, ok in results.items():
            print(f"{name}: {'IN BAND' if ok else 'not in band'}")
    return 0


def _factory(a) -> int:
    if a.cmd == "factory":
        from . import swarm
        swarm.run(a.run, a.business or swarm.DEFAULT_BUSINESS, modes_from=a.modes_from, families=a.families,
                  gate_stages=gate.stage_list(a.gate), rows=a.rows, only=a.only.split(",") if a.only else None)
    elif a.cmd == "qwen":
        import shlex
        import subprocess
        from .qwen import KOKO
        root = Path(__file__).resolve().parent.parent
        what = "a few answers per family (smoke)" if a.smoke else (
            "levels adjusted until the before pass rate is about 50% (headline) and 25% (harder) on held-out tasks, LoRA "
            "post-trained over several rounds on its own graded-correct answers, benchmarked after each round")
        cmd = ["python3", str(KOKO / "kokopelli" / "ops" / "gpu.py"), "run", "--name", f"rsi_qwen_{a.run}" + ("_smoke" if a.smoke else ""),
               "--priority", "50", "--gb", "18", "--cwd", str(root),
               "--about", f"rsi-rl-dev Qwen leg on run {a.run}: Qwen3-VL-8B-Instruct 4-bit on task families written for "
                          f"that run's business data; {what}",
               "--output", f"runs/{a.run}/qwen" + ("_smoke" if a.smoke else ""), "--",
               str(KOKO / ".venv" / "bin" / "python"), "-u", "-m", "rsi.qwen", "--run", a.run]
        cmd += ["--smoke"] if a.smoke else ["--rounds", str(a.rounds), "--k", str(a.k), "--bench-n", str(a.bench_n),
                                            "--collect-n", str(a.collect_n), "--headline-target", str(a.headline_target),
                                            "--harder-target", str(a.harder_target)]
        print(shlex.join(cmd))
        return 0 if a.dry else subprocess.call(cmd)
    elif a.cmd == "demo":
        from . import demo
        if a.export:
            print(f"demo: wrote {demo.export(CONFIG.runs_dir / a.run, a.export)}")
        else:
            demo.serve(CONFIG.runs_dir / a.run, a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
