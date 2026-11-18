"""rsi: complaints + papers -> combined hard tasks -> Harbor-calibrated at ~25%.

  rsi loop     [--run NAME] [--tasks N]   whole pipeline, N tasks
  rsi scan     --run NAME                 stage 1, X complaints
  rsi papers   --run NAME                 stage 2, cross-reference with papers
  rsi combine  --run NAME                 stage 3, stack failure modes into a design
  rsi build    --run NAME                 stage 4, write the Harbor task
  rsi gate     TASK_DIR [--stages a,b]    stage 5, the testing suite
  rsi summary  TASK_DIR                   stage results, STALE if the task changed
  rsi calibrate TASK_DIR                  stage 6, hold the pass rate in the band

  rsi intake   --run NAME [--business DIR]  stage 0, prune a business export (CSV folder)
  rsi swarm    --run NAME [--business DIR] [--skip-scan] [--gate STAGES] [--levels 8,5,3]
                                          the whole factory as a swarm, logged for the demo page
  rsi qwen     --run NAME [...]           queue the Qwen leg on kokopelli's GPU queue:
                                          benchmark, anneal, post-train, benchmark again
  rsi demo     --run NAME [--port 8791]   the demo page, live or replay
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import build, calibrate, combine, gate, papers, scan_x
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
    p = sub.add_parser("intake")
    p.add_argument("--run", required=True)
    p.add_argument("--business", type=Path)
    p.add_argument("--no-stewards", action="store_true")
    p = sub.add_parser("swarm")
    p.add_argument("--run", required=True)
    p.add_argument("--business", type=Path)
    p.add_argument("--skip-scan", action="store_true")
    p.add_argument("--gate", help="gate stages to run on the exported tasks, e.g. shape,oracle,nop,mutations")
    p.add_argument("--levels", default="8,5,3", help="ladder levels to export as Harbor tasks")
    p = sub.add_parser("qwen")
    p.add_argument("--run", required=True)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--bench-n", type=int, default=6)
    p.add_argument("--collect-n", type=int, default=10)
    p.add_argument("--dry", action="store_true", help="print the queue command only")
    p = sub.add_parser("demo")
    p.add_argument("--run", required=True)
    p.add_argument("--port", type=int, default=8791)
    p.add_argument("--export", type=Path, help="write a self-contained replay page instead of serving")
    for name in ["gate", "summary", "calibrate"]:
        p = sub.add_parser(name)
        p.add_argument("task_dir", type=Path)
        p.add_argument("--stages")
    a = ap.parse_args(argv)

    print(f"models: author {CONFIG.author_model}, judge {CONFIG.judge_model}, "
          f"solvers {', '.join(CONFIG.solver_models)} via {CONFIG.solver_agent}")

    if a.cmd in ("gate", "summary", "calibrate"):
        if a.cmd == "gate":
            ok = gate.run(a.task_dir, gate.stage_list(a.stages))
            print(gate.summary(a.task_dir))
            return 0 if ok else 1
        if a.cmd == "summary":
            print(gate.summary(a.task_dir))
            return 0
        return 0 if calibrate.run(a.task_dir) else 1

    if a.cmd in ("intake", "swarm", "qwen", "demo"):
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
    from . import intake, swarm
    if a.cmd == "intake":
        intake.run(a.business or swarm.DEFAULT_BUSINESS, run_dir(a.run), stewards=not a.no_stewards)
    elif a.cmd == "swarm":
        swarm.run(a.run, a.business or swarm.DEFAULT_BUSINESS, a.skip_scan, gate.stage_list(a.gate),
                  [int(x) for x in a.levels.split(",")])
    elif a.cmd == "qwen":
        import shlex
        import subprocess
        from .qwen import KOKO
        root = Path(__file__).resolve().parent.parent
        cmd = ["python3", str(KOKO / "kokopelli" / "ops" / "gpu.py"), "run", "--name", f"rsi_qwen_{a.run}",
               "--priority", "50", "--gb", "18", "--cwd", str(root),
               "--about", f"rsi-rl-dev Qwen leg on run {a.run}: Qwen3-VL-8B-Instruct 4-bit benchmarked on held-out "
                          "business-data tasks, annealed from level 8 down to the 25% band, LoRA post-trained on "
                          "graded-correct samples, benchmarked again",
               "--output", f"runs/{a.run}/qwen", "--",
               str(KOKO / ".venv" / "bin" / "python"), "-u", "-m", "rsi.qwen", "--run", a.run,
               "--rounds", str(a.rounds), "--k", str(a.k), "--bench-n", str(a.bench_n), "--collect-n", str(a.collect_n)]
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
