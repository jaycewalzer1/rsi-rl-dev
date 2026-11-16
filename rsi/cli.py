"""rsi: complaints + papers -> combined hard tasks -> Harbor-calibrated at ~25%.

  rsi loop     [--run NAME] [--tasks N]   whole pipeline, N tasks
  rsi scan     --run NAME                 stage 1, X complaints
  rsi papers   --run NAME                 stage 2, cross-reference with papers
  rsi combine  --run NAME                 stage 3, stack failure modes into a design
  rsi build    --run NAME                 stage 4, write the Harbor task
  rsi gate     TASK_DIR [--stages a,b]    stage 5, the testing suite
  rsi summary  TASK_DIR                   stage results, STALE if the task changed
  rsi calibrate TASK_DIR                  stage 6, hold the pass rate in the band
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


if __name__ == "__main__":
    sys.exit(main())
