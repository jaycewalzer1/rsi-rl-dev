"""Thin wrapper over the Harbor CLI: run a job on one task, read back trial rewards."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .config import CONFIG


@dataclass
class Trial:
    name: str
    reward: float | None  # None = the trial errored before grading
    exception: str | None
    dir: Path

    @property
    def solved(self) -> bool:
        return self.reward is not None and self.reward >= 1.0


def run_job(
    task_dir: Path,
    agent: str,
    *,
    model: str | None = None,
    attempts: int = 1,
    jobs_dir: Path,
    job_name: str | None = None,
    extra: list[str] | None = None,
) -> list[Trial]:
    job_name = job_name or f"{agent}-{int(time.time())}"
    cmd = [
        *CONFIG.harbor_cmd, "run",
        "-p", str(task_dir),
        "-a", agent,
        "-k", str(attempts),
        "-n", str(min(CONFIG.n_concurrent, attempts)),
        "-e", CONFIG.harbor_env,
        "-o", str(jobs_dir),
        "--job-name", job_name,
        "-q", "-y",
    ]
    if model:
        cmd += ["-m", model]
    cmd += extra or []
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (jobs_dir / f"{job_name}.log").parent.mkdir(parents=True, exist_ok=True)
    (jobs_dir / f"{job_name}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    trials = read_trials(jobs_dir / job_name)
    if not trials:
        raise RuntimeError(
            f"harbor produced no trials (exit {proc.returncode}); see {jobs_dir / job_name}.log"
        )
    return trials


def read_trials(job_dir: Path) -> list[Trial]:
    trials = []
    for result in sorted(job_dir.glob("*/result.json")):
        data = json.loads(result.read_text())
        rewards = (data.get("verifier_result") or {}).get("rewards") or {}
        reward = rewards.get("reward", next(iter(rewards.values()), None)) if rewards else None
        exc = data.get("exception_info")
        trials.append(
            Trial(
                name=data.get("trial_name", result.parent.name),
                reward=reward,
                exception=(exc or {}).get("exception_type"),
                dir=result.parent,
            )
        )
    return trials


def failed_tests(trial: Trial) -> list[str]:
    """Names of failing hidden tests, from the CTRF report the verifier writes."""
    ctrf = trial.dir / "verifier" / "ctrf.json"
    if not ctrf.exists():
        return []
    tests = json.loads(ctrf.read_text()).get("results", {}).get("tests", [])
    return [t["name"] for t in tests if t.get("status") == "failed"]
