"""Settings, all overridable from the environment."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Config:
    # where every stage writes its output; one folder per pipeline run
    runs_dir: Path = Path(_env("RSI_RUNS_DIR", "runs"))

    # model that writes designs and tasks, and model whose pass rate we calibrate against
    author_model: str = _env("RSI_AUTHOR_MODEL", "claude-opus-5-5")
    judge_model: str = _env("RSI_JUDGE_MODEL", "claude-sonnet-5-5")
    solver_models: tuple[str, ...] = tuple(
        _env("RSI_SOLVER_MODELS", "anthropic/claude-opus-5-5").split(",")
    )
    solver_agent: str = _env("RSI_SOLVER_AGENT", "claude-code")

    # difficulty band: pass rate per solver model, all-or-nothing reward per attempt
    target_rate: float = float(_env("RSI_TARGET_RATE", "0.25"))
    band_low: float = float(_env("RSI_BAND_LOW", "0.15"))
    band_high: float = float(_env("RSI_BAND_HIGH", "0.35"))
    attempts: int = int(_env("RSI_ATTEMPTS", "8"))
    max_calibration_rounds: int = int(_env("RSI_MAX_ROUNDS", "4"))

    # harbor command; default assumes `uv tool install harbor`
    harbor_cmd: tuple[str, ...] = tuple(shlex.split(_env("RSI_HARBOR", "harbor")))
    harbor_env: str = _env("RSI_HARBOR_ENV", "docker")
    n_concurrent: int = int(_env("RSI_N_CONCURRENT", "4"))

    # sources
    x_bearer_token: str | None = os.environ.get("X_BEARER_TOKEN")
    s2_api_key: str | None = os.environ.get("S2_API_KEY")

    # combine at least this many independent failure modes into one task
    min_modes: int = int(_env("RSI_MIN_MODES", "3"))

    extra: dict = field(default_factory=dict)


CONFIG = Config()
