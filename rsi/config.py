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
    # model that must solve a new family's tickets cold; kept different from the author
    check_model: str = _env("RSI_CHECK_MODEL", "claude-opus-5-5")
    solver_models: tuple[str, ...] = tuple(
        _env("RSI_SOLVER_MODELS", "anthropic/claude-opus-5-5").split(",")
    )
    solver_agent: str = _env("RSI_SOLVER_AGENT", "claude-code")

    # red team: models told to get reward without building the feature (default: the solvers)
    redteam_models: tuple[str, ...] = tuple(
        _env("RSI_REDTEAM_MODELS", _env("RSI_SOLVER_MODELS", "anthropic/claude-opus-5-5")).split(",")
    )
    # red-team a task once it is in band, before it joins the pool (three attempts per red-team model)
    redteam: bool = _env("RSI_REDTEAM", "1") != "0"
    # replay every passing attempt's recorded patch in a clean container; only a pass that
    # survives counts (set RSI_REPLAY=0 to trust the in-container reward)
    replay: bool = _env("RSI_REPLAY", "1") != "0"

    # difficulty band: pass rate per solver model, all-or-nothing reward per attempt
    target_rate: float = float(_env("RSI_TARGET_RATE", "0.25"))
    band_low: float = float(_env("RSI_BAND_LOW", "0.15"))
    band_high: float = float(_env("RSI_BAND_HIGH", "0.35"))
    attempts: int = int(_env("RSI_ATTEMPTS", "8"))
    max_calibration_rounds: int = int(_env("RSI_MAX_ROUNDS", "4"))
    # independent fairness reviews per distinct set of failing tests; one "unfair" is enough
    inspect_reviews: int = int(_env("RSI_INSPECT_REVIEWS", "3"))
    # hinted attempts: the solver is told every criterion and where builds usually slip. Run
    # when no cold attempt solved the task ("auto"), on every measurement ("always"), or never ("off")
    hinted: str = _env("RSI_HINTED", "auto")
    hinted_attempts: int = int(_env("RSI_HINTED_ATTEMPTS", "4"))

    # harbor command; default assumes `uv tool install harbor`
    harbor_cmd: tuple[str, ...] = tuple(shlex.split(_env("RSI_HARBOR", "harbor")))
    harbor_env: str = _env("RSI_HARBOR_ENV", "docker")
    n_concurrent: int = int(_env("RSI_N_CONCURRENT", "4"))

    # sources
    x_bearer_token: str | None = os.environ.get("X_BEARER_TOKEN")
    s2_api_key: str | None = os.environ.get("S2_API_KEY")

    # combine at least this many independent failure modes into one task
    min_modes: int = int(_env("RSI_MIN_MODES", "3"))
    max_modes: int = int(_env("RSI_MAX_MODES", "4"))
    # weight of history measured with a solver that is not in solver_models (a newer model
    # handles more, so old slips are weak evidence about it)
    stale_weight: float = float(_env("RSI_STALE_WEIGHT", "0.25"))

    # after calibration, read each failed attempt's patch for the slip it made (rsi/mine.py)
    mine: bool = _env("RSI_MINE", "1") != "0"

    # a design this similar (0..1, word n-gram cosine) to a task already in the pool is a copy
    novelty_max: float = float(_env("RSI_NOVELTY_MAX", "0.6"))

    extra: dict = field(default_factory=dict)


CONFIG = Config()
