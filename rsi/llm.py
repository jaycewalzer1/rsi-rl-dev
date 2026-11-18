"""One way to ask a model something.

Default backend is headless Claude Code (`claude -p`), which works with a Claude
subscription login. With ANTHROPIC_API_KEY set and RSI_LLM=api it calls the
Messages API directly instead. Every call is logged with the model it used.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path


LOG = Path(os.environ.get("RSI_LLM_LOG", "runs/llm_calls.jsonl"))


def _log(entry: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def ask(
    prompt: str,
    model: str,
    *,
    tools: list[str] | None = None,
    allowed: list[str] | None = None,
    cwd: Path | None = None,
    timeout: int = 1800,
    budget_usd: float | None = None,
) -> str:
    """Return the model's final text. `tools` enables Claude Code tools (CLI backend only).
    `allowed` narrows what those tools may do without asking, as Claude Code permission rules
    (e.g. "Bash(uv run pytest *)"); by default every listed tool is allowed outright.
    `budget_usd` caps what one session may spend."""
    backend = os.environ.get("RSI_LLM", "cli")
    started = time.time()
    if backend == "api":
        if tools:
            raise ValueError("tool use needs the cli backend (RSI_LLM=cli)")
        text = _ask_api(prompt, model, timeout)
    else:
        text = _ask_cli(prompt, model, tools, cwd, timeout, allowed, budget_usd)
    _log(
        {
            "backend": backend,
            "model": model,
            "tools": tools or [],
            "seconds": round(time.time() - started, 1),
            "prompt_head": prompt[:200],
            "chars_out": len(text),
        }
    )
    return text


def cli_command(model, tools=None, allowed=None, budget_usd=None) -> list[str]:
    cmd = ["claude", "-p", "--model", model, "--output-format", "json"]
    if tools:
        cmd += ["--tools", ",".join(tools)]
        cmd += ["--allowedTools", *allowed] if allowed else ["--allowedTools", ",".join(tools)]
        cmd += ["--permission-mode", "acceptEdits"]
    else:
        cmd += ["--tools", ""]
    if budget_usd:
        cmd += ["--max-budget-usd", str(budget_usd)]
    return cmd


def _ask_cli(prompt, model, tools, cwd, timeout, allowed=None, budget_usd=None) -> str:
    cmd = cli_command(model, tools, allowed, budget_usd)
    proc = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True, cwd=cwd, timeout=timeout
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p failed ({proc.returncode}): {proc.stderr[-2000:]}")
    return json.loads(proc.stdout).get("result", "")


def _ask_api(prompt, model, timeout) -> str:
    import httpx  # only the api backend needs it

    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": model,
            "max_tokens": 16000,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"])


def ask_json(prompt: str, model: str, **kw):
    """Ask for JSON and parse the first JSON value in the answer."""
    text = ask(prompt + "\n\nAnswer with JSON only, no prose.", model, **kw)
    return extract_json(text)


def extract_json(text: str):
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
    if start < 0:
        raise ValueError(f"no JSON in model answer: {text[:300]}")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    return value
