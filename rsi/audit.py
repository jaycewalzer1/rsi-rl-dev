"""Stage 5b: a static audit of a task directory. No model, no Docker, a second at most.

Every check is a way a task has leaked its answer or let a broken verifier hand out reward,
taken from published audits of agent benchmarks and from the Terminal-Bench task guideline
(docs/RESEARCH.md, "Reward integrity"). A finding is `block` (do not measure the task until
it is fixed) or `warn` (read it; it is often a defect).

    python -m rsi.audit TASK_DIR

A check is a function `(Task) -> list[finding]` in CHECKS. Add checks freely; the fixtures in
tests/test_audit.py keep the existing ones firing.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

CODE = {".py", ".js", ".ts", ".tsx", ".mjs", ".sh", ".patch", ".diff", ".go", ".rs", ".java", ".rb", ".sql"}
ANSWER_NAME = re.compile(r"^(expected|answers?|golden|snapshot|reference|solution)", re.I)
RUNNER = re.compile(r"\b(pytest|python3?|node|npm|npx|jest|vitest|mocha|go test|cargo test|bash)\b")
DOWNLOAD = re.compile(r"\b(curl|wget|uvx|pip3? install|uv pip|uv tool|apt(-get)? (update|install)|npm (i|ci|install)|yarn (add|install)|git clone)\b")
TICKET_LEAK = re.compile(r"(/tests\b|\bauthoring/|criteria\.json|reference\.patch|solve\.sh|hidden tests?)", re.I)
HIDDEN_SRC = re.compile(r"^(\./)?(tests|solution|authoring)(/|$)|\.\./")


@dataclass
class Task:
    dir: Path

    def text(self, rel: str) -> str:
        p = self.dir / rel
        return p.read_text(errors="replace") if p.is_file() else ""

    def files(self, sub: str) -> dict[Path, str]:
        """Text files under a subfolder, keyed by path relative to the task."""
        out = {}
        for p in sorted((self.dir / sub).rglob("*")):
            rel = p.relative_to(self.dir)
            if p.is_file() and rel.parts[:2] != ("authoring", "gate") and p.stat().st_size < 2_000_000:
                try:
                    out[rel] = p.read_text()
                except UnicodeDecodeError:
                    pass
        return out

    @cached_property
    def environment(self) -> dict[Path, str]:
        return self.files("environment")

    @cached_property
    def hidden(self) -> dict[Path, str]:
        return {**self.files("tests"), **self.files("solution"), **self.files("authoring")}

    @cached_property
    def criteria(self) -> list[dict]:
        try:
            return json.loads(self.text("authoring/criteria.json") or "[]")
        except json.JSONDecodeError:
            return []


def _f(check: str, severity: str, detail: str) -> dict:
    return {"check": check, "severity": severity, "detail": detail}


def _shell_lines(script: str) -> list[str]:
    """Logical lines of a shell script or Dockerfile: continuations joined, comments dropped."""
    joined = re.sub(r"\\\n", " ", script)
    return [line.strip() for line in joined.splitlines() if line.strip() and not line.strip().startswith("#")]


def hidden_copies(t: Task) -> list[dict]:
    """A hidden file (test, patch, answer key) that also sits in the image the agent works in."""
    out = []
    env = {text: rel for rel, text in t.environment.items()}
    for rel, text in t.hidden.items():
        answerish = rel.suffix in CODE or ANSWER_NAME.match(rel.name)
        if answerish and len(text) >= 64 and text in env:
            out.append(_f("hidden-copy", "block", f"{env[text]} is a copy of the hidden file {rel}"))
    return out


def reference_leak(t: Task) -> list[dict]:
    """The starting repo already holds most of what the reference solution adds."""
    added = {line[1:].strip() for line in t.text("solution/reference.patch").splitlines()
             if line.startswith("+") and not line.startswith("+++")}
    added = {a for a in added if len(a) >= 30}
    if len(added) < 6:
        return []
    env = "\n".join(t.environment.values())
    present = sorted(a for a in added if a in env)
    if len(present) / len(added) >= 0.6:
        return [_f("reference-leak", "block", f"{len(present)} of {len(added)} lines the reference adds are already in "
                                             f"environment/, e.g. {present[0][:80]!r}")]
    return []


def hidden_test_names(t: Task) -> set[str]:
    names = {re.sub(r"\[.*$", "", n) for c in t.criteria for n in c.get("tests", [])}
    for rel, text in t.hidden.items():
        if rel.parts[0] == "tests":
            names |= set(re.findall(r"def (test_\w+)", text))
    return {n for n in names if len(n) >= 12}


def name_leak(t: Task) -> list[dict]:
    """Hidden test names the agent can read: in the ticket (block) or in the image (warn)."""
    out = []
    ticket = t.text("instruction.md")
    env = "\n".join(t.environment.values())
    for n in sorted(hidden_test_names(t)):
        if n in ticket:
            out.append(_f("test-name-leak", "block", f"instruction.md names the hidden test {n}"))
        elif n in env:
            out.append(_f("test-name-leak", "warn", f"environment/ contains the hidden test name {n}"))
    return out


def ticket_leak(t: Task) -> list[dict]:
    hits = sorted({m.group(0) for m in TICKET_LEAK.finditer(t.text("instruction.md"))})
    return [_f("ticket-leak", "block", f"instruction.md points at the grader: {hits}")] if hits else []


def dockerfile(t: Task) -> list[dict]:
    out = []
    text = t.text("environment/Dockerfile")
    for line in _shell_lines(text):
        words = line.split()
        if words and words[0].upper() in ("COPY", "ADD"):
            srcs = [w for w in words[1:-1] if not w.startswith("--")]
            bad = [s for s in srcs if HIDDEN_SRC.search(s)]
            if bad:
                out.append(_f("dockerfile-copies-hidden", "block", f"Dockerfile copies {bad} into the image"))
        if re.search(r"reference\.patch|solve\.sh|/solution\b", line):
            out.append(_f("dockerfile-copies-hidden", "block", f"Dockerfile touches the solution: {line[:100]}"))
    if text and not ("git init" in text and "commit" in text):
        out.append(_f("dockerfile-base-commit", "warn", "Dockerfile does not commit the starting repo (contract: /app is a git repo with one commit)"))
    return out


def verifier_script(t: Task) -> list[dict]:
    out = []
    script = t.text("tests/test.sh")
    if not script:
        return out
    lines = _shell_lines(script)
    if not re.search(r"reward\.(txt|json)", script):
        out.append(_f("verifier-no-reward", "block", "tests/test.sh never writes /logs/verifier/reward.txt"))
    for line in lines:
        if m := DOWNLOAD.search(line):
            out.append(_f("verifier-downloads", "block", f"tests/test.sh downloads at test time ({m.group(0)}): the reward then depends on the network"))
        piped = re.search(r"(?<!\|)\|(?!\|)", line)
        if RUNNER.search(line) and piped and "pipefail" not in script:
            out.append(_f("verifier-masked-status", "block", f"a pipe hides the test runner's exit status (no `set -o pipefail`): {line[:100]}"))
        if RUNNER.search(line) and re.search(r"\|\|\s*(true|:)(\s|;|$)", line):
            out.append(_f("verifier-masked-status", "block", f"`|| true` hides the test runner's exit status: {line[:100]}"))
    writes_one = re.search(r"echo\s+['\"]?1(\.0)?['\"]?\s*>+\s*\S*reward", script)
    if writes_one and not re.search(r"\bif\b|&&", script):
        out.append(_f("verifier-unconditional", "block", "tests/test.sh writes reward 1 without a condition"))
    return out


def solve_sh(t: Task) -> list[dict]:
    extra = [line for line in _shell_lines(t.text("solution/solve.sh"))
             if not re.fullmatch(r"set -[a-z]+( -o pipefail)?", line) and "reference.patch" not in line]
    return [_f("solve-sh-extra", "warn", f"solution/solve.sh does more than apply the reference: {extra[:3]}")] if extra else []


def weak_tests(t: Task) -> list[dict]:
    out = []
    for rel, text in t.hidden.items():
        if rel.parts[0] != "tests" or rel.suffix not in CODE or rel.suffix in (".sh", ".patch", ".diff"):
            continue
        if re.search(r"def test_|\bit\(|\btest\(", text) and not re.search(r"\bassert\b|raises|expect\(", text):
            out.append(_f("test-asserts-nothing", "block", f"{rel} has tests and no assertion"))
        if re.search(r"\btime\.sleep\(|\bsleep\(\d|setTimeout\(", text):
            out.append(_f("test-sleeps", "warn", f"{rel} sleeps (contract: no sleeps, no wall-clock races)"))
        if re.search(r"(#|//).*\bAC\d+\b", text):
            out.append(_f("test-names-criterion", "warn", f"{rel} has a comment naming a criterion"))
    return out


def criteria_map(t: Task) -> list[dict]:
    """criteria.json must name tests that exist, or slips get booked to the wrong failure mode."""
    tests = "\n".join(text for rel, text in t.hidden.items() if rel.parts[0] == "tests")
    missing = sorted({re.sub(r"\[.*$", "", n) for c in t.criteria for n in c.get("tests", [])} - set(re.findall(r"\w+", tests)))
    return [_f("criteria-unknown-test", "warn", f"criteria.json names tests that tests/ does not define: {missing[:5]}")] if missing else []


CHECKS = [hidden_copies, reference_leak, name_leak, ticket_leak, dockerfile, verifier_script, solve_sh, weak_tests, criteria_map]


def audit(task_dir: Path) -> list[dict]:
    t = Task(Path(task_dir))
    return [f for check in CHECKS for f in check(t)]


def main() -> int:
    findings = audit(Path(sys.argv[1]))
    for f in findings:
        print(f"{f['severity'].upper():5} {f['check']:<26} {f['detail']}")
    blocks = sum(f["severity"] == "block" for f in findings)
    print(f"AUDIT: {'CLEAN' if not blocks else f'{blocks} BLOCKING'} ({len(findings) - blocks} warnings)")
    return 1 if blocks else 0


if __name__ == "__main__":
    sys.exit(main())
