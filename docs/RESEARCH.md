# Research map

Where each part of the factory sits in the published work, what was taken from it, and what
was left out on purpose. One section per area; each landed as its own commit (Nov 18 2026).

Two words used below:
- **built**: in the code, with unit tests that run without Docker or a model.
- **live**: also run end to end through Harbor with real solvers.

Nothing in this file is live unless it says so. The README's Status section is the record
of what has been run for real.

## 1. Reward integrity: the static audit

**The work.** Audits of agent benchmarks keep finding the same defects. The Terminal-Bench
task guideline (arXiv:2604.28093) lists six kinds of bad task, two of them about the grader:
tests that validate the wrong things, and reward hacking through environment leakage. Auto
Benchmark Audit (arXiv:2605.26079) ran an agent over 168 benchmarks and reports ambiguous
design, environment conflicts or wrong ground truth in over 25.7% of tasks; dropping the
flagged tasks moved SWE-bench Verified and Terminal-Bench 2 scores by about ten points.

**What this repo does.** `rsi/audit.py` is a gate stage that needs no model and no Docker.
It blocks a task when:
- a hidden file (test, patch, answer key) is also in the image, or the Dockerfile copies
  `tests/`, `solution/` or `authoring/`
- the starting repo already contains most of the lines the reference adds
- the ticket names a hidden test or points at the grader
- `tests/test.sh` downloads at test time, hides the runner's exit status behind a pipe or
  `|| true`, writes reward 1 without a condition, or never writes a reward
- a hidden test file has tests and no assertion

The shape stage now also checks what the contract always said: every criterion's sentence
is in the ticket word for word.

**Left out.** ABA is an agent that replays trajectories and proposes fixes. This stage is
regexes over one directory. It catches the cheap, common defects before any solver attempt is
paid for; it cannot see a verifier that is wrong in a way only running it shows. The
red-team stage and the replay (section 3) cover part of that.

**Status.** Built. `tests/test_audit.py` holds one broken fixture per check.
