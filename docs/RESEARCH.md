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

## 2. Reward integrity: a verifier the candidate cannot reach

**The work.** TRACE (arXiv:2601.20103, ICML 2026) catalogues 54 kinds of reward exploit in
code environments and shows LLM judges miss many of them. Countdown-Code (a testbed for
reward hacking under RL with verifiable rewards) reports small Qwen models learning to edit
the verifier within a few hundred RL steps once the exploit is reachable. The lesson both
draw: an environment used for training has to make the exploit unreachable, because a
judge reading trajectories afterwards will not catch it reliably.

**What was wrong here.** The first exported family verifier imported the candidate's
`solve()` into the pytest process, with `expected.json` on the disk beside it. Two cheats of
a dozen lines each scored reward 1 on a full eight-rule task (checked on the old code before
it was changed): open the answer key and return it, or walk the call stack and return the
test module's `SPEC`.

**What this repo does now.** `rsi/family.py` exports a three-phase `tests/test.sh`:
0. reward starts at 0, and the agent's change to `/app` is written to
   `/logs/verifier/agent.patch` (diffed against the repo's first commit, so an agent that
   commits its work is still recorded)
1. the candidate runs once per data slice in its own process. The answer key is off the
   disk while it runs. As root (the Harbor default) it runs as `nobody` and can neither read
   nor write `/tests` or the logs. It gets the tables on stdin and returns JSON
2. pytest compares that JSON with the answer key. No candidate code is imported

A hash of the test files is taken before phase 1 and checked after it.

**Left out.** An agent with root in the container can replace `python3` or `pytest` before
the verifier starts. Nothing inside `test.sh` stops that; the replay in section 3 does.
Without root the privilege drop is off: the key removal and the hash check still hold, a
process the candidate leaves behind does not get stopped.

**Status.** Built, and run outside Docker: `tests/test_verifier.py` runs the real `test.sh`
on a git checkout of the starting repo (as root, with the privilege drop). The reference
scores 1; every mutation, the untouched repo and five cheats score 0. Not yet run under
Harbor.
