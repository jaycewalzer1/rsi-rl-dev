# Task contract

Every task the pipeline builds is a [Harbor](https://github.com/harbor-framework/harbor) task
directory. The gate (`rsi gate`) relies on every line below.

```
<slug>/
  task.toml                 Harbor schema 1.1; [verifier] and [agent] timeouts; allow_internet = true
                            (the agent needs the model API; tests must still never use the network)
  instruction.md            the ticket the agent sees: 150 to 250 words, rules not scenarios,
                            exact codes and messages in a "Pins" section in backticks
  environment/Dockerfile    builds the starting repo into /app as a git repo with one commit
                            (`git init && git add -A && git commit`), installs everything the
                            tests need at build time; nothing is downloaded at test time
  environment/app/...       the starting codebase (copied into /app by the Dockerfile)
  solution/reference.patch  `git diff` from the starting commit to the reference solution
  solution/solve.sh         applies it: `git -C /app apply --whitespace=nowarn /solution/reference.patch`
                            and nothing else that changes behaviour (the gate appends mutation
                            patches after this line)
  tests/test.sh             runs the hidden tests in /tests against /app, writes a CTRF report
                            to /logs/verifier/ctrf.json and 1 or 0 to /logs/verifier/reward.txt:
                            1 only if EVERY hidden test passes (all-or-nothing)
                            It starts with these lines, so a crash scores 0 and the gate can
                            replay the attempt and read what it built:
                              mkdir -p /logs/verifier; echo 0 > /logs/verifier/reward.txt
                              BASE="$(git -C /app rev-list --max-parents=0 HEAD | tail -1)"
                              git -C /app add -A -- . ':(exclude,glob)**/__pycache__/**'
                              git -C /app diff --cached --binary "$BASE" > /logs/verifier/agent.patch
  tests/...                 hidden tests; the agent never sees this folder
  authoring/design.json     the design from `rsi combine` (written by the pipeline)
  authoring/criteria.json   [{"id": "AC1", "text": "...", "mode": "<failure mode id>",
                              "sentence": "<the ticket sentence, word for word>",
                              "tests": ["<CTRF test name>", ...]}, ...]
  authoring/mutations/      NN-<AC id>-<what>.patch, at least one per criterion: a plausible
                            wrong fix, written as a diff that applies ON TOP of the reference
  authoring/selfcheck.sh    applies reference, runs the hidden tests outside Docker if the
                            toolchain allows; used by the builder only
```

## Hidden tests
- Test names in the CTRF report equal the names listed in `criteria.json`.
- Every test asserts positively and whole: set-up proven present, the whole value compared,
  a refused write proven not applied by reading back. Never absence only.
- Expected values come from a live twin where the ticket states parity ("same answer as a
  never-used id", "identical to the run without the failure"), never hand-written.
- No test depends on another test's side effects; each boots its own state.
- No sleeps, no wall-clock races, no network. The whole suite runs in under 5 minutes at
  one CPU. A timeout is a task defect, not a slip.
- No comment in tests that names the ticket or a criterion.

## Mutations
- Each changes request-time behaviour only (no imports, start-up, fixtures, tests).
- Each must turn at least one test of its own criterion red BY ASSERTION while the rest of
  the suite still runs to the end.
- They represent the textbook build for each failure mode, so they double as the proof that
  the hidden cases separate the textbook build from the real one.

## Integrity (checked by `rsi gate --stages audit`, no Docker needed)
- Nothing under `tests/`, `solution/` or `authoring/` is in the image, in any form. The
  Dockerfile copies `app/` only.
- The starting repo does not already contain the reference's lines, and the ticket never
  names a test, a criterion file or the grader.
- `tests/test.sh` installs nothing, never pipes the test runner without `set -o pipefail`,
  never ends the runner with `|| true`, and writes reward 1 only inside the runner's `if`.
- The answer is the repo diff and nothing else. The gate replays every passing attempt's
  `agent.patch` alone in a clean container (`rsi/gate.py`, `replay`); a pass that does not
  survive is not a pass. A solution that needs anything outside `/app` fails by design.
- Code under test never shares a process with an answer key. Tests compute expected values
  from a live twin, or, when the values are precomputed, run the candidate in a separate
  process and compare data with data (`TEST_SH` and `RUN_JOB` in `rsi/family.py` are the
  worked example: key off the disk, candidate as `nobody`, JSON back).

## Fairness
- Every hidden test maps to one ticket sentence quoted word for word (`criteria.json`).
- The reference passes every test three times from a clean image.
- If the ticket changes, every earlier difficulty attempt is void.
