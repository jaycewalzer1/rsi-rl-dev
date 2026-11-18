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

## 3. Reward integrity: red team and replay

**The work.** The Terminal-Bench guideline (arXiv:2604.28093) recommends "please hack" trials,
with the test signatures put in the prompt, to find exploitable tasks before release. TRACE
(arXiv:2601.20103) finds judges detect exploits better when they compare a suspect
trajectory with benign ones than when they classify it alone. SWE-bench style evaluation
applies the model's patch in a clean container, so nothing the agent did outside the repo
can touch the grade.

**What this repo does.**
- The `shortcut` stage was one attempt, same model, one prompt. It now runs three
  strategies per red-team model (`RSI_REDTEAM_MODELS`): exploit the grader; the same with
  the hidden test names in the prompt; hard-code the narrowest thing that passes.
- Every passing attempt is replayed (`gate.replay`): its recorded `agent.patch` is applied
  to a clean copy of the task and verified again. In the red-team stage, a pass that dies
  on replay was tampering and is logged as neutralised; a pass that survives is read by the
  judge next to the reference, and only "a real implementation", backed by lines that are
  literally in the patch, clears it. Everything else blocks the task.
- The `difficulty` stage counts a cold pass only if it survives replay. Passes that do not
  are listed as `unreproduced`.
- The contract now makes `tests/test.sh` write reward 0 first and record `agent.patch`.

**Left out.** Replay re-runs the verifier, it does not prove the tests are right. A patch
that special-cases what the tests check survives replay; the three red-team prompts and the
contrastive judge are the only defence against that, and TRACE is clear that judges miss
exploits. Files git ignores are not in the patch, so a solution that depends on them fails
replay by design.

**Status.** Built. `tests/test_redteam.py` fakes Harbor and the judge and checks every
outcome. Not yet run under Harbor.

## 4. Feasibility: the hint gap

**The work.** In SPADE (arXiv:2608.19197) one model writes environments and solves them.
The designer also writes a privileged hint, the solver plays each environment with and
without it, and the designer is rewarded on the gap (regret, floored at zero) plus an anchor
that favours a mid-range win rate; the paper's overview gives the weights as 0.4 and 0.6.
The reading of the gap is the useful part: a large gap means solvable with guidance but not
yet alone; success both ways means mastered; failure both ways means intractable.

**The weakness it fixes here.** `docs/DIFFICULTY_RULES.md` already said "zero solves is not
very hard, it is unproven", and the calibration loop then softened the task anyway by
deleting the cases most attempts missed. That cannot tell a hard task from a broken one.

**What this repo does.** The `hinted` stage runs when no cold attempt solved the task (at
least 6 graded). The solver gets the ticket plus reviewer notes: every criterion, and where
the textbook build slips if the design recorded it. No test code.
- a hinted solve: feasible. Calibration softens the task as before, and says so
- no hinted solve: unproven. Calibration sends back the tests every hinted attempt failed,
  with the instruction to restate the ticket or fix the cases, not to delete cases until
  something passes

`gate.design_score` records regret, the band anchor and their 0.4/0.6 mix for each task.
Hinted attempts never count toward the pass rate.

**Left out.** SPADE trains its designer on this reward. Here the designer is a frozen
frontier model, so the score is only recorded. A hinted solve proves the task is possible
for a solver that was told where the traps are; whether the ticket alone is enough is the
inspect stage's question. SPADE itself notes that a hint can mislead and that regret is an
estimate.

**Status.** Built. `tests/test_hinted.py`, Harbor faked. Not yet run under Harbor.

## 5. Fairness: repeated review, and what the attempts show

**The work.** EvoEnv (arXiv:2605.14392) audited its own semantic self-review: one review of
a generated environment reached F1 47.1% against an external label; three independent
reviews with an any-reject rule reached 87.0%. LHAW (arXiv:2602.10525) makes the wider
point about ambiguity: do not ask a model whether a task is underspecified, run agents and
classify by whether their end states diverge. The Terminal-Bench guideline asks the same of
task authors: instructions specified well enough "that meeting them implies passing the
tests", and failures reviewed in batches.

**What this repo already had, and it is the strong part.** Every failing test of a failed
attempt must map to a ticket sentence quoted word for word, and the quote is checked against
the ticket by string match, not by the judge's say-so.

**What changed.**
- Each distinct set of failing tests is reviewed `RSI_INSPECT_REVIEWS` times (default 3) and
  one "unfair" is enough. Reviews run per distinct set, not per attempt.
- The attempts are evidence. A test that every failed attempt failed, each with a different
  error message, is flagged (a rule in `DIFFICULTY_RULES.md` that was prose until now).
  So is a test that every hinted attempt failed. Flags go into the judge's prompt and into
  `inspect.json`; they do not decide the verdict on their own.
- The difficulty stage stores one line per failing test saying how it failed (from CTRF).

**Left out.** LHAW removes information from a known-good task and measures the damage. This
stage only reads divergence off attempts it already has, so with 3 to 8 attempts the signal
is thin. The 47% to 87% figures are EvoEnv's, for a 4B reviewer on generated puzzles; the
judge here is a frontier model on tickets, and nobody has measured its review accuracy.

**Status.** Built. `tests/test_inspect.py`, judge faked. Not yet run with a real judge.

## 6. The design policy: aim at a pass rate, with stacked modes

**The work.** Three lines of work say the same thing from different sides.
- Curriculum design: Sampling for Learnability (Rutherford et al., NeurIPS 2024,
  arXiv:2408.15099) shows that regret-style scores in practice track success rate, and that
  scoring levels by p(1-p), the ones solved sometimes and not always, works better. A 2026
  survey post on environment scaling puts it as "difficulty and training value are simply
  different axes": with a binary reward the variance p(1-p) is the signal a group-relative
  method like GRPO can use, and it is zero at both ends.
- Psychometrics: work on agentic coding benchmarks fits item response models to predict
  per-task success before running agents (Agent Psychometrics, arXiv:2604.00594), and a
  mid-range filter (keep tasks with 30 to 70% pass rate, arXiv:2603.23749) cuts evaluation
  cost because mid-range items carry the information.
- Composition: Skill-Mix (arXiv:2310.17567) makes hard, unseen tasks by combining k skills
  from a list, with k as the difficulty dial.

**What was here.** A per-mode bandit: sample a slip rate per failure mode, take the top
three. Three problems. It aimed at the most slips, when the target is a band. It treated
modes as independent, while the hackathon finding was that the stack matters. And it had a
bug: a mode got an update only if it appeared in `mode_slips`, so a mode that never slipped
(idempotency, 0 in 15) was never marked as handled and its "handicap" did not exist.

**What this repo does now.** `rsi/combine.py`:
- `stats`: per mode, attempts slipped and attempts handled, counting every mode in the task
  (the bug fix). History measured with a solver outside `RSI_SOLVER_MODELS` counts at
  `RSI_STALE_WEIGHT` (0.25): a newer model handles more.
- `predict`: P(solve a stack) = product of handle rates x pair interactions. An interaction
  is estimated only from rows that record which modes slipped in the same attempt, and is
  shrunk toward 1 (independent).
- `pick_modes`: Thompson-samples the handle rates and takes the stack of 3 to
  `RSI_MAX_MODES` modes whose predicted pass rate is nearest the target, with a small
  penalty for a stack where one mode carries everything.
- `calibrate.record` writes, per solver, which modes slipped together in each attempt, the
  rate the design predicted, and the rate its first measurement gave.
- `backtest`: leave-one-task-out error of the prediction, next to a baseline that predicts
  the mean pass rate.

**Evidence, for and against.** On the six hackathon tasks the model's leave-one-out error is
0.206 against 0.208 for the baseline. That is a tie. Six tasks with 3 to 6 attempts each
cannot show the product model predicts anything, and the pair term has no data at all yet
(the hackathon rows have no per-attempt patterns). What the fit does reproduce is the
qualitative finding: authorization parity is handled 10 of 10 on its own, derived surfaces 6
of 12, and the in-band tasks stacked both.

**Next test.** After about 20 more calibrated tasks, `combine.backtest()` again. If the model
still does not beat the baseline, the stacking claim is a recipe for writing tasks, not a
predictive model, and `pick_modes` should go back to something simpler.

**Left out.** No learned item-response model (too few solvers and tasks to fit one). No
difficulty prediction from the ticket text.

**Status.** Built. `tests/test_policy.py`.
