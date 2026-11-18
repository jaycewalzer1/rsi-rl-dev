# The improvement loop

What "RSI" means in this repo, in one sentence: Claude Code changes the factory's own code
and prompts, and a part of the repo it is not allowed to change decides whether the change
stays.

The model that writes the tasks is never trained. `docs/WHY_NOT_TRAIN_THE_AUTHOR.md` says why.

## One generation

`rsi improve` runs one. Each step is in `rsi/improve.py`.

1. **Evidence.** Everything the runs have measured, condensed by `rsi/proposer.py`: the
   scorecard, the worst tasks, which gate stages fail and why, unfair and suspect tests, the
   slips read from failed attempts, how well designs were aimed, and what earlier
   generations tried.
2. **Propose.** A headless Claude Code session (the author model) works in its own git
   worktree on a new branch `rsi/gen-N`. It is asked for ONE focused change and a note,
   `docs/improvements/gen-N.md`, with four parts: the evidence, the change, the scorecard
   number it predicts will move, and the result that would prove it wrong. It may run the
   tests and read-only git. Nothing else runs without a person.
3. **Check.** Mechanical, cheapest first. The first failure discards the generation and
   keeps its patch for reading.
   - `protected`: no protected path was edited, deleted or renamed
   - `tamper`: no added line under `rsi/` reaches around protected code at run time
     (assigning to its attributes, setting `RSI_*` variables, writing into a gate folder)
   - `note`: the note exists and has its four parts
   - `tests`: the whole test suite passes in the worktree
   - `backtest`: if `rsi/combine.py` changed, its leave-one-out prediction error on the
     recorded history is not worse than the parent's
4. **Commit** on `rsi/gen-N`. Nothing is merged and nothing is pushed.
5. **Live comparison** (`--live N`). The parent and the candidate each make N tasks through
   the whole pipeline, from the same history and seed, in separate folders. `rsi/meter.py`
   scores both from gate output. The candidate is accepted only if it is more efficient
   (in-band tasks per 100 graded attempts) and no guardrail got worse (unfair tests,
   red-team passes, unproven tasks).
6. **Archive.** `runs/improve/archive.jsonl` records every generation, kept or not, with
   its checks and numbers. The next generation is shown it.

## What a generation may change

Everything that makes tasks:

| open | what lives there |
|---|---|
| `rsi/combine.py` | the design prompt, the central-change review, the stack-selection policy |
| `rsi/build.py`, `rsi/calibrate.py` | the build and revise prompts, what each verdict sends back to the author |
| `rsi/audit.py` | static checks (a generation may add checks; the fixtures keep the old ones firing) |
| `rsi/mine.py`, `rsi/archive.py`, `rsi/papers.py`, `rsi/scan_*.py` | where failure modes come from, novelty, the pool |
| `rsi/author.py`, `rsi/intake.py`, `rsi/swarm.py` | the business-data factory |
| `rsi/proposer.py` | what a generation is shown and how it is asked: the loop's own method |
| `docs/DIFFICULTY_RULES.md`, `seeds/` | the rules learned so far, the seed modes |
| `tests/gen/` | tests a generation writes for its own change |

## What it may not change

How its work is judged (`PROTECTED` in `rsi/improve.py`):

| protected | why |
|---|---|
| `rsi/improve.py`, `rsi/meter.py` | the loop's shell and the score |
| `rsi/gate.py`, `rsi/harbor.py` | every measurement the score is computed from |
| `rsi/config.py`, `rsi/llm.py`, `rsi/store.py` | the band and targets, model calls and their log |
| `rsi/family.py`, `rsi/family_check.py`, `rsi/families/`, `rsi/qwen.py` | graders, the control family, the Qwen benchmark |
| `tests/*.py`, `examples/` | the existing tests and the measured history |
| `docs/TASK_CONTRACT.md`, this file, `docs/WHY_NOT_TRAIN_THE_AUTHOR.md` | what a task must be, and the rules of the loop |
| `pyproject.toml`, `uv.lock`, `.gitignore` | no new dependency without a person |

Changing a protected file is a person's job, in an ordinary commit.

## Why the shell is fixed

The Darwin Gödel Machine (arXiv:2505.22954) is the reference for this kind of loop: an
agent rewrites its own code and each change is validated empirically on benchmarks, since
proving a change beneficial is impossible in practice. Its follow-up, HyperAgents
(arXiv:2603.19461), goes one level up and lets the agent rewrite the procedure it uses to
improve itself, while the outer loop that selects and scores stays fixed and people oversee
every run. This repo makes the same split. `rsi/proposer.py` is editable, so a generation
can improve how later generations look at the evidence. `rsi/improve.py`, the meter and the
gate are not.

The reason is the same one that makes reward hacking a problem one level down: a loop that
can edit its own judge will find that editing the judge is the cheapest way to score.

Two limits, stated plainly:
- The tamper check is a lint over added lines. It catches the obvious ways around the
  protected files, not every way. A person reads `git diff` before merging a generation.
- Nothing merges by itself, with one exception you have to ask for: `--adopt` fast-forwards
  the current branch to a generation the live comparison accepted. Nothing ever pushes.

## What counts as evidence

- The offline checks say a change is not broken. They do not say it is better. A generation
  that passes them is `validated-offline`: safe to read, unproven.
- The live comparison says better or not on that sample. With N tasks per arm the two arms
  build different tasks, so at N = 3 one task landing in or out of the band decides the
  verdict. Treat `accepted-live` at small N as "worth a second run", and budget for it: one
  task costs 8 to 32 frontier attempts plus its builds, and a comparison costs 2N of them.
- The claim that this loop improves the factory is true when accepted generations pile up
  and the scorecard of a fixed-size run goes up across them. No generation has been run
  yet, so today it is a loop with the right shape and no results.

## Running it

```bash
uv run rsi loop --tasks 3                 # the loop needs measured tasks to reason from
uv run rsi score                          # the scorecard a generation is shown
uv run rsi improve                        # propose, check, commit on rsi/gen-N
git diff HEAD..rsi/gen-1                  # read it
uv run rsi improve --live 4 --budget 20   # also compare parent and candidate on 4 tasks each
uv run rsi improve --log                  # every generation so far
git merge --ff-only rsi/gen-1             # adopt one yourself
```

## What it does not do

- One parent at a time. The Darwin Gödel Machine keeps an archive of agents and samples
  parents from it so that a weak-looking branch can still lead somewhere. Here the archive
  is a record, and `--parent` is how you branch from an older generation by hand.
- The scorecard covers the Harbor loop only. The Qwen leg's before and after are not in it.
- It does not improve a model. It improves the scaffold around one. That is the point.
