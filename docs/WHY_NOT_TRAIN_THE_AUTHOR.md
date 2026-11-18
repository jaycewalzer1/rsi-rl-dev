# Why this repo does not train the model that writes the tasks

**Short version.** The tasks here are written by the newest frontier model and have to be
hard for the newest frontier model. An open-weight model we can train ourselves is weaker
than both, and will almost certainly stay weaker. Training it to write the tasks would be
teaching a junior to set the exam for a senior. So the weights of the author are left
alone, and what improves is everything around it: the code, the prompts, the rules, the
gates and the evidence. That is the loop in `docs/RSI_LOOP.md`.

## What "training the teacher" would mean

In this repo three models do three jobs:

| role | model today | what it does |
|---|---|---|
| author | `RSI_AUTHOR_MODEL` (Claude Opus 5.5) | designs a task, builds its codebase, hidden tests, reference and mutations |
| solver | `RSI_SOLVER_MODELS` (Claude Opus 5.5 in Claude Code) | attempts the task cold; its pass rate is what gets calibrated to about 25% |
| student | local Qwen3-VL-8B, 4-bit | is post-trained on the task families and measured before and after |

"Training the teacher" means making the author a policy with weights we update: reward it
when its tasks land at the right difficulty, punish it when they are broken or trivial,
and let it get better at writing tasks. That is recursive self-improvement in the sense of
the 2026 papers:
- SPADE (arXiv:2608.19197): one model writes environments and solves them, and the
  environment-writing role is trained on a hint-gap regret reward.
- EvoEnv (arXiv:2605.14392): one policy generates and solves Python environments, and the
  generator is rewarded for quality, difficulty and novelty.
- AutoEnvScaling (arXiv:2610.14163): reports both arrangements, a frontier model building
  environments for an open student, and an open model acting as proposer and solver for
  itself.

## Why it works in those papers

Three conditions hold there, and all three matter.

1. **The designer and the solver are the same model.** A model writes tasks for itself, so
   the designer is never weaker than the solver it designs for.
2. **The tasks are small.** A short Python class with a generator, a reference answer and
   a scorer, which a 4B to 30B model can write. EvoEnv reports hundreds of accepted
   environments from ten seeds in a hundred training steps.
3. **The reward is cheap and plentiful.** Measuring one candidate environment is a handful
   of short solver calls on the same GPU, so the designer gets thousands of gradient
   signals.

## Why none of the three holds here

1. **The designer would be weaker than the solver, by a lot.** The product is a task the
   frontier agent solves about one time in four, fairly. Writing one means knowing where
   that agent slips better than it does. At the Frist hackathon the frontier model itself
   could not do this unaided: every first design came out too easy, with cold Opus 5.5
   solving three of the first tasks 3 of 3 or 2 of 2 in 10 to 20 minutes
   (`docs/DIFFICULTY_RULES.md`). It took the stacking rules and a measurement loop to get
   into the band. An 8B model whose own benchmark is a handful of business rules in one
   function is not going to out-think Opus about where Opus fails.

2. **The author we would want to train cannot be trained.** It is a closed model. At the
   time of writing (November 2026), public documentation and reporting say the Claude API
   has no fine-tuning and OpenAI's fine-tuning covers older models and is being wound
   down. Check before relying on that sentence; it changes. The argument does not depend on
   it, because of the next point.

3. **A tuned model is frozen and the frontier is not.** Suppose we could fine-tune today's
   best model on our own task-writing data. In a few months there is a new one. On work
   like this we expect the new untuned model to beat the old tuned one, and the tuning
   would have to be redone from scratch on whatever the vendor then allows. Everything else in this repo
   survives a model upgrade and gets more useful with it: point `RSI_AUTHOR_MODEL` at the
   new model and the gates, the contract, the history and the rules all still apply.

4. **The target moves with the solver.** The band is defined against the newest solver.
   When the solver gets better, tasks have to get harder, and the author has to be at
   least that good. A trained open author would always be designing for something a
   generation ahead of it.

5. **There is no data to train a designer on.** One measured task costs 8 to 32
   frontier-agent attempts, each a full agent session, plus builds. That buys tens of
   (design, outcome) pairs, each with one noisy number attached. Designer training in the
   papers runs on hundreds to thousands of environments.

Put together: the open-weight model we can afford to train is almost definitely never
going to write better frontier tasks than the most recent OpenAI or Anthropic model. Not
because training does not work, but because the comparison is against a moving target that
gets replaced every few months by something we did not have to pay to improve.

## Where the open-weight model does belong

As the student. `rsi qwen` post-trains the local Qwen on the task families and measures
held-out rule stacks before and after. That is the experiment that shows the environments
teach something, and it is the right place for the training budget of one Mac.

## So what is "self-improving" here

The scaffold. `rsi improve` has the newest frontier model, through Claude Code, change the
factory's own code and prompts, one focused change per generation, and a part of the repo
it cannot edit decides whether the change stays (`docs/RSI_LOOP.md`).

- The improver is always the best model available, for free: it is the same model, and it
  upgrades when the vendor ships.
- The thing that improves is the thing that carries over: how to design, what to stack,
  how to check a task is fair and cannot be gamed, what the measurements say.
- It is recursive in a narrow sense you can check: the loop improves the code that makes
  the next tasks, including the code that decides what the next improvement should be
  (`rsi/proposer.py`). It is not a model improving its own weights, and this repo should
  never be described that way.

## When this would be wrong

Reasons to reopen the question, in order of how likely they are to come up:

- **The solver is the open model.** If the goal becomes "the best curriculum for a
  Qwen-class model", the designer and solver can be the same open model and the SPADE
  recipe applies as published. The task families already have a cheap difficulty dial for
  this (how many rules are on).
- **A small trained model as a filter, not as the author.** Predicting a design's pass rate
  from its text (Agent Psychometrics, arXiv:2604.00594), or scoring novelty with learned
  embeddings, could save frontier attempts. That trains a predictor to stand in front of
  the gate. It is a reasonable use of open weights and does not conflict with anything
  above.
- **The vendors open up.** If the current flagship becomes fine-tunable at a sane price
  and the factory has produced thousands of measured designs, tuning the author is worth a
  pilot.
- **Open models close the gap.** If an open model within reach of our hardware comes
  within one generation of the frontier at agentic coding, point 1 weakens.

The scorecard can settle any of these: run `rsi loop` with the author swapped, same seed
and task count, and compare `rsi score`. The prediction this document makes is that a
trained open author's designs fail the central-change review and miss the band far more
often than the frontier author's. If a run shows otherwise, this document is wrong and
should be rewritten.
