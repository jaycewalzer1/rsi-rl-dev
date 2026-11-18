# Difficulty rules

Learned at the Frist RL-task hackathon (2026-11-15), where every first design came out too easy:
cold Opus 5.5 solved three of the first tasks 3 of 3 or 2 of 2 in 10 to 20 minutes, and 21 of 22
on a fourth. These rules are what the `combine`, `build` and `calibrate` prompts enforce.

## The band
- Measure: cold frontier agent (Claude Code headless through Harbor), ticket and starting repo
  only, graded all-or-nothing on the hidden tests. One failed test is a failed attempt.
- Target about 25%. In band: pass rate between 15% and 35% with at least one full solve.
  Zero solves is not "very hard", it is unproven: a task nobody solves may be impossible.
  The gate settles it with hinted attempts: the solver is told every criterion and where
  builds slip. If it still cannot solve the task, fix the task; never soften it by deleting
  cases until something passes.
- Trials that error before grading (gateway 429/503, harness crash) are void, not failures.
- Small counts mislead: 1 of 4 became 3 of 6 in twenty minutes. The gate uses a Wilson interval
  and asks for more attempts until the interval clears the target.

## What makes frontier agents slip
- Parity at width: one general rule that must hold on many surfaces the ticket names only by
  business class (see, count, search, download, undo), above all on writes, tested against a
  live twin. There is no single place to fix it.
- Derived values on unlike surfaces: totals, exports, history, a line with zero units.
- State across two processes under failure: in-process locks and caches look right with one copy.

## What does not work
- Traps the ticket states. Neighbours the ticket names. Agents read carefully.
- More of the same kind of work. More cases behind one row filter: 120 cases, 0 slips.
- Small tricky tasks. They get solved in 15 minutes.
- One failure mode carrying the whole difficulty: that gives a coin flip (3 of 6), not a band.
  Stack at least three independent modes, each in its own criterion and part of the code.

## Before building
- Central-change test: if one filter, helper, table or lock satisfies the ticket, reject it.
- Name the textbook build for each mode and the hidden case it fails. No such case, no build.

## Fair, or it is worthless as training signal
- Ticket of 150 to 250 words of general rules; exact codes in Pins.
- Every failing test maps to one plain ticket sentence. A miss that needs a second reading is
  the task's defect, not the agent's slip: fix the ticket or drop the case, then re-measure.
- A test every attempt fails in a different way is usually an unstated requirement.

## Counting honestly
- Fix the count rules before results come in. Never add or change cases after solves just to
  move the number; a revision makes a new task version and voids the old attempts.
