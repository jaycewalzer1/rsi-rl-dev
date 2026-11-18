# rsi-rl-dev

A factory for RL coding environments that improves itself where that is possible to check. It
takes the ways coding agents fail (from practitioners, from papers, and from its own failed
attempts), stacks several into one task, and uses [Harbor](https://github.com/harbor-framework/harbor)
to hold every task at about a **25% pass rate for frontier models**: hard enough to teach
something, possible enough to give a learning signal.

What "self-improving" means here, exactly:
- **Inside a run**, the design policy learns from every measured task which failure modes the
  current solver slips on and aims the next design at the band.
- **Across runs**, `rsi improve` has Claude Code change the factory's own code and prompts, and
  a part of the repo it cannot edit decides whether a change stays
  ([`docs/RSI_LOOP.md`](docs/RSI_LOOP.md)).
- **Not** the model that writes the tasks. Its weights are never trained, and
  [`docs/WHY_NOT_TRAIN_THE_AUTHOR.md`](docs/WHY_NOT_TRAIN_THE_AUTHOR.md) says why.

[`docs/RESEARCH.md`](docs/RESEARCH.md) maps every part to the published work it follows, what
was left out, and what has and has not been run.

Built from what we learned at the Frist RL-task hackathon (Nov 15 2026): every first task design came
out too easy, and the difficulty rules in [`docs/DIFFICULTY_RULES.md`](docs/DIFFICULTY_RULES.md)
are what it took to get frontier agents to slip fairly. The six hackathon tasks and their measured
Opus 5.5 pass counts ([`examples/frist-hackathon/`](examples/frist-hackathon/)) are the loop's
starting history and the author model's worked examples.

## The loop

```
 X + HN posts ──► scan ──► complaints.jsonl    concrete failures only (judge drops venting, jokes)
                             │
 arXiv + Semantic ──► papers ──► failure_modes.json   a mode needs a complaint AND a paper, or to have
 Scholar                     │      ▲                 been read out of this factory's own failed
                             │      │                 attempts; each mode is tagged with its evidence
                             ▼      │
                          combine ──┼──► design.json   3-4 modes whose predicted pass rate is nearest the
                             │      │                  target; central-change and duplicate review
                             ▼      │
                           build ───┼──► Harbor task   (docs/TASK_CONTRACT.md)
                             │      │
                             ▼      │
        ┌───────────────── gate (testing suite) ─────────────────┐
        │ shape · audit · oracle ×3 · nop · mutations · shortcut  │
        │ difficulty (K cold attempts, passes replayed) · hinted  │
        │ inspect (3 reviews)                                     │
        └────────────────────────┬────────────────────────────────┘
                                 ▼
                             calibrate
              too easy  → add the mode that would land it on target
              too hard  → hinted attempts decide: possible → drop the case family most
                          attempts miss; unproven → fix the tests nobody passed
              unfair    → restate the ticket sentence or drop the cases
              in band   → history row, into the pool
                                 │
                                 ▼
          history.jsonl  who slipped on which modes, together or apart ──► combine
          mine           what each failed attempt built, read from its patch ──► failure modes
          archive.jsonl  the pool: novelty check, `rsi recheck` when the solver changes, retirement
```

The feedback inside the loop: every calibrated task writes to `runs/history.jsonl` which failure
modes each solver slipped on, and which slipped together in one attempt. `combine` fits a small
model to that (a task is solved when every stacked mode is handled: product of handle rates,
times pair interactions) and picks the stack whose predicted pass rate is nearest the target.
On six hackathon tasks that model does not yet beat predicting the mean (docs/RESEARCH.md §6);
`combine.backtest()` is how to check again as history grows.

## Quick start

```bash
uv sync
uv tool install harbor            # or: export RSI_HARBOR="uv run --project /path/to/harbor harbor"
export X_BEARER_TOKEN=...         # optional; without it the scan falls back to web-search snippets
export ANTHROPIC_API_KEY=...      # or CLAUDE_CODE_OAUTH_TOKEN (claude setup-token): used by Harbor's claude-code solver

uv run rsi loop --tasks 1         # whole pipeline, one task
```

Stage by stage:

```bash
uv run rsi scan    --run r1       # runs/r1/complaints.jsonl
uv run rsi papers  --run r1       # runs/r1/failure_modes.json
uv run rsi combine --run r1       # runs/r1/design.json
uv run rsi build   --run r1       # runs/r1/task/<slug>/
uv run rsi gate      runs/r1/task/<slug>                       # every stage
uv run rsi gate      runs/r1/task/<slug> --stages oracle,nop,mutations
uv run rsi summary   runs/r1/task/<slug>                       # STALE if the task changed since
uv run rsi calibrate runs/r1/task/<slug>
uv run rsi mine      runs/r1/task/<slug>                       # read its failed attempts for the slips they made
uv run rsi mine --promote                                      # slips of a new kind, seen twice, become failure modes
uv run rsi pool                                                # calibrated tasks, active and retired
uv run rsi recheck                                             # re-measure them when the solver changes
```

Try the gate without spending a model call:

```bash
uv run python -m rsi.audit examples/tasks/smoke-ledger         # the static audit alone, no Docker
uv run rsi gate examples/tasks/smoke-ledger --stages audit,oracle,nop,mutations   # Docker (it is too small to pass shape, on purpose)
uv run rsi family-task --run check1                            # the shipped task family as a Harbor task;
                                                               # it prints the gate command to run on it
```

## The testing suite

`rsi gate` is a Harbor-native port of the pre-submit suite we ran at the hackathon. Each stage
writes `authoring/gate/<stage>.json` stamped with the task checksum.

| stage | passes when |
|---|---|
| shape | contract files exist, ticket 150 to 250 words, every criterion has hidden tests and a mutation, every criterion's sentence is in the ticket word for word |
| audit | static leak and verifier checks (`rsi/audit.py`, no model, no Docker): no hidden file in the image, no reference already in the starting repo, no grader hints in the ticket, a `test.sh` that cannot hand out reward by accident |
| oracle | the reference solution scores 1 in three separate Harbor runs |
| nop | the untouched starting repo scores 0 |
| mutations | every `authoring/mutations/*.patch` applied on top of the reference scores 0, with failing tests reported by assertion |
| shortcut | three red-team attempts per model (exploit the grader; the same with the hidden test names; hard-code the narrowest pass) score 0. A pass is replayed in a clean container: if it dies there it was tampering (logged, neutralised); if it survives, the judge reads it next to the reference and only a real implementation clears it |
| difficulty | K cold attempts per solver model; pass rate in the band with at least one full solve (Wilson interval, errored trials void). A pass counts only if its recorded patch passes again in a clean container |
| hinted | only when no cold attempt solved it: attempts that are told every criterion and where builds slip. A solve proves the task possible; none means unproven, and calibration fixes the tests instead of deleting cases |
| inspect | every failing test of every failed attempt maps to a ticket sentence quoted word for word (the quote is string-matched against the ticket). Each distinct failure is reviewed three times and one "unfair" is enough; tests every attempt failed in its own way, or every hinted attempt failed, are flagged for the judge |

## The improvement loop

The loop above tunes what it stacks. `rsi improve` changes the factory itself: a headless
Claude Code session edits the code and prompts in a git worktree, and a part of the repo it
cannot edit (the gate, the scorecard, the existing tests) decides whether the change stays.

```bash
uv run rsi score                          # in-band tasks per 100 attempts, and the guardrails
uv run rsi improve                        # one generation: propose, check, commit on rsi/gen-N
uv run rsi improve --live 4               # also run parent and candidate on 4 tasks each
uv run rsi improve --log
```

[`docs/RSI_LOOP.md`](docs/RSI_LOOP.md) has the steps, the protected list and what counts
as evidence. No generation has been run yet.

## Business data in, a measured Qwen improvement out

`rsi factory` takes a company's data export and produces proven task families and Harbor
tasks; `rsi qwen` post-trains a local Qwen on them and measures it before and after. The
post-training is rejection fine-tuning (sample, keep what the oracle grades fully right, SFT
on that); it is not a policy-gradient method. Every
agent is logged to `runs/<run>/events.jsonl`; `rsi demo` shows the run live or as a replay.

```
 business export (CSV) ──► intake: prune (PII, free text, test rows, dupes, orphans), money to
                            integer cents, cut into disjoint slices; a steward agent per table
 HN (Algolia) + X ───────► scouts ──► labellers ──► complaints.jsonl
 arXiv + Semantic Scholar ► one reader per complaint category ──► failure_modes.json
                         ──► authors: a planner proposes business questions for THIS data; one
                             author agent per question writes a task family (rsi/family.py):
                             a plain job plus 6-8 switchable rules, each tied to a failure mode
                               · mechanical checker (rsi/family_check.py): every rule must change
                                 the answer on every slice of the real data, alone and in the full stack
                               · a different model must solve the ticket cold and match the oracle
                         ──► export: each family's top level as a Harbor task ──► rsi gate
                         ──► qwen: calibrate, benchmark, post-train, benchmark
```

**Task families** (`rsi/family.py`). A family is one business question with an exact
oracle and switchable rules. A task turns L rules on (its level); every rule that is off is
replaced by an explicit default sentence, so the ticket always says exactly what to
compute. The top level is the "impossible" start; level 0 is the plain job. Grading is
all-or-nothing on every slice of the data: the slice the solver saw sample rows from and
the hidden ones. `rsi/families/shoe_store.py` is the worked example, over the synthetic
shoe store in `examples/business/` (`rsi/business_gen.py`).

**The Qwen leg** (`rsi/qwen.py`) runs local Qwen3-VL-8B-Instruct (4-bit MLX, from the
kokopelli repo) through kokopelli's GPU queue:

1. calibrate: per family, come down from the top level. Levels are screened with 5 tasks until
   the first solve; from there every level gets 20, and the descent stops once the pass rate
   reaches the headline target. A family Qwen cannot solve even with every rule off is reported
   as out of reach and kept out of the headline
2. settle and benchmark before: per family pick a headline level and a harder level so the
   BEFORE pass rates pooled over families sit at 50% (headline) and 25% (harder)
   (`--headline-target`, `--harder-target`). The pick is measured on the held-out benchmark
   tasks and adjusted until the levels it picks are levels it measured; decoding is greedy, so
   those are the before rates themselves. Also measured: the plain job of any out-of-reach
   family, and a control family from a different dataset. All-or-nothing grading
3. collect: sample answers on training combinations from the plain job up to the harder
   level. An answer that crashes on the rows the model was shown gets its own
   traceback back and up to two more goes. Keep only answers the oracle grades fully right.
   The reward is the only teacher: no reference solution or expected value reaches the model
4. train: LoRA (r16 on q/k/v/o) with mlx-vlm's SFT trainer on every kept answer, as
   ticket -> final correct code, loss on the answer only
5. benchmark after: the same held-out tasks with the adapter, with a paired exact test
6. repeat 3 to 5 (3 rounds by default); each round samples with the previous round's adapter

Each round also reports how many training tasks had both passing and failing samples: the
share a group-relative RL step could have learned from. And two ablation arms test the
stacking claim: `rsi qwen --run NAME --arm single` trains only on tickets with one rule on,
`--arm stacked` only on tickets with two or more, both against the same before benchmark.

Benchmark rule combinations are never trained on, so "after" measures transfer to rule
stacks the model never saw graded. `runs/<run>/qwen/REPORT.md` names model, weights and
adapter. Stages write as they go, so a stopped job resumes where it was.

```bash
# a real dataset: Olist, ~100k orders from a Brazilian marketplace (CC BY-NC-SA 4.0, not in this repo)
mkdir -p data/olist/raw && cd data/olist/raw && for f in customers orders order_items order_payments order_reviews products sellers; do
  curl -sLo $f.csv https://huggingface.co/datasets/bulutttt/olist-raw-data/resolve/main/olist_${f}_dataset.csv; done; cd -

uv run rsi factory --run olist1 --business data/olist/raw      # needs claude (agents) + docker (gate)
uv run rsi qwen    --run olist1                                # queues the Qwen job
uv run rsi demo    --run olist1                                # http://127.0.0.1:8791
```

## Configuration

All settings are environment variables (see `rsi/config.py`):

| variable | default |
|---|---|
| `RSI_AUTHOR_MODEL` | `claude-opus-5-5` (writes designs and tasks) |
| `RSI_JUDGE_MODEL` | `claude-sonnet-5-5` (labels complaints, judges papers, reviews designs, inspects failures) |
| `RSI_CHECK_MODEL` | `claude-opus-5-5` (must solve a new family's tickets cold; keep it different from the author) |
| `RSI_SOLVER_MODELS` | `anthropic/claude-opus-5-5` (comma-separated; each must land in the band) |
| `RSI_SOLVER_AGENT` | `claude-code` (any Harbor agent) |
| `RSI_REDTEAM_MODELS` | the solver models (comma-separated; models told to get reward without building the feature) |
| `RSI_REPLAY` | `1` (replay every pass from its recorded patch; `0` trusts the in-container reward) |
| `RSI_TARGET_RATE` / `RSI_BAND_LOW` / `RSI_BAND_HIGH` | `0.25` / `0.15` / `0.35` |
| `RSI_ATTEMPTS` | `8` cold attempts per model per round |
| `RSI_MAX_ROUNDS` | `4` calibration rounds |
| `RSI_INSPECT_REVIEWS` | `3` independent fairness reviews per distinct failure; one "unfair" is enough |
| `RSI_HINTED`, `RSI_HINTED_ATTEMPTS` | `auto` (hinted attempts only when nobody solved it cold; `always`, `off`), `4` |
| `RSI_MIN_MODES` / `RSI_MAX_MODES` | `3` / `4` failure modes stacked per design |
| `RSI_MINE` | `1`: after calibration, read each failed attempt's patch for the slip it made |
| `RSI_NOVELTY_MAX` | `0.6`: a design this similar in wording to an existing task is dropped as a copy |
| `RSI_STALE_WEIGHT` | `0.25`: weight of history measured with a solver that is not in `RSI_SOLVER_MODELS` |
| `RSI_HARBOR`, `RSI_HARBOR_ENV` | `harbor`, `docker` |
| `RSI_LLM` | `cli` (headless `claude -p`) or `api` (Messages API with `ANTHROPIC_API_KEY`) |

Every model call is logged with its model in `runs/llm_calls.jsonl`; every difficulty result
records solver model and agent.

## Status

Run for real, before the research-sync branch:
- Harbor oracle / nop / mutation stages on the smoke task (Docker, local), and scan + papers
  (web-snippet fallback, Sonnet 5.5 judge): 13 concrete complaints, 5 paper-backed modes.
- The six hackathon tasks in `examples/frist-hackathon/`.

Built on the research-sync branch and checked by unit tests only (`uv run pytest`: no Docker,
no model): the audit stage, the three-phase family verifier (its real `test.sh`
runs outside Docker in `tests/test_verifier.py`, with cheats), replay, the three red-team
prompts, the hinted stage, the three-review inspect, the stack-prediction policy, the archive,
the rollout reader, the Qwen arms and reward-variance report, and the improvement loop (whole
generations in a throwaway git repo with the session faked). None of it has been through
Harbor or a real model yet. `docs/RESEARCH.md` ends with the order to run things in.

Not done: no Qwen run (there is no `REPORT.md` anywhere), no generation of `rsi improve`.

Costs to know before a first run:
- A full calibration is roughly 8 to 32 frontier-agent attempts per task. On top of that the
  gate now spends 3 red-team attempts per task (was 1), 4 hinted attempts when nobody solves a
  task cold, one Docker-only replay per pass, 3 judge calls per distinct failure in inspect
  (was 1 per attempt), and 1 judge call per distinct failed attempt in the rollout reader.
  `RSI_REPLAY=0`, `RSI_HINTED=off`, `RSI_INSPECT_REVIEWS=1` and `RSI_MINE=0` turn each back off.
- `rsi improve --live N` runs the whole pipeline 2N times.
- The X scan without an API token sees only search snippets (x.com refuses fetchers), so its
  evidence is tagged `snippet`, the weakest tier.
