# rsi-rl-dev

A self-improving factory for RL coding environments. It reads what developers complain about
on X, checks each complaint against published research, stacks several real failure modes into
one task, and uses [Harbor](https://github.com/harbor-framework/harbor) to hold every task at
about a **25% pass rate for frontier models**: hard enough to teach something, possible enough to
give a learning signal.

Built from what we learned at the Frist RL-task hackathon (Nov 15 2026): every first task design came
out too easy, and the difficulty rules in [`docs/DIFFICULTY_RULES.md`](docs/DIFFICULTY_RULES.md)
are what it took to get frontier agents to slip fairly. The six hackathon tasks and their measured
Opus 5.5 pass counts ([`examples/frist-hackathon/`](examples/frist-hackathon/)) are the loop's
starting history and the author model's worked examples.

## The loop

```
 X posts ──► scan ──► complaints.jsonl      concrete failures only (judge drops venting, jokes)
                          │
 arXiv + Semantic ──► papers ──► failure_modes.json   a mode survives only with a complaint AND a paper
 Scholar                  │                           (+ seed modes from seeds/failure_modes.json)
                          ▼
                       combine ──► design.json        ≥3 independent modes, central-change review
                          │        ▲
                          ▼        │  runs/history.jsonl: which modes actually made solvers slip
                        build ──► Harbor task dir     (docs/TASK_CONTRACT.md)
                          │        │
                          ▼        │
          ┌──────────── gate (testing suite) ─────────────┐
          │ shape · oracle ×3 · nop · mutations · shortcut │
          │ difficulty (K cold attempts) · inspect         │
          └──────────────────────┬─────────────────────────┘
                                 ▼
                             calibrate
              too easy  → add a criterion from a fresh failure mode
              too hard  → drop the case family most attempts miss
              unfair    → restate the ticket sentence or drop the cases
              in band   → record mode slips in history, done
```

The self-improving part: every calibrated task writes which failure modes made solvers fail into
`runs/history.jsonl`. `combine` samples modes from a Beta posterior over those slips, so the next
design leans on modes that actually work against current models, and drops the ones they have
learned to handle.

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
uv run rsi gate      runs/r1/task/<slug>                       # all seven stages
uv run rsi gate      runs/r1/task/<slug> --stages oracle,nop,mutations
uv run rsi summary   runs/r1/task/<slug>                       # STALE if the task changed since
uv run rsi calibrate runs/r1/task/<slug>
```

Try the gate on the bundled smoke task (Docker only, no model calls):

```bash
uv run rsi gate examples/tasks/smoke-ledger --stages oracle,nop,mutations
```

## The testing suite

`rsi gate` is a Harbor-native port of the pre-submit suite we ran at the hackathon. Each stage
writes `authoring/gate/<stage>.json` stamped with the task checksum.

| stage | passes when |
|---|---|
| shape | contract files exist, ticket 150 to 250 words, every criterion has hidden tests and a mutation |
| oracle | the reference solution scores 1 in three separate Harbor runs |
| nop | the untouched starting repo scores 0 |
| mutations | every `authoring/mutations/*.patch` applied on top of the reference scores 0, with failing tests reported by assertion |
| shortcut | a solver told to game the grader without building the feature scores 0 |
| difficulty | K cold attempts per solver model; pass rate in the band with at least one full solve (Wilson interval, errored trials void) |
| inspect | every failing test of every failed attempt maps to a ticket sentence quoted word for word |

## The swarm, the business data and the Qwen leg

`rsi swarm` runs the factory as a swarm of agents over a company's data export and logs
every agent to `runs/<run>/events.jsonl`; `rsi demo` shows it live or as a replay.

```
 business export (CSV) ──► intake: loader + pruner + steward agent per table
                            PII columns, test rows, exact dupes, sparse/constant columns,
                            orphans, enum noise dropped; resent events kept (they are data)
 HN (Algolia) + X ───────► scouts (one per query) ──► labellers ──► complaints.jsonl
 arXiv + Semantic Scholar ► one reader per complaint category ──► failure_modes.json
                         ──► crossref: mapper agents tie each task rule to complaints + papers
                         ──► design: the impossible task (all 8 rules) and the ladder down,
                             weakest-slipping rule off first (Beta prior from history)
                         ──► export: ladder steps as Harbor tasks ──► rsi gate (your pipeline)
                         ──► qwen: benchmark, anneal down the ladder, post-train, benchmark
```

**Tasks from the business data** (`rsi/bizops.py`). One function, refunds owed per order,
with up to eight stacked rules, each one a sourced failure mode: in-store returns stamped
in local time (sibling path), the 30-day window in store-local calendar days (time zones),
final-sale exceptions (existing rule carried), resent webhook events (idempotency),
quantity caps in request order (state across steps), net-of-discount refunds (derived
value), exchanges that refund nothing but use up quantity, and one store only (tenancy).
Rules a task leaves out are replaced by an explicit default sentence, so the ticket always
says exactly what to compute. Oracles are exact; grading is all-or-nothing over the real
pruned data plus two twin datasets. Exported tasks follow `docs/TASK_CONTRACT.md`: one
mutation per rule (the reference with that rule forgotten), hidden tests mapped to the
rule whose mutation they catch.

The sample data in `examples/business/shoe-store/raw` is **synthetic** (`rsi/business_gen.py`,
seed 7): a messy shoe-store export. Point `--business` at any folder of CSVs; the bizops
rules are written for this schema, the intake is not.

**The Qwen leg** (`rsi/qwen.py`) runs local Qwen3-VL-8B-Instruct (4-bit MLX, from the
kokopelli repo) through kokopelli's GPU queue:

1. benchmark before: held-out rule combinations at every level 8..1, greedy, all-or-nothing
2. anneal: start at level 8 and walk the swarm's ladder down until the sampled pass rate
   reaches the band (15 to 35%, target 25%)
3. collect: rejection sampling on training combinations at and above that level, keep only
   answers the oracle grades fully right
4. train: LoRA (r16 on q/k/v/o) with mlx-vlm's SFT trainer, loss on the answer only
5. benchmark after: the same held-out set with the adapter

Benchmark combinations are never trained on (`bizops.split_combos`), so "after" measures
transfer to rule stacks the model never saw graded. `runs/<run>/qwen/REPORT.md` names model,
weights and adapter.

```bash
uv run rsi swarm --run r1 --gate shape,oracle,nop,mutations     # needs claude (judge) + docker (gate)
uv run rsi qwen  --run r1                                       # queues the Qwen job
uv run rsi demo  --run r1                                       # http://127.0.0.1:8791
uv run rsi demo  --run r1 --export r1.html                      # self-contained replay
```

## Configuration

All settings are environment variables (see `rsi/config.py`):

| variable | default |
|---|---|
| `RSI_AUTHOR_MODEL` | `claude-opus-5-5` (writes designs and tasks) |
| `RSI_JUDGE_MODEL` | `claude-sonnet-5-5` (labels complaints, judges papers, reviews designs, inspects failures) |
| `RSI_SOLVER_MODELS` | `anthropic/claude-opus-5-5` (comma-separated; each must land in the band) |
| `RSI_SOLVER_AGENT` | `claude-code` (any Harbor agent) |
| `RSI_TARGET_RATE` / `RSI_BAND_LOW` / `RSI_BAND_HIGH` | `0.25` / `0.15` / `0.35` |
| `RSI_ATTEMPTS` | `8` cold attempts per model per round |
| `RSI_MAX_ROUNDS` | `4` calibration rounds |
| `RSI_HARBOR`, `RSI_HARBOR_ENV` | `harbor`, `docker` |
| `RSI_LLM` | `cli` (headless `claude -p`) or `api` (Messages API with `ANTHROPIC_API_KEY`) |

Every model call is logged with its model in `runs/llm_calls.jsonl`; every difficulty result
records solver model and agent.

## Status

- Verified live: Harbor oracle / nop / mutation stages on the smoke task (Docker, local), and
  scan + papers (web-snippet fallback, Sonnet 5.5 judge): 13 concrete complaints, 5 paper-backed modes.
- Unit tests: `uv run pytest`.
- The difficulty, shortcut and calibrate stages need solver credentials for Harbor and spend
  real model time: a full calibration is roughly 8 to 32 frontier-agent attempts per task.
- The X scan without an API token sees only search snippets (x.com refuses fetchers), so its
  evidence is marked `web_snippet` and weighted as weaker.
