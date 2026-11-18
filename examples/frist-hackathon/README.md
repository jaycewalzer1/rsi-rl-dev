# Frist hackathon examples (Nov 15 2026)

The task designs we built at the Frist RL-task hackathon and what cold Opus 5.5 did on them
(Claude Code headless, ticket and starting repo only, graded all-or-nothing on hidden tests).
The hackathon's starting apps belong to Frist, so this folder has our designs and measured
results only: no app code, tickets or hidden tests.

| task | modes stacked | cases | solved | verdict |
|---|---|---|---|---|
| issue-tracker-security-levels | authz-by-id, derived-surface-omitted, sibling-path-missed | 22 | 1 of 3 | in band |
| issue-tracker-space-administrators | authz-by-id, sibling-path-missed | n/a | 1 of 4 | in band |
| hr-department-pay-reviewers | authz-by-id, derived-surface-omitted, timezone-math | 155 | 1 of 3 | in band |
| web-store-order-lifecycle | derived-surface-omitted, idempotency-under-failure, cross-process-state | 118 | 3 of 6 | too easy |
| shoe-store-warehouse-holds | idempotency-under-failure, seam-assumption, cross-process-state | 62 | 3 of 6 | too easy |
| shoe-store-returns-refund-once | idempotency-under-failure, cross-process-state | n/a | 3 of 3 | dead |

What the numbers say:
- Every in-band task stacked authorization parity at width with a derived surface. The slips came
  from counts, report cells and sibling routes, not from the rule the ticket was about.
- The warehouse task's difficulty all came from one failure mode, so it landed at a coin flip
  (3 of 6) even though it nominally stacked three modes.
- The returns task stated every trap and Opus planned around all of them.

## How the pipeline uses this
- `history.jsonl` is read by `rsi combine` as the starting prior, together with
  `runs/history.jsonl`. It gives each mode a handle rate: derived-surface-omitted 6 of 12
  attempts, sibling-path-missed 4 of 7, idempotency-under-failure 15 of 15. `combine` stacks
  modes so the product of those rates lands near the target pass rate.
- `designs/*.json` (five of the six; space administrators has no write-up) show the expected shape of a design and what a measured one looks like.
  `rsi combine` also passes them to the author model as worked examples.
