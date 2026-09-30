# SHARED-KEY-BENCH-002 results

**The ranked-reading method completed both full1,197-key workloads and separated
the best candidate from its unseen-reading upper bound.** The earlier full
compatibility-state decoder hit its500k-state cap on these same inputs; those
failures remain. This is an artificial feasibility result, not key recovery or
Voynich evidence.

| Workload | Unique candidate tuples | Keys supporting winner | Best-minus-unseen bound, nats | Wall seconds |
| --- | ---: | ---: | ---: | ---: |
| Source-generated | 4,887 | 171 | 4.341520004 | 52.433 |
| Glyph shuffle | 7,507 | 1 | 0.204734124 | 24.782 |

All1,197keys remained in each uniform mixture. Fixedk8produced8,600/7,616per-key
readings before deduplication. Each returned tuple uses one consistent key across
both records. The source-generated winner is supported by171different full keys;
this is compatible uncertainty, not evidence that the literal generating key
is recovered. No generating-text accuracy was measured.

The bound says every tuple absent from all lists has score below the chosen
reading under this finite-bank model. Both margins exceed the fixed floating
tolerance. This is not an interval-arithmetic proof or correctness of the source,
key family or text. The shuffled control also has an optimum: finding an optimal
model reading cannot by itself establish language or meaning.

## Checks and resources

Code/protocolbacce67ce31394da7728703857e3466fd2caf8e8was published and remotely
verified before one invocation. All16,216per-key tuples passed independent literal
re-encoding and source-path replay.78original forward-decoder record checks per
workload agree with backward evidence. Maximum error6.37e-12nats.

A post-run audit verifies38source/input bindings, both complete archive hashes,
all2,394progress rows, list ordering/next-score thresholds, unique candidate counts,
evidence and upper-bound arithmetic, and the winner's full literal key-support
set. It does not independently re-score every candidate against every key; tiny
exhaustive tests validate that algorithm. Full2,095tests+23subtestsPASS,10skips.

Total77.541wall/77.472CPU seconds, peakRSS519,864,320bytes,0paid. External
process78.741seconds/exit0. No cap extension, k change, rerun or dropped workload.
Bulk readings stay ignored with SHA/byte manifests in the compact results.

## Next state

KEY-BANK-FIT-001 is independently scoring complete fitting-only neighborhoods
for all16exposed original learned keys under its frozen90minute planning limit.
It has not read transfer data or answers. Those banks differ substantially from
this artificial uniform bank, and empirical nulls are much more expensive.
Freeze completed banks and a separate inference/evaluation protocol before
claiming any improvement in passage recovery. Incomplete control fits remain
failures and cannot be excluded from a campaign success claim.
