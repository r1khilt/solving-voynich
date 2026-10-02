# SOURCE-ACTION-DIAG-001 — guided reuse improves; initial dictionary inference fails

The 96,039,982-parameter reader's loss improvement predominantly comes from actions that reuse a dictionary binding already supplied by the correct history. Free reading still fails at its first new binding. This is a measured bottleneck on exposed synthetic Latin development cases, not a manuscript discovery or a verified internal circuit.

[Prospective registration](SOURCE-ACTION-DIAG-001.md), [method and primary-source review](../research/source-action-causal-diagnosis-2026-10-02.md), [actual result](../../results/SOURCE-ACTION-DIAG-001/result.json), and [read-only closure](../../results/SOURCE-ACTION-DIAG-001/closed-check001.json).

## What actually ran

Exact published revision `517251cda522725d2243aa378804d630113a655a` preceded one CPU diagnostic, without retry or extension. Same seed92403, checkpoints0/1000, all64 fixed exposed Pliny episodes, four variants each: **128 checkpoint/case pairs and512 full causal forward passes**. The original training campaign's73 dependencies and all81 diagnostic dependencies stayed unchanged. No optimizer, GPU diagnostic, source-posterior search, reserved author or manuscript evaluation.

Sham duplicates the memory packet. Erase replaces occupied unit features with their own row's unbound feature. Rotate permutes bound units across occupied rows while preserving row identity, occupancy and the unit multiset. The other seven packet tensors—including legal actions, observed cipher, previous actions and correct targets—stay fixed. All variants use the correct previous action history; this is not free behavior under intervention.

Actual status: `PASS_complete_cpu_feature_intervention_and_first_error_diagnostic`. This PASS means the registered diagnostic completed and its numerical controls passed; it does not qualify recovery or a circuit. Actual275.451wall/292.072CPU seconds, peakRSS1,535,098,880bytes, paid spend$0; all1hwall/5000absoluteCPU/3GiBRSS/32MiBprivate bounds held.

## Results on every registered case

All rates below compare the highest-scoring next action with the correct action while supplying correct previous actions. Denominators remain fixed:1,246 first-binding actions and17,526 reuse actions at each checkpoint. No action in either group had only one legal choice.

| Correct-history measure | Untrained | 1,000 updates |
| --- | ---: | ---: |
| Correct first-binding actions | 60/1246 (4.82%) | 192/1246 (15.41%) |
| Correct reuse actions | 1619/17526 (9.24%) | 14648/17526 (83.58%) |
| First-binding loss per action, nats | 3.373680 | 2.957626 |
| Reuse loss per action, nats | 2.595397 | 0.415085 |
| Mean whole-path loss, nats | 776.414682 | 171.249600 |
| Exact free action paths | 0/64 | 0/64 |
| First free error creates a wrong binding | 63/64 | 64/64 |

**98.6615% of the total guided loss reduction occurs in reuse actions.** This is an accounting decomposition of the loss, not causal attribution to an internal mechanism. The much larger reuse denominator matters; both group-specific average losses are shown.

At1000 updates,59/64 free readings make their first error at action zero; the remaining five make one correct action first. Every first error creates a wrong dictionary binding. The failure begins before a long wrong history exists. Revising only a late suffix would leave the earliest erroneous binding fixed.

## Neural memory input sensitivity

Each entry below is the paired whole-path loss change from its own base case. We report mean absolute change as well as signed change, so cancellation cannot hide large effects.

| Checkpoint / intervention | Mean change, nats | Mean absolute change | Minimum / maximum | Cases with worse loss |
| --- | ---: | ---: | ---: | ---: |
| 0 / erase | +6.978184 | 7.605583 | −7.541647 / +20.630332 | 57/64 |
| 0 / rotate | −0.595497 | 5.587463 | −21.485839 / +19.872689 | 26/64 |
| 1000 / erase | +1.268303 | 2.275603 | −6.765569 / +9.230306 | 46/64 |
| 1000 / rotate | −0.019590 | 0.302736 | −0.768671 / +1.241172 | 31/64 |

At each checkpoint, erase changes18,708/18,772 query packets and321,298 slots; rotate changes18,639 queries and314,145 slots. Rotation can be a no-op with fewer than two occupied rows or identical occupied values. Sham changes zero slots and its logits match base exactly. At1000, erase gives189 correct first bindings/14626 correct reuse actions; rotation gives190/14643 versus base192/14648.

The registered explicit memory channel shows little loss sensitivity to reassignment at1000. **This does not establish that the model ignores dictionary information:** legal masks and the correct action history still encode bindings. The changed feature packet need not be a valid symbolic state and can be outside the training distribution. We have not identified the redundant route, intervened on an internal neuron/path, measured free behavior under these interventions, or replicated with another seed.

## Validation and archive

All128 base CPUfloat32 path scores match saved MPS scores within the registered.01-nat bound: maximum1.320320e−5 at0 and1.467197e−5 at1000. Before and at the first greedy divergence, the saved greedy choice has zero logit deficit under the causal reference. Every legal score is finite, illegal masks match exactly, and sham is bit-exact.

One bounded read-only closure independently sums the saved per-action probabilities/correctness flags, reconstructs all legal counts and symbolic first errors, checks every per-case/group/whole summary, reproduces changed-query/slot counts from true binding states, and verifies all512 forward phases and81 input hashes. It rehashes the two weight archives without new model calls; the actual diagnostic itself checked their finite model identity. This is a same-researcher closure, not an independent expert audit. Its source is retained privately with a digest in the receipt. No repeated intervention or newly generated neural scores.

Before the actual diagnostic,2968 tests plus23 subtests passed,13 skipped,238.73s. Final metadata extensions subsequently passed the eight focused tests and scoped lint. The five previously recorded unrelated whole-tree lint findings remain. Private per-action arrays, trace, checkpoint archives and plaintext sources stay ignored; compact case tables and hash receipts are tracked.

## Consequence for the next design

Prioritize inferring the first mapping from the entire observation and making coherent revisions that can undo early bindings. A lower guided loss or a larger model alone is not the evidence we need. Source-informed correction still needs admission of the actual stochastic proposal density and source score; correct joint-MH theory by itself does not demonstrate usable coverage or mixing. Later checkpoints and the binding-off/second-seed comparisons retain their original training plan.

The unchanged training run subsequently reached4000 updates: mean guided loss162.795802, still4/64 complete readings, zero exact records/128 and zero complete used keys;60 failures. This is a separate intermediate result, checked by [the4000 read-only receipt](../../results/SOURCE-ACTION-TRAIN-002/step4000-readonly-check001.json), not an extension of this0/1000 intervention panel. The planned full campaign audit remains pending. **Voynich remains unsolved.**
