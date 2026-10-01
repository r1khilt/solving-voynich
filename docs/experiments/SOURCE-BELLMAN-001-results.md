# SOURCE-BELLMAN-001 — one-step lookahead misses its recovery criterion

2026-10-01. **Exploratory signal NOT SUPPORTED.** The one-step guide reduced total reading errors from 788 to 768 across 1152 known source letters, a 2.538% reduction versus the prospectively required 10%. Two cases improved, one deteriorated, and the already solved case stayed solved. Summed search time increased from 113.198 to 446.630 seconds, about 3.946 times. No additional passage or used dictionary was recovered exactly. These are artificial, previously exposed ciphers; there is no new Voynich reading.

[Frozen registration](SOURCE-BELLMAN-001.md), [methods and primary-source reading depths](../research/source-bellman-2026-10-01.md), [closed result](../../results/SOURCE-BELLMAN-001/result.json), [full replay audit](../../results/SOURCE-BELLMAN-001/audit.json), [publication checks](../../results/SOURCE-BELLMAN-001/publication-audit.json).

## Paired observations

Each case contains two passages. Errors are Levenshtein insertions/deletions/substitutions against the known source, summed across both passages; they are not a percentage of independently classified characters.

| Case | Source letters | IID errors | One-step errors | IID correct used rows | One-step correct used rows | IID search seconds | One-step search seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 |128|0|0|19/19|19/19|19.777|58.970|
| 1 |128|99|95|3/18|3/18|14.106|50.232|
| 2 |448|313|290|6/21|7/21|45.382|180.763|
| 3 |448|376|383|1/20|3/20|33.932|156.666|
| Total |1152|788|768|29/78|32/78|113.198|446.630|

Both arms returned all eight passages; both recovered 2/8 exactly, from case 0 alone. Both returned the correct used dictionary in 1/4 cases. The registered all-readings and at-least-two-case-improvements clauses passed, but the aggregate 10% clause failed. No criterion was revised after observing results.

All eight searches exhausted their remaining frontiers and all eight are incomplete because states were pruned. Frontier exhaustion is not exhaustive enumeration or an optimality certificate. The lookahead still builds a heuristic tail; it changes ranking without changing any actual source edge, prior, stopping probability or reader.

## What the extra work bought

IID built 10,135,933 tail tables; one-step built 39,964,364 tables and evaluated 31,008,285 child actions in 836,737 backup calls. These obey the fixed early-layer bounds. Identical beam/state limits therefore did not mean identical CPU work. No matched-compute advantage is claimed.

The selected full-key source log mass was unchanged in case 0. One-step minus IID differences were +6.246 nats in case 1, −50.666 nats in case 2, and −98.717 nats in case 3. Case 2's reading became closer to the known truth while its selected key received a lower source score. Reading error and selected-key objective are different measures; different pruned frontiers need not improve either monotonically.

The new search visited 9993/8319/219/44 terminal groups across cases 0/1/2/3. Cases 2 and 3 returned every visited group, so absence of their correct used dictionary is not caused by the 512-group output limit. Case 1 returned only 512 of 8319: its reported absence applies only to returned groups. No gold-path observer was attached to this experiment, so these records do not establish when literal truth paths disappeared or whether lookahead delayed their loss.

## Execution, checks and limits

Registration `db6cbc0e7bb86bf4a92ca5b05592acae95a67614` was pushed and exact remote identity verified before the single run. One run and one complete audit finished with exit 0; no restart, extension, retuning, new model training, GPU use, paid API call or manuscript holdout access occurred. All 140 frozen paths remained unchanged.

The audit exactly replayed all eight fitted outputs, traces, guide/action/table counts, terminal masses, selected keys and predictions. Depth0 exactly reproduced all four prior wide fits. Sixteen alternate Python full-key record scores agreed within 5.684341886080801e−13 against the registered 1e−7 tolerance. This is same-author/same-native-implementation full search replay, supplemented by alternate scalar scoring and independent tiny rational arithmetic; it is not an independent full solver review.

| Stage | Wall seconds | CPU seconds | Peak sampled host RSS bytes |
| --- | ---: | ---: | ---: |
| Run |560.763932667|559.308258|896614400|
| Complete replay audit |553.466569291|552.695990|883261440|

Both stages stayed below 2400 wall-seconds / 2200 CPU-seconds / 2 GiB host memory caps. Ignored compressed fitted/prediction archives total 225410 bytes, below 256 MiB; compact receipts are tracked. Python 3.12.13 / NumPy 2.5.3, BLAS threads 1. Input/source/compiler/library hashes are in the registration, build manifest and result. Source counts, actual source models, prior neural checkpoints and all old experiment laws stayed unchanged.

Before registration, 13 new tests and the full 2675 tests + 23 subtests passed, with 13 skips; scoped lint passed and five unrelated pre-existing full-tree lint findings remained. Post-outcome publication checks independently reconstructed all 16 reading errors/literal encodings, cipher-only key fills, used-row totals and the decision gate without another source/search/model call. An initial utility invocation omitted the documented PYTHONPATH and failed at import; the corrected invocation only checked closed artifacts, not another empirical run or audit.

Result: 5442 bytes, SHA 331a3a267205e7b56736fa093cf089898b1ac84c80e1442231f16a3c046f53c6; audit: 864 bytes, SHA f9ce6f027c0ee408366844b3fc13d072aa61a2b00e774e7576bdc181d424912c.

## Next inference question

This experiment does not justify buying deeper lookahead by brute force. [Post-outcome exact moment analysis](../research/after-bellman-2026-10-01.md) isolates the correlations created by one fixed unknown dictionary and shows why even pairwise corrections are not generally sufficient. That is a mathematical design constraint, not an observed whole-text recovery improvement. A new guide must earn its own tiny-law, resource, exposed-development and subsequent disjoint qualification tests before historical use. Voynich remains unsolved.
