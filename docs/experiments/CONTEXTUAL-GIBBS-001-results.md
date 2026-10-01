# CONTEXTUAL-GIBBS-001 results: recovery FAIL, replay PASS, cap-handling deviation

Neither the frozen mapping-improvement gate nor stable-calibration gate passed. No final bank contains the complete used generator key. [Registration](CONTEXTUAL-GIBBS-001.md), [original result](../../results/CONTEXTUAL-GIBBS-001/result.json), [full replay audit](../../results/CONTEXTUAL-GIBBS-001/audit.json), [cap-classification correction](../../results/CONTEXTUAL-GIBBS-001/cap-classification.json).

Registration720906e0d16943a9f5c0c5e6aff7c7d3bb07bddb was committed/pushed/exactremote verified before ONE16100run and ONE50640fullaudit, both terminal0. FourEXPOSED root cases,23unknown rows,32particles/stride16/two seeds,16predefined calls. No answer-derived prefix, prior fitted bank, reader, source training, neural model, GPU, paid service or manuscript holdout was used. All195frozen files remain unchanged. Replay correctness is separate from protocol adherence and inference accuracy.

| Outcome | 42uniform row-MH updates | One42-code row-Gibbs update |
| --- | ---: | ---: |
| Complete positive calls | 4/8 | 1/8 |
| Extinct populations | 2/8 | 7/8 |
| Cumulative call-work budget exhausted | 2/8 | 0/8 |
| Selected complete used keys | 0/8 | 0/8 |
| Banks with complete used key | 0/8 | 0/8 |
| Selected used-row matches, failures counted0 | 4/156 | 4/156 |
| Actual booked target tables | 311040 | 92416 |
| Summed call wall seconds | 560.141965 | 140.994893 |

Both descriptive match fractions are2.5641%, not a gain of the required10percentage points. The only uniform case with both seeds positive, case2, has log-evidence spread101.58494937nats versus≤2. Gibbs has no case with both seeds positive; undefined spreads are not stable zero. Complete-call maximum incremental weights are.999977589–1; corresponding minimum ESS is approximately1. A positive population is not a calibrated estimate. Full-key diversity includes unused rows and does not establish independent readings.

Nominal target tables match only for complete paired schedules; early deaths and budget limits make realized work unequal. The Gibbs update can change ONErow per stage, while42uniform MH mutations can change many rows. This experiment does not isolate all mechanisms, establish CPU-matched superiority, or refute Gibbs inference generally.

## The cap-handling defect

The two case3uniform calls were originally labeled graph_cap. Their cumulative edge counters are EXACTLY2,000,000,001: each hit its2Bper-call work budget, including the first rejected edge. The native wrapper lowers its per-record edge limit to the remaining cumulative budget; both failures therefore reused the native message “Exact lattice edge cap; no pruning.” The runner classified only the message and treated the failures as allowed per-record graph caps. It continued the remaining three predefined calls after the first such failure, contrary to the registered global-work-stop semantics. The replay reproduced that same behavior.

This is a **protocol deviation**, not evidence that a single graph needed2Medges, or that a language/mapping was impossible. It did not create a likelihood, refill a population, lift a resource limit, retune, or add calls outside the fixed16-call inventory. Recorded campaign work6,285,399,766edges remained below8B, and stage CPU/wall/memory caps held. Nevertheless the study must not be presented as fully protocol-compliant. The raw frozen code/results are preserved; a [static correction utility](../../scripts/classify_contextual_caps001.py) binds them without rerunning inference.

A [separate strict adapter](../../src/voynich/strict_censored_bridge.py) promotes cumulative exhaustion to an explicit work-budget exception for FUTURE registrations. Four actual-native tests check per-record versus cumulative/equal limits and byte-identical uncapped scores/counters. It is not retroactively inserted into this closed experiment, and no replacement campaign is queued.

## Verification and resources

ALL16RNG/native trajectories, statuses, dictionaries, weights, mutations, counters, hashes, source/array/library identities and decisions replay exactly. Every32final particle from each of5complete calls gets an independent contextual Python-DP score:160checks, maximum discrepancy5.00222085975e−12nats versus1e−7. Intermediate replay shares the native backend, and the audit is same-author, not independent expert review. No reference cap occurred. A passing replay does not erase the work-cap classification defect or make failed recovery successful.

| Resource | Run | Full audit |
| --- | ---: | ---: |
| Wall seconds | 702.420478 | 710.155291 |
| CPU seconds | 700.870557 | 708.601495 |
| Peak RSS bytes | 843350016 | 1244725248 |
| Paid spend | $0 | $0 |

The run's full compressed banks/traces total43141ignored bytes; compact metrics/manifests are tracked. Native calls787517/cumulative nodes2,403,099,954/edge operations6,285,399,766; cumulative counts repeat states across evaluations, not distinct model states. Maximum nodes in a single native record computation36400, far below300k. Maximum owned-work envelope78,277,888bytes excludes399,571,824pinned source bytes. Thus the case3failure is cumulative repeated scoring work, not a measured large single lattice. Source context helps candidate evaluation but does not automatically make large inference cheap.

## Read-only diagnosis and next question

[Static diagnostic](../../scripts/summarize_contextual_gibbs001.py) and [receipt](../../results/CONTEXTUAL-GIBBS-001/post-outcome.json) bind closed banks without new model calls. All9extinctions occur before EOS closure, within the first0–3completed stages, at cuts no later than(32,32). Both methods share one initialization failure before mutation, case1seed81402. Since source probabilities are strictly positive, loss of finite target support at these stages means the sampled code inventories cannot encode the new prefixes. Language weighting may influence earlier survival; these records do not isolate its causal contribution or prove global target impossibility.

For the5selected positive mappings, oracle maximum used-row matches after ANYrow permutation are13/19,13/18,17/21,17/21,17/21, versus actual0,0,4,0,4. Minimum code replacements before a permutation could match all used generator rows are6,5,4,4,4. NONEof160final particles has an inventory compatible with the complete used generator key. These are combinatorial oracle ceilings, not achieved/scored improved mappings.4536tiny exhaustive permutation checks independently verify the ceiling formula; intermediate support and probability barriers are not included.

[Own support-orbit mathematics](../research/dictionary-permutation-orbits-2026-10-01.md) explains why row swaps can coordinate label changes while preserving encoding possibility, and verifies a separate inventory barrier that swaps plus single replacements cannot cross. Next work must address both inventory and labeling, and early observation-support loss, with exact proposal corrections and new bounded controls. The experiment does not establish a competent decoder, neural circuit, historical language, semantic reading or Voynich decipherment. Goal ACTIVE.
