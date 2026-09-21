# EPISODIC-0012 results — Fresh-task prediction and explicit inference

Completed 2026-09-21. Training on many fresh synthetic processes improved the raw-symbol transformer over training on 32 fixed processes. The larger transformer produced a smaller improvement that did not meet the registered practical threshold. Neither result establishes recovery of a generating rule or any manuscript meaning.

This study was registered as EXP-0012 on the isolated episodic branch. The `EPISODIC-0012` document name and `results/episodic-20260921/` namespace distinguish it from the separate null-removal experiment assigned the same original number. Frozen JSON retains its original identifiers. See the [registration](EPISODIC-0012.md), [confirmation report](../../results/episodic-20260921/EXP-0012/confirmation_report.json), and [archive summary](../../results/episodic-20260921/EXP-0012/summary.json).

## Design and execution

All scientific phases used unchanged source `c9c95a0cfef9bdfc542136e4d38482e7d37c59ea`. Seven conditions, each with seeds 121/122/123, trained for a maximum of 1,200 updates at batch 128 and context 128. Each training regime supplied 153,600 streams of 129 symbols: fresh training used a distinct parameter seed per stream; fixed training reused 32 processes. The data seed was 12120000. Inputs were four-symbol simulations, never manuscript text, plaintext or hidden-state labels. Exact configuration, split identities and hashes are in the [data manifest](../../results/episodic-20260921/EXP-0012/data_manifest.json).

Development loss selected among checkpoints at 400, 800 and 1,200 updates. All raw-fixed models selected step 800; every other model selected 1,200. Thus conditions shared the **maximum training budget and selection procedure**, but their selected checkpoints did not receive equal exposure. All 21 completed runs together scored 309,657,600 training targets, including repeated exposure across conditions/seeds and excluding aborted partial runs.

The host interrupted supervision during training. Recovery retained 13 completed runs and replayed two partial runs with unchanged configuration and seeds before opening confirmation. Total elapsed time from the original launch was **93.548 minutes**, including the interruption and recovery; this was not uninterrupted supervision. The recovered training supervision interval was 56.37 minutes and analysis took 5.63 minutes. Completed runs sum to 144.30 minutes because workers overlapped; signed recurrence accounts for 63.92 minutes and the large transformers 47.67 minutes. The last original resource sample preceded recovery launch by **255.63 seconds (4.26 minutes)**; the exact worker-termination time and extra partial-run update count were not recorded. No recorded resource-limit or numerical stop occurred. Source guards and the frozen archiver passed. See [completion provenance](../../results/episodic-20260921/EXP-0012/complete.json) and the [resource and restart accounting](../../results/episodic-20260921/provenance/resource_and_restart_summary.json). No paid API was used.

## Prediction results

“Familiar” means new parameters from the four training families: cycle, branch, pair parity and IID. “Excluded” means new tasks from two families absent from training: third-bit XOR and switching regimes. Confirmation contains 32 familiar and 16 excluded tasks, eight streams each. Scores average within task and across three model seeds; lower is better. Bits per symbol measure prediction error after ignoring the first 32 next-symbol targets in each stream.

| Condition | Parameters | Familiar bits | Excluded bits | Beyond-first KL, all 48 tasks |
| --- | ---: | ---: | ---: | ---: |
| Raw fixed transformer | 641,152 | 1.674412 | 1.920766 | 0.507727 |
| Raw fresh transformer | 641,152 | 1.546466 | 1.881863 | 0.304321 |
| Canonical fixed transformer | 641,152 | 1.551479 | 1.878762 | 0.312167 |
| Canonical fresh transformer | 641,152 | 1.547808 | 1.880624 | 0.304600 |
| Large canonical fresh transformer | 10,625,664 | 1.537298 | 1.880770 | 0.277184 |
| Fresh GRU | 792,068 | 1.548922 | 1.879292 | 0.311308 |
| Fresh signed recurrence | 399,882 | 1.695625 | 1.902135 | 0.516650 |

Beyond-first KL measures error in the three-symbol future distribution beyond its first-symbol marginal, using two fixed 64-symbol prefixes per task. It checks future dependence, not just the next answer; it does not identify a unique hidden machine. The generator oracle knows the true parameters, whereas a learner must infer them from its finite prefix.

A registered gain required mean difference below −0.02 bits and the upper 95% paired task-bootstrap bound below zero. Three seeds were averaged within each task before 2,000 resamples. These are descriptive intervals without multiplicity correction.

| Left minus right | Familiar difference [95% interval] | Excluded difference [95% interval] |
| --- | ---: | ---: |
| Raw fresh − raw fixed | **−0.127945 [−0.190717, −0.073080]** | **−0.038903 [−0.058053, −0.021321]** |
| Canonical fresh − canonical fixed | −0.003671 [−0.007739, 0.000018] | 0.001862 [−0.000874, 0.004605] |
| Canonical fresh − raw fresh | 0.001342 [−0.001040, 0.003830] | −0.001239 [−0.004206, 0.001687] |
| Large − small canonical fresh | −0.010510 [−0.014588, −0.006817] | 0.000145 [−0.001630, 0.001873] |
| GRU − raw fresh transformer | 0.002456 [−0.000402, 0.005262] | −0.002571 [−0.004968, −0.000169] |
| Signed recurrence − GRU | 0.146703 [0.108537, 0.188133] | 0.022843 [0.014284, 0.031335] |

Only the bold pooled comparisons meet both requirements. Raw fresh–fixed changes both process-parameter diversity and symbol-permutation coverage, so it does not isolate which caused the gain. Canonical fresh–fixed removes arbitrary naming variation and shows little difference. The large model's familiar-family improvement is detectable but smaller than the frozen practical threshold; increased compute did not establish excluded-family improvement. The simplified signed recurrence performed worse under this schedule; this is not a verdict on the DeltaProduct paper or all recurrent architectures.

Online order-0/1/2 baselines scored 1.901561/1.656272/1.697415 familiar bits and 1.907782/1.893770/1.943545 excluded bits. They update counts from each visible prefix; more context can hurt when counts become sparse. Oracle scores were 1.468012 and 1.649987. Substantial gaps remain: the large model scores 1.878731 versus oracle 1.743336 on pair parity, and 1.914940 versus 1.656344 on XOR. See [all baseline task scores](../../results/episodic-20260921/EXP-0012/confirmation_baselines.json).

Canonical renaming sensitivity was approximately 5×10⁻⁸ TV, as expected from the representation's built-in equivariance. Raw fresh sensitivity averaged 0.034388 versus 0.130514 for raw fixed across three registered permutations. Neither exact canonical invariance nor reduced raw sensitivity is evidence of decipherment.

## Explicit models with additional adaptation data

The HMM comparison uses exactly the first two confirmation tasks per family: **12 matched task IDs**, the same scoring streams, warmup and joint prefixes as the neural subset. Each HMM additionally receives 4,096 fitting and 1,024 development symbols for its task. It selects among K=1/2/4/8, two restarts, 25 EM iterations, using development loss plus the frozen complexity penalty; it does not refit on development. This compares adaptation budgets, not architectures given the same information.

| Same 12-task subset | Bits per symbol | Full three-symbol oracle KL |
| --- | ---: | ---: |
| Explicit HMM | 1.619340 | 0.199730 |
| Large canonical transformer | 1.679974 | 0.336096 |
| Raw fresh transformer | 1.683799 | 0.364486 |

The HMM selects K4 for cycle/branch and K2 for switching, but K1 for IID **and every sampled pair-parity/XOR task**. Its overall advantage therefore does not establish parity-rule recovery. On the matched pair-parity subset, HMM joint KL is 0.511439 versus the large transformer's 0.343769. All candidates, fitting histories and adaptation counts remain in the [explicit report](../../results/episodic-20260921/EXP-0012/explicit_report.json).

## Limits and next state

This is a clean, stationary, four-symbol benchmark with few confirmation tasks and short joint horizons. It excludes copy-mutate/history-copy controls, glyph ambiguity, uncertain boundaries and semantic grounding. The equivalent-state-split implementation was property-tested but not exercised as a scientific campaign condition. Large/small and recurrent comparisons are not compute matched. Shared training pools across seeds are not independent corpus replications.

The accompanying [causal study](EPISODIC-0013-results.md) establishes neither primary claim. A useful next experiment would isolate parameter diversity from permutation coverage and improve qualified pair-parity prediction before interpreting transplanted states. These are proposals requiring fresh final pools and new frozen rules. Every confirmation pool in this campaign is now exposed; no manuscript rule or meaning has been recovered.
