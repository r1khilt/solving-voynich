# CAMPAIGN-0001 results — Three completed parallel experiments

Completed 2026-09-21. All three jobs finished normally from clean, published source `72f632804a162009b488e6de5d8550493670f937`. The full campaign took **3,277.179 seconds /54.62 minutes elapsed**, training **25 new models for41,200 updates**. No Voynich word, filler assignment or historical encoding rule was recovered.

## What we learned

| Track | Observation | Interpretation |
| --- | --- | --- |
| [Blind recovery](EXP-0008-results.md) | Bigger models improved some familiar synthetic keys; no unfamiliar cycle/branch key passed the stronger recovery criterion. Every omitted-family RRXOR partition selected one cluster. | Label-free fitting works as a benchmark, but this method does not reliably recover unfamiliar encodings. Output-only clusters generally match or outperform hidden-vector clusters. |
| [Causal mapping](EXP-0009-results.md) | Earlier-component replacements changed later predictions in all three trained toy models beyond matched controls. The untrained control and all three Voynich models failed. | We have stronger causal calibration on a known artificial process. Broad replacements do not isolate a state-only variable, transition algorithm or shared manuscript mechanism. |
| [Longer context](EXP-0010-results.md) | None of four registered longer-context contrasts passed. Main2048 lost0.034861bits/unit versus main256 on average; compact2048 lost0.030688. | More preceding text was not a reliable improvement under this fixed training budget. This is not evidence that the manuscript lacks long-range dependencies. |

There are informative caveats. The copying control mostly recovered the next copied symbol, with poor later predictions. One-cluster random/XOR partitions exactly reproduced a unigram estimate: their small gain over last-symbol counts was simpler estimation, not hidden-rule recovery. The 2048-unit models used additional context relative to their own truncated inputs, yet lost to separately trained256-unit controls. Hiding line/record-boundary identity added0.035553bits overall; glyph-only effects were not positive in every seed. The full2048-prefix subgroup contains only four pages on two physical leaves.

These results narrow the next question: can we infer explicit, reusable updates that survive unfamiliar keys and predict joint future sequences? [Predictive-rule source notes](../research/PREDICTIVE_RULES.md) and an [explicit HMM comparator review](../research/BELIEF_NET_REVIEW.md) describe unimplemented candidates. They were researched after registration and did not alter any running method or threshold. All historical manuscript hypotheses remain unresolved; no new run is scheduled.

## Work and resource accounting

| Track | Models / work | Measured track wall time |
| --- | --- | ---: |
| EXP-0008 | Four synthetic LMs,16,000updates,131,072,000 sampled targets;160 blind partitions across20datasets |1,320.414s|
| EXP-0009 | Seven frozen models;48-site discovery/confirmation maps and future controls; zero LM updates |240.840s|
| EXP-0010 |21Voynich LMs,25,200updates,88,088,063 sampled targets |3,273.273s|

The **219,160,063 sampled targets** include repeated training examples. They do not represent additional independent manuscript evidence. Track durations overlap and must not be added as campaign elapsed time or isolated speed benchmarks. Local PyTorch2.14/MPS, two CPU threads per worker, finite deadlines and allocator limits were used. No paid research API or cloud training was used.

The supervisor's largest sampled sum of process RSS was **1,686,142,976bytes /1.5703GiB**. This is a conservative process-residency measure with possible shared-page double counting, not total GPU or physical memory. The largest recorded EXP-0010 Metal driver sample was about5.757GiB, neither additive with RSS nor a proven whole-run peak. No resource sampling errors, deadline stops or allocator failures occurred. Artificial-input preflight measurements are archived separately and explicitly labeled as dirty-implementation hardware probes.

## Integrity and artifacts

- Source remained clean and unchanged until every job exited; root captured completion integrity before documentation/archival edits. [Campaign manifest](../../results/CAMPAIGN-0001/manifest.json), [final status](../../results/CAMPAIGN-0001/status.json), [completion audit](../../results/CAMPAIGN-0001/completion_integrity.json) and [resource samples](../../results/CAMPAIGN-0001/resource_samples.json).
- EXP-0008 froze320 analysis artifacts before true-state diagnostics;83 dataset files and four selected checkpoints were verified. All32 held-out/training key pairs differ in observable bigram probabilities. An independent second review agreed with the negative transfer interpretation.
- EXP-0009's archival audit recomputed raw-score aggregates, discovery selections, controls and threshold decisions, and checked source/data/checkpoint/report hashes. Numerical identity/restoration and future-invariance errors were0.0.
- EXP-0010's archive and independent audit rederived all512 coordinates and exact sampled target streams, checked matching initial weights and all63 checkpoint payloads/hashes, and preserved every seed/condition. No final manuscript test scoring occurred.
- Final software checks: **295 tests plus23subtests passed**; Ruff passed. The figure was generated from archived measurements and visually inspected. Link/JSON/SVG/charter/whitespace and repository-size checks accompany publication.

Compact numerical results live in `results/EXP-0008/`, `results/EXP-0009/` and `results/EXP-0010/`; weights, raw/derived text, large arrays and original logs remain ignored with recorded paths/digests. [Scientific overview](../../results/CAMPAIGN-0001/overview.png), [plain-English guide](../PROGRESS_EXPLAINED.md), [current handoff](../CURRENT_STATUS.md), [original registration](CAMPAIGN-0001.md).
