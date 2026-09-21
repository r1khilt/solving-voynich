# Agent handoff and current status

Updated 2026-09-21. **The episodic campaign completed; neither primary causal claim passed.** No manuscript word, language, filler assignment or historical encoding rule was established. This work did not score the final manuscript holdout.

The audit fixes and completed results are integrated on this branch. Final combined validation: **471 tests plus 23 subtests passed in 8.05s**, including MPS, and full source/test/script lint passed. Archive hashes, both notebook histories and local links were checked. Notebook NB-INTEGRATION-20260921 records publication attempts; verify the remote ref rather than treating a local commit as backup.

## Latest results

The [prediction study](experiments/EPISODIC-0012-results.md) trained 21 models across seven conditions and three seeds. Many varied artificial processes improved the raw transformer over 32 fixed processes: −0.127945 bits on new tasks from familiar families and −0.038903 on excluded families. Both passed the frozen practical rule. Canonicalizing symbol names nearly eliminated the fixed/fresh difference; parameter diversity alone is not established as the cause.

Scaling the canonical transformer from 641,152 to 10,625,664 parameters improved familiar-family prediction by only 0.010510 bits, below the 0.02 threshold, with no established excluded-family gain. The signed recurrence was worse than the GRU under this schedule. Explicit HMMs performed well overall on 12 matched tasks with **5,120 extra adaptation symbols per task**, but selected one state on every parity/XOR task, failing to recover those rules.

The [causal study](experiments/EPISODIC-0013-results.md) ran 12 model/rank audits and 12 untrained controls. GRU rank-4 patches reduced donor-future discrepancy by 55.8–67.1% and beat all six registered controls, but changed immediate answers beyond tolerance. Every recurrent teacher exceeded the oracle-error ceiling. Delayed-Fisher directions preserved immediate answers better while moving future predictions; temporal consistency and teacher quality still limit interpretation. These secondary findings require new final pools.

See the [plain-English guide](PROGRESS_EXPLAINED.md), [figure](../results/episodic-20260921/EXP-0012/overview.png) and notebook NB-EPISODIC-20260921-B.

## Source, isolation and resources

- Scientific source **`c9c95a0cfef9bdfc542136e4d38482e7d37c59ea`** was pushed before execution and unchanged through completion. Prelaunch: 421 tests plus 23 subtests with MPS. Publication edits are later.
- Branch/worktree: `codex/episodic-rule-recovery`, `/Users/rikhil/.codex/worktrees/episodic-rule-recovery/solving-voynich`. Remote: `https://github.com/r1khilt/solving-voynich`. Shared original checkout has concurrent work; preserve it.
- **ID collision:** c9 used EXP-0012/0013; public aliases are EPISODIC-0012/0013, archives `results/episodic-20260921/`. Historical JSON/config/output IDs stay unchanged. Main's separate EXP-0012 null-removal and EXP-0013 CTC records must remain intact.
- All campaign workers completed. A host interruption lost supervision; two known orphan workers were terminated, 13 completed runs retained, and two preserved partial runs replayed with identical manifests/initialization before confirmation. Monitoring has a gap; original exit codes and exact discarded updates are unknown. [Recovery records](../results/episodic-20260921/provenance/resource_and_restart_summary.json).
- Wall time from original launch through analysis: **93.55 minutes**, including recovery. Completed-run durations sum to 144.30 minutes because CPU/GPU work overlapped. Completed work: 25,200 updates /309,657,600 scored targets. Discarded attempts recorded at least another 1,200 updates; actual extra work is unknown.
- Peak sampled summed RSS: 1,954,512,896 bytes (1.82 GiB); maximum reported Metal driver allocation: 4,441,456,640 bytes (4.14 GiB). These overlap and are not total physical-memory peaks. No paid API or manuscript input; conversation credit usage unmeasured.
- Bulk originals/weights remain ignored under `outputs/EXP-0012/`, not backed up by Git. Compact results, hashes and provenance are tracked. [Runbook](RUNBOOK.md).

## Prior work and continuation

The [deep review](research/deep-review-2026-09-21/README.md) contains 93 source records /92 works with reading depth and uncertainty. Fresh-task, explicit-transition and joint-future proposals are now partly implemented and tested. [Implementation map](research/EPISODIC_IMPLEMENTATION.md).

Earlier [CAMPAIGN-0001](experiments/CAMPAIGN-0001-results.md) completed blind recovery, broad causal calibration and longer-context tests. Transfer/context failures and synthetic-versus-manuscript distinctions remain relevant. Corpus: official ZL3b, frozen 177/24/25 physical-group split, 112-entry vocabulary; [data record](research/DATA.md).

[External-audit fixes](research/EXTERNAL_AUDIT_2026-09-21.md) from pushed checkpoint `aa502e9` are integrated here: alignment-aware recovery diagnostics, optional normalized rank proxy with matched controls, and opt-in reproducible random-offset training. Legacy gates/defaults remain unchanged. Integration retains latent-recovery history through `3e956c8` (EXP-0011a/0012/0013 failures and EXP-0015 registration); main has later concurrent work outside this snapshot. Episodic IDs are namespaced; neither history overwrites the other.

Next experiments should separate parameter diversity from permutation coverage and qualify parity prediction before stronger causal explanations. Alternative unitization/random-offset training need registered comparisons; implementation alone establishes no quality gain. All campaign final pools are exposed. New adaptive work requires fresh final keys/contexts and frozen rules. No successor is scheduled.

Read `AGENTS.md`, `MEMORY.md`, latest notebook and the [protocol](research/PROTOCOL.md) first. Standing authorization covers bounded research, delegation, commits/pushes and useful compute. Preserve charter, old gates, held-out manuscript data and concurrent changes.
