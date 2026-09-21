# Completed episodic campaign, 2026-09-21

Publication aliases **EPISODIC-0012 / EPISODIC-0013** distinguish this campaign from the separate latent-recovery experiments with the same original numbers. The `EXP-0012/` and `EXP-0013/` subdirectories preserve historical artifact identifiers. Do not merge their contents into top-level `results/EXP-0012/` or `results/EXP-0013/`.

- [Prediction and explicit-model results](../../docs/experiments/EPISODIC-0012-results.md).
- [Causal results](../../docs/experiments/EPISODIC-0013-results.md).
- [Measured overview](EXP-0012/overview.png); SVG/PDF siblings available.
- [Recovery and resource summary](provenance/resource_and_restart_summary.json).

All scientific phases ran from `c9c95a0cfef9bdfc542136e4d38482e7d37c59ea`, unchanged through completion. Each experiment's `archive_manifest.json` is the frozen archiver's original digest inventory. `publication_manifest.json` additionally covers the final figure and recovery supplement; it does not replace those original manifests. Reported elapsed time begins with the original training supervisor and excludes earlier data preparation/pilots. Termination verification and unopened-confirmation status are recorded operator assertions; unavailable original process exit codes are not reconstructed.

The host interrupted the original supervisor. Recovery preserved 13 completed runs and two partial attempts; their manifests/initialization match the replays exactly. Both completed replay runs contribute once to reported results. The partial attempts recorded at least 1,200 discarded updates /14,745,600 targets; exact terminal work is unknown. Last original sample to recovery start was 255.629525 seconds. No continuous-monitoring claim is made. The resource logs show sampled process RSS, not physical-memory peaks.

`provenance/recovery_supervisor_source.py` is the **exact historical recovery script**, preserved byte-for-byte with its launch-time SHA in `supervision_recovery.json`. It is not a generic launcher; its hardcoded paths and one-time state transitions must not be used on a populated campaign. See the [runbook](../../docs/RUNBOOK.md) for reproduction.

Weights, synthetic arrays and full local outputs remain ignored, with hashes and source provenance retained here. They are not backed up by this Git archive. No paid API or manuscript input was used; neither primary causal claim was established and no decipherment follows.
