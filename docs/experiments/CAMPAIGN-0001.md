# CAMPAIGN-0001 — Three parallel local research tracks

Registered 2026-09-21 UTC. The user explicitly authorized all three proposals in parallel and agent delegation. This document coordinates execution; each scientific question, source review, controls and decision criteria lives in its experiment registration.

| Track | Experiment | Work |
| --- | --- | --- |
| Blind recovery | EXP-0008 | Infer predictive states from visible synthetic text/activations without using latent truth for fitting or selection; evaluate unseen keys and a new family. |
| Causal mapping | EXP-0009 | Map interventions across components and positions; separate discovery from confirmation and test consequences across multiple later predictions. |
| Longer context | EXP-0010 | Compare 256/512/1024/2048-unit contexts with matched training exposure and common validation targets, plus compact and locus-marker controls. |

## Resource policy

- Three worker agents implement independently in nonoverlapping files; the root agent owns shared notebook/memory, source review integration, Git and orchestration. No paid API, cloud compute or manuscript final-test scoring.
- Scientific source and registrations must be committed and pushed before real runs. All intermediate data, weights, logs and reports remain in ignored `outputs/` / `data/processed/` while any track runs, so new run manifests can continue to record clean source. Record implementation corrections honestly and restart affected experiments under a new revision if needed; do not hide dirty-source runs.
- The supervisor starts three process groups concurrently; each track runs its own subjobs sequentially. Shared-GPU contention is expected and recorded. Parallel launch is not a claim of threefold speedup or independent hardware.
- Per-process environment sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.23` and low watermark0.18, with two CPU threads. At the measured Metal recommendation of51.84GiB, each MPS allocator has a ceiling near11.92GiB, totaling35.77GiB across three workers before nonallocator/CPU allocations. This is an allocator constraint, not a total-system-memory guarantee. See [PyTorch 2.14 MPS environment variables](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html), checked2026-09-21 UTC.
- Supervisor samples process-group RSS every five seconds and stops its own remaining jobs if the conservative sum exceeds48GiB. Shared pages may be counted more than once. Never add this number to MPS driver allocation and call it physical memory use. Each track should separately report MPS current/driver samples, measured runtime and CPU RSS. API scope: [PyTorch driver memory](https://docs.pytorch.org/docs/2.14/generated/torch.mps.driver_allocated_memory.html).
- Campaign wall-clock ceiling:8hours. Blind/context scientific loops at most3hours each; causal mapping at most2hours. The supervisor permits60seconds beyond each internal deadline for final checkpoint/status writes before scoped termination. These are limits, not minimum utilization targets or promises of completion. Preserve checkpoints and distinguish completed, deadline-stopped, failed and unstarted work. The supervisor terminates only process groups it started; no broad process cleanup or persistent OS setting changes.
- Failure of one track does not silently cancel independent work. Logs and status are written per track, with an atomic campaign status file. A numerical/integrity failure is investigated before repeating that experiment; no test-informed retuning disguised as the original registration.

## Preflight measurements

EXP-0010's artificial-input benchmark used four2048-position source blocks per optimizer update (8192positions), gradient accumulation with2048-token microbatches, five warmups and ten timed updates. It measured full optimizer updates, not only one microbatch. Main1.814M-model times:256=.063685s;512=.085208s;1024=.116956s;2048=.178713s. Compact430720-model times:256=.028180s;2048=.084705s.

The 21-run,1200-update proposal therefore has a **44.17minute training-only estimate** before validation, startup, shorter/padded-page effects or shared-GPU contention. Benchmark inputs were random integers, not manuscript data. Its working tree was intentionally still under implementation; it is a hardware probe and does not supply scientific prediction results. Exact measurement/environment report stays in `outputs/EXP-0010/benchmark.json` until archival. MPS driver statistics are sampled rather than measured peaks.

The blind-recovery larger model has3,220,224parameters. Its artificial-input benchmark measured0.071974seconds/update at batch32/context256 (20timed updates after3warmups), process RSS467,402,752bytes and sampled MPS driver allocation1,286,307,840bytes. This is a brief throughput probe with a simpler loss loop than the full experiment, not a utilization or generalization result. Its stdout was saved at `/private/tmp/voynich-exp0008-benchmark.log`; the notebook records the exact command. No benchmark input was taken from the manuscript.

The agents' first implementation turns were interrupted by a reported usage limit. The user requested continuation; the saved files were resumed, independently reviewed and completed before the scientific run. No research job was running at recovery. A transient pre-final lint pass found an unused import and a lambda-style violation; both were corrected before publication, with no research outcomes available.

## Execution and interpretation

`scripts/run_parallel_campaign.py` consumes the frozen command arrays and finite limits in `configs/campaign-0001.json`. Without `--execute` it prints the plan. Execution refuses a dirty working tree and a nonempty campaign output directory. Each track also maintains its experiment-specific overwrite/provenance rules.

```sh
.venv/bin/python scripts/run_parallel_campaign.py --execute
```

Supervisor tests use only short artificial subprocesses: simultaneous starts, isolated logs, independent-job failure, hard deadlines, scoped memory stopping and invalid-plan rejection. No training is performed by these tests.

Final reports must explain outcomes in ordinary language, distinguish artificial-generator scores from manuscript findings, retain failed controls and identify remaining uncertainty. Cleanly compare all specified conditions rather than publishing only the best result. Update the notebook and GitHub at coherent registration/execution/reporting checkpoints. This campaign does not itself claim cipher recovery or translation.
