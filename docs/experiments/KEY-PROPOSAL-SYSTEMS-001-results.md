# KEY-PROPOSAL-SYSTEMS-001 — both actual model sizes fit and execute locally

**Engineering PASS, not key recovery.** Source
`83135786764a6222c462f8c42007d3da131ce68e` was pushed and its exact remote main
ref verified before one fixed four-arm campaign. All four children exited0,
completed every registered update/proposal/reference, and stayed within caps.
No repeat, extension, missing arm, corpus/panel input or paid API/cloud use.

The large model is an actual **94,981,674-parameter conditional inverse**;
it proposes entire duplicate-allowing23-row dictionaries from several ciphertext
records. It is not a larger source-only predictor. The control has5,423,146
parameters. Architecture, mathematical caveats and source review are in
[the method memo](../research/joint-key-proposal-implementation-2026-09-30.md).
This result establishes feasibility for sustained local training, not competence.

## Actual measured cost

All inputs have four448glyph padded records per episode. Each arm used15random
optimizer updates; means cover the registered final12steps. Driver and host
figures below are **GiB**, not decimalGB, and are sampled/peak-process resources.

| Arm | Parameters | Episodes/update | Seconds/update | Episodes/sec | Sampled Metal driver GiB | Host peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small-b1 | 5,423,146 | 1 | .039316 | 25.4349 | 1.1866 | .7842 |
| small-b4 | 5,423,146 | 4 | .087607 | 45.6586 | 2.2472 | .7811 |
| large-b1 | 94,981,674 | 1 | .287810 | 3.4745 | 3.8814 | 2.1630 |
| large-b4 | 94,981,674 | 4 | .709309 | 5.6393 | 6.9284 | 2.1694 |

At batch4the large architecture costs about8.10times the control's step time
for about17.51times its parameters. This is a resource ratio, not improved
statistical efficiency. All four arms generated8sampled complete keys plus one
greedy complete key after updates. Measured proposal durations respectively
2.778209/.350111/1.919429/1.064603seconds include encoding, both generation paths,
synchronization and trace callbacks. They do not include independent CPU replay.
Fixed process order/system shader caching and only one short measurement per arm
prevent attributing the surprising batch-pair generation times to batch size.
These are not replicated speed comparisons or sustained thermal forecasts.

## Correctness and complete accounting

Separate CPUfloat64 initial/final short-record/all23-row logit checks have maximum
error1.1576821e-5, below5e-5. Full448glyph input/nine complete joint-key log
probabilities have maximum error3.8832303e-6nats, below.002. A separate CPU-only
auditor reloads/checksums both checkpoints of all four arms, verifies finite
float32 state tensors/configuration/exact parameter counts, recomputes logits
and whole-key normalized probability factors, and checks all work/resource traces.
Both batch pairs have identical initial state digests. All stages/23choices
accounted for, all60random optimizer updates retained.

The independent code path is a same-author replay, **not independent-agent
approval**. Tiny exact normalization/causal-label/finite-difference/renaming
controls and actual optimizer/archive integration preceded the GPU run. Full
2,436tests+23subtests passed with13skips; changed Ruff/diff checks passed. Five
unrelated existing full-tree Ruff findings remain. No source change after freeze.

Additional read-only publication/version/arithmetic accounting verifies33unique
file bindings, all12compact runtime records, paired seeds/state digests,
Torch2.14.0/NumPy2.5.3/Python3.12.13, CPUthreads2/disabledMHAfastpath and actual
timing/resource arithmetic. It computes no model scores and does not replace
the one registered auditor. Its exact source is bound in
`publication-accounting.json`. Both complete auditsPASS. Missing partial work
would remain unknown; there are no missing or partial arms in this actual run.

## Resources and retention

Single campaign:40.489854elapsed seconds,18.661154all-childCPU seconds including
child admission/imports. Measured arm times sum36.948968seconds; maximum host
peak2,329,346,048bytes, sampled driver7,439,286,272bytes. Single CPU audit:
9.265766wall/10.249804CPU seconds,2,644,787,200peakRSSbytes. Publication accounting:
1.240173wall/1.125710CPU seconds,605,913,088peakRSSbytes. Audit/accounting timers
have their documented setup boundaries and are not whole user-visible durations.

**60random-data optimizer updates; zero language-training updates; $0paid.**
All eight initial/final checkpoints total1,606,811,496bytes, retained under ignored
`outputs/KEY-PROPOSAL-SYSTEMS-001/`, along with full artificial inputs, traces
and terminal logs. Compact metadata and receipts are in
`results/KEY-PROPOSAL-SYSTEMS-001/`. A later poll of the already closed campaign
handle reported unknown process; the terminal campaign/allfourchild records
confirm completion, and the campaign was not restarted.

## Implication and next state

The local machine can train this model size with realistic inputs and generate
consistent complete-key probability traces. Random-data losses/keys do not
measure cipher generalization. The next meaningful test is language-grounded
training with fresh synthetic dictionaries, matched small/large exposures and
two seeds, then all-case recovery and prior-shift/null controls. Source blocks
and keys need explicit isolation; previous exposed panels cannot be relabeled
fresh. Proposal probabilities remain separate from exact fitting weights.

As a **planning calculation only**,20,000updates atbatch4would take .486704hours
for the control and3.940607hours for the large model if these short-run rates
persisted. Two seeds at both sizes sum8.854622hours of updates before changing
data, validation, checkpoint and thermal overhead. This is neither a runtime
promise nor a launched/registered training campaign. A finite schedule and caps
must be published after corpus/data/validation design, before starting that work.
No new language training job is running. No latent circuit, source language or
Voynich reading has been established; the decipherment goal remains active.
