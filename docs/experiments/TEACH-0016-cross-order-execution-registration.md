# TEACH-0016 crossed-order execution freeze

**Status:** prospective implementation specification, 2026-09-24. The full
TEACH-0014 v3 neural campaign is still training. No trained TEACH-0015/16
screen or state transfer has been run. This document supplements, without
changing, `TEACH-0016-cross-order-registration.md`.

## Fixed cells and controls

Use the frozen 128 groups and eight `(distractor, marker, source_order)`
surfaces in each TEACH-0016 split. For every surface, capture the three
`(F1,G_a,source_order)` donors, three `(F0,G_a,source_order)` wrong-key and
reverse donors, three matched distinct-group `(F1,G_a,source_order)` donors,
and all six `(F0/F1,G_b,recipient_order)` cells, where
`recipient_order=1-source_order`. Cross `a,b∈{0,1,2}` in row-major order.
The distinct group is the globally deranged split-wide permutation produced
by the frozen TEACH-0015 control-pair algorithm's first returned permutation;
it has a different `key1`, and every group is used exactly once as a donor.
There is no donor selection based on neural scores.

At `query.1`, patch into `F0,G_b` the unchanged source `F1,G_a` first state
(transfer); patch into `F1,G_b` the same-key `F1,G_b` source-order first state
(order-change positive control); patch into `F0,G_b` the `F0,G_a` source-order
first state to reverse the target state. Patch the native `F0,G_b` first state
into itself for identity. For wrong-key and distinct-group controls, scale
their source-minus-base delta to the norm of the actual transfer delta, then
add it to the same base state. For random control, draw a CPU standard-normal
512-vector with SHA-256-derived seed
`TEACH-0016-random|group_id|a|b|d|marked_bit|source_order`, scale it to that
same norm and add it to base. At `query.2`, patch the `F1,G_a` source's final
state into `F0,G_b` for final-answer injection. The two native identity paths
must have maximum full-logit error≤0.003 and identical ordinary argmax.

Archive each attempt's identities, exact public answers, physical source and
recipient `G(k1)` row indices, shifted flag, clean source/base/target
predictions and every intervention prediction. Archive the ten actual float32
vectors named in `teacher16_intervene.VECTOR_NAMES` for **every** attempt, in
`(128,8,9,10,512)` `.npy` arrays per split/arm/seed. This deliberately
retains repeated vectors to make each attempt independently inspectable;
the worst-case raw vector allocation is2,264,924,160B across all twelve
split/arm/seed cells, below the3GiB assay artifact cap before metadata.
Archive full2,064-class logits for all clean and intervention conditions at
surface positions0,512,1023 per split/arm/seed only. Other attempts have
ordinary predictions, vector replacements and public targets, not full
probability or attention traces. Do not infer those unsaved quantities.

## Admission, scoring and failure

Freeze all execution/audit source, benchmark complete nine-attempt surfaces
with all controls on random weights (all eight surface types, one warmup and
three timed surfaces per type), and independently require
`1.75×slowest-type median×12,288+300<14,400s`; enforce4h wall,12GiB sampled
MPS and3GiB artifact caps. A benchmark is a resource gate, not a result.
Before trained inference, require the TEACH-0014 primary artifact and sampled
checkpoint replay, TEACH-0015 clean screen and finite assay no-model+CPU
replays, and TEACH-0016 fresh clean screen no-model+CPU replay. If no named arm
passes both seeds at each prior gate, record a distinct not-entered reason
without loading a checkpoint. Never substitute an oracle-row teacher.

The independent no-model assessor must verify the full visible grid, global
distinct-group pairing, every actual vector formula and norm, all prediction
counts, sample-logit argmax, exact denominators, per `(a,b)` and per-order
rates, and4,000 paired group resamples. The separately implemented CPU replay
must recompute all condition logits and replacement vectors at the frozen
sampled surfaces from checkpoint bytes. A candidate result cannot become the
registered `ORDER-ROBUST-PORTABLE-STATE-SUPPORTED` label before replay.
Apply the **unchanged** confirmation thresholds in the parent registration,
with shifted/off-diagonal as the primary denominator and source-G triples
counted only when all three recipient outcomes for that source are correct.
The positive control must pass in both seeds before a transfer failure has
mechanistic meaning. Discovery remains descriptive. This challenge rejects
only the literal absolute-row-index rival on unseen synthetic programs;
neither success nor failure is a Voynich decipherment or unique J-space key.
