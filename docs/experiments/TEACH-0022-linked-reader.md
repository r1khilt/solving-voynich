# TEACH-0022: shared address geometry and explicit route supervision

**Status:** prospective new synthetic campaign after TEACH-0021,
2026-09-24. TEACH-0014 remains interrupted and its reserved seed 84311
remains unopened. This experiment is separate and cannot repair or
retroactively pass TEACH-0014.

## Scientific question

TEACH-0021 found that two correctly parsed oracle-row models normally
read the first row but miss the second row on nearly every erroneous
two-hop answer. Injecting the correct final row value rescues 30/31 and
28/28 errors on exposed development data. Is the failure fixable by
(a) a *linked* key/value state interface, or (b) explicit supervision of
the row-address trajectory, and does either generalize to fresh graph
families?

This is a controlled synthetic mechanism study. It does not assume
Voynich rows are graph edges. The prior TEACH-0004 typed-row positive
control and TEACH-0018 architecture contingency motivate addressable
memory. Graves et al.'s Neural Turing Machine and CLRS/Bieber graph
execution work, reviewed in TEACH-0018, support testing such
architectures, not a prediction that they solve a manuscript. The
project's architecture review notes that ALICE's supervised
substitution solver uses paired cipher/plaintext examples and a
one-to-one cipher family; explicit graph-row supervision here is
likewise a favorable calibration condition absent from Voynich text.

## Arms and matched exposure

Retain the completed TEACH-0014 `oracle_rows_workspace` checkpoints
as the answer-only **unlinked** baseline, only after rechecking their
source, checkpoint, training trace and answer-input/label hashes.
Train three new arms from the *same* two initialization seeds,
6,000 training steps, batch 32, training episode seeds and AdamW
learning rate 3e-4/weight decay .01 used by that baseline:

1. `unlinked_route`: unchanged public-row model plus route loss.
2. `linked_answer`: tie the query and row-key projection weights,
   carry the attended row's ordinary-symbol embedding directly as the
   next state value, and retain the old learned update behind a
   trainable positive scale initialized at 0.1; answer loss only.
3. `linked_route`: the identical linked architecture plus route loss.

The route loss is cross-entropy on the native address logits against the
unique correct visible row at each active hop, with coefficient 0.2.
It uses only public rows and visible query/answer; no generator
`signal_paths` or unseen future query is fed to the model. It *does*
provide a stronger training signal than answer-only, which the
manuscript does not offer. The four-cell design separates this
supervision effect from linking. The baseline and `unlinked_route`
share architecture/parameter count. The linked arms have fewer
independent projection parameters; compare actual parameters and
measured compute, not claim exact parameter matching. Same training
examples and step budget address exposure, not capacity.

## Fresh evaluation and decision

Generate full 19-panel/128-group suites once at seeds 84411
(development) and 84511 (confirmation) using the unchanged
TEACH-0014 graph generator. No redraw, adaptive selection, or early
final scoring. Audit both suites before training for canonical
structure, stage partitions and exact graph/logical/render overlap
against the exposed TEACH-0014 seed-74111 and prior reserved
seed-84311 manifests; the latter may be inspected **only for IDs and
exposure**, never for model scores. If the fixed draws overlap
prohibited units or fail structural validity, stop and register a
prospective amendment; do not silently resample. Raw suite files
remain ignored with hashes in compact manifests.

All four arms are scored on development and confirmation with exact
item and group outcomes, first and second target-row attention argmax,
copy/one-hop/two-hop, marker-free, alias and longer-hop/OOD panels.
The **primary endpoint** is confirm-confirm two-hop item accuracy on
seed 84511 in both model seeds, with exact factorial groups a
co-primary structural check. A useful linked-reader result requires
at least +15 percentage points over the original unlinked baseline
for confirm-confirm in each seed, and at least +15 points over
`unlinked_route` for `linked_route` if claiming a link effect
beyond supervision. Require at least 90% first-hop, 90% direct, 95%
copy and 70% exact factorial groups in each seed to call a model
competent; otherwise report its particular deficits. These absolute
gates were selected from the prior exposed development pattern,
not tuned on the fresh final outcomes. Route-only superiority is a
supervision result, not a learned unsupervised mechanism.

The 84511 result is the sole confirmation use of that suite. Any
subsequent architecture change needs a new suite. The known
row-pointer/rank/query-relative symbolic shortcuts remain alternative
explanations for finite interchange tasks; this study does not claim
key-coordinate identification.

## Execution, controls and resources

Before trained inference, source-freeze model, trainer, independent
suite/trace/score auditor and sampled checkpoint replay. Use no paid
service. A local MPS benchmark must project all three new arms×two
seeds×6,000 steps under four hours using
`1.5 × worst measured median step × 36,000 + 900s` and verify a
sampled 12-GiB MPS cap and 2-GiB total new-artifact cap. If the
projection or actual limit fails, stop and record an amendment; never
replace six thousand steps post hoc with a convenient shorter run.
Write atomic per-run checkpoints and progress with enough optimizer
state to resume an interrupted run without changing its example
stream; never infer liveness from a status file alone.

Independent audit checks source and dataset hashes, every checkpoint
identity, exact per-step input/label hashes against the prior baseline,
all panel/row denominators, decision arithmetic and artifact sizes.
Sampled full-logit CPU replay checks fixed panel positions from all
new checkpoints. Mechanistic follow-up requires a separate
intervention registration and native positive/wrong controls; high
attention accuracy or benchmark performance alone is not a circuit
identification. A synthetic competence gain does not imply a Voynich
plaintext or justify applying graph semantics to manuscript glyphs.
