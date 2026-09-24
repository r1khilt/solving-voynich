# TEACH-0015: reusable-key and causal-geometry assay for TEACH-0014

**Status:** conditional, prospective design written while the TEACH-0014 v3
scientific campaign is running, before seeing any final neural prediction,
checkpoint, parser score or behavior decision. This is a separate follow-on;
it does not alter TEACH-0014 training, its frozen suite, or its decision rules.
The eligible checkpoint source is `37288feb7769ef62317cd3325ea1375d729f7d46`.
The experiment must receive a separate implementation registration, source
freeze, independent suite audit and finite resource benchmark before any
mechanism confirmation is run. This document alone is not launch admission.

## Question, prior basis and restricted claim

The primary question is whether the state after the first memory read,
`query.1`, carries the episode-specific intermediate symbol `k=F(n)` in a
form that a recipient's different `G` table can use. A raw model may obtain
correct answers through shortcuts; a readable key direction may not control
the later computation. Recipient-specific transfer is therefore the primary
criterion. The six-block encoder, candidate-edge parser, separate K/V
projections and exposed reader hooks define where the intervention occurs.

The source basis and transfer limits are in
[`TEACH-0014-design.md`](TEACH-0014-design.md) and the earlier
[`TEACH-0013.md`](TEACH-0013.md). The latter is **not entered** because the
TEACH-0012 raw competence gate failed; its controls can inform this new
registration but its thresholds/results cannot be inherited silently.
The repo's [J-space methods review](../research/latent-mechanisms-2026-09-21/METHODS_REVIEW.md)
distinguishes Anthropic's averaged downstream-Jacobian sparse token cones
from an arbitrary PCA subspace. This bidirectional parser plus recurrent
reader is not the paper's autoregressive transformer, so the Jacobian work
below is explicitly **J-inspired causal sensitivity**, not a replication of
Anthropic's J-space. The [token-geometry program](../research/latent-mechanisms-2026-09-21/TOKEN_AND_CAUSAL_GEOMETRY_PROGRAM.md)
separates static cosine, contextual relations and finite causal effects.

A passing assay can support a synthetic reusable-key mechanism in a named
TEACH-0014 arm. It cannot identify a Voynich plaintext token, historical
language, cipher, manuscript meaning, or a general global workspace.

## Entry and labels

Use only TEACH-0014 v3 final checkpoints after its no-model artifact auditor
and independent sampled numerical replay both pass. Require the shuffled
answer-label null to pass its registered leakage ceiling and the public-row
oracle to meet its two-hop executor gate. For an **answer-only emergent**
mechanism label, both `latent_rows_answer` seeds must separately pass the
registered native answer and parser absolute gates. For a **causal-supervised**
mechanism label, both `latent_rows_causal` seeds must separately pass the same
absolute gates; its interchange training must be named in every result.
`latent_rows_edge_aux` may enter an explicitly grammar-supervised secondary
assay if both seeds pass. None of these arms may borrow the other's pass.
If no raw arm qualifies, record `NOT ENTERED: COMPETENCE`; raw activation maps
may be exploratory but cannot receive mechanism labels. The oracle remains a
positive executor control and is not a learned-parser or emergent-mechanism
claim.

## Fresh counterfactual population

Generate **128 independent logical groups for discovery and 128 for
confirmation** under a new `TEACH-0015-key-transfer-v1` namespace, with a
separate fixed seed for each split to be frozen in implementation before
checkpoint access. Assign every rendering and counterfactual of a logical
group to one split by a predeclared hash. Require every F and G signal-stage
family to satisfy the existing TEACH-0014 `confirm` hash partition; training
uses `train` signal-stage families, so this proves no exact training signal
family can recur without regenerating millions of training episodes. Also
check exact graph and logical IDs against development seeds74111/74117, final
suite84311, stress seed141499, and between the two TEACH-0015 splits. Do not
use the TEACH-0014 final suite
as the mechanism confirmation pool. Independently reconstruct all visible
rows and requested outputs from tokens. The generator and auditor must reject
duplicate left-side mappings, missing outputs, identical counterfactual
targets, overlong sequences and incomplete groups.

Each group fixes one queried `n`, two F assignments `k0 != k1`, and three
independently remapped recipient G tables `G0,G1,G2`. Require all six
`Gj(ki)` outputs distinct within the group where feasible, and at minimum
require the unchanged donor answer `G0(k1)` to differ from both `G1(k1)` and
`G2(k1)`. Construct a donor `(F1,G0)` and bases `(F0,Gj)` plus the clean
`(F1,Gj)` targets for all three `j`. Cross two physical row permutations,
one marker-rich and one fully marker-free rendering, and two distractor
topologies. Preserve the same logical relation set across pure surface
variants. Follow-on implementation must report rejection counts and the
actual input lengths/marker counts rather than assuming perfect balance.

Fresh-panel clean competence is scored **before** interventions on the full
confirmation denominator. Both seeds of an eligible arm must reach at least
90% composed answers over all F/G cells, 80% exact three-recipient groups,
80% exact marker-rich/marker-free pairs, and 95% first-hop/direct plus 98%
copy on matched task variants. A failure yields `INCONCLUSIVE: FRESH-PANEL
COMPETENCE`; causal discovery maps remain exploratory only. A both-clean-
correct subset is reported but never replaces full denominators.

## Finite intervention contract

The primary site is **fixed** at the complete `query.1` state after the first
read and before the second. There is no layer/site search for the primary
assay. Capture the donor state once under `(F1,G0)`; patch that unchanged
tensor into each `(F0,Gj)` recipient with all other recipient states, K/V
slots, gate values and model parameters native. The desired output is
`Gj(k1)`, not the donor's fixed `G0(k1)` except for recipient0. Repeat in the
reverse F direction. Record full-vocabulary logits, native and target
probabilities, exact answers, first/second address mass on correct rows,
and every attempted item. A site intervention must preserve unpatched
components bit-for-bit where the hook contract allows.

Controls use the same pairs and output calls: identity patch and native-state
restoration; clean base/donor; recipient's own `(F1,Gj)` input-level target;
post-second-read `query.2` full-donor state as an answer-injection positive
control; wrong-key donor from a different logical group; cyclic within-cell
derangement; a norm-matched random delta; a same-key/different-render donor;
recipient-only G remapping; one-read and mean-address ablations when they pass
clean competence; and direct/copy task specificity. Wrong/random deltas are
scaled to the per-item norm of the true donor-minus-base `query.1` delta.
Store the actual replacement vectors and pairing IDs for an independent
intervention audit. Exact identity/restoration logits must agree with clean
logits under a predeclared CPU/MPS tolerance calibrated on random weights.

Primary `REUSABLE-KEY-STATE-SUPPORTED` requires **both seeds** of a named arm
to reach at least75% recipient-specific target items and60% exact
three-recipient groups on the full confirmation denominator; at least90% of
changed-G targets must avoid the fixed donor answer; reverse necessity must
restore the base-key answer on at least70% of eligible items; copy preservation
must be at least95%; both marked and marker-free transfer rates must be at
least70%; and the correct donor target rate must exceed each wrong-key,
deranged and norm-matched random control by at least35 percentage points.
Each control's full denominator and a group-bootstrap interval are reported.
If the state is necessary but fails transfer, label `NECESSARY, NOT PORTABLE`.
If only the causal-supervised arm passes, label it **supervision-dependent**.

## Causal geometry and J-inspired sensitivity

Only after the finite full-state site passes discovery may a discovery-fitted
subspace be tested on untouched confirmation. On discovery, form paired
donor-minus-base `query.1` deltas with logical-group weights, and separately
form VJPs of a predeclared recipient-specific target-vs-base logit contrast
with respect to the live `query.1` tensor. Parameter gradients are disabled;
do not use detached activation caches to compute a Jacobian. Save the
delta covariance and the **gradient second moment**
`E[g g^T]` separately from the signed mean gradient, since recipient remaps
can cancel the mean. A finite-difference screen on16 fixed discovery items at
steps `1e-2,1e-3,1e-4` checks the VJP path. No causal claim follows from a
large singular value, probe, cosine or Jacobian alone.

Candidate ranks are fixed to `1,2,4,8,16,32,64`. For each rank, construct
paired-delta PCA, VJP-second-moment, raw output-head span and Haar-random
matched-rank bases on discovery. Orthogonalize with a declared float64 CPU
SVD tolerance; report actual ranks, condition numbers and variance spectra.
Select the **smallest** rank that gives at least70% correct recipient target
items and50% exact three-recipient groups on discovery while its complement
alone transfers at most20%; ties follow the listed method order. Freeze the
method, rank, vectors and random seeds before confirmation. At confirmation,
apply the projected donor delta once, with the native orthogonal complement
preserved. Compare the complement-only edit, equal-energy off-subspace edits,
32 Haar-random subspaces of the same rank, deranged-pair PCA and full-state
transfer. A subspace label additionally requires at least70% target items,
50% groups, at least90% base-answer preservation under complement-only edit,
and at least35-point advantage over the strongest matched random/deranged
control in both seeds. Report all ranks descriptively even when selection
fails; do not choose a new rank on confirmation.

For a genuine J-lens-style token frame, a separate estimator and test suite
must define an averaged downstream Jacobian, final normalization, token
dictionary and sparse nonnegative cone fit exactly as specified in the
methods review. The present VJP covariance is a **different mathematical
object** and cannot be called Anthropic's J-space. Compare local linearized
predictions `g·Δh` with actual finite logit changes at scale factors
`0.25,0.5,0.75,1.0`; large discrepancies mark nonlinear or route-switching
behavior, not a failed finite mechanism by themselves.

## Neurons and token similarity, secondary only

At the frozen site, rank native coordinates by discovery mean absolute paired
delta and separately by mean absolute VJP-times-delta. Test top
`8,16,32,64` coordinate edits and ablations against 32 random same-count,
equal-energy coordinate sets and the equal-rank subspace. Report participation
ratio and cross-seed overlap after a discovery-only alignment. A single
neuron/function label requires its own necessity, sufficiency and rescue on
confirmation in both seeds; coordinate names are not invariant to a change of
basis.

Static input/output token cosine is a sanity check, not a vocabulary meaning
test. These 2,048 ordinary symbols receive episode-varying roles and random
relation pairings; a stable semantic neighborhood is not built into the
generator. Compare raw, centered and shrinkage-whitened cosine with
model-native query-key bilinear scores, random initialization, token exposure
frequency and matched permutations. For the actual relational question,
compare contextual states of the same symbol in F-left, F-right, G-left,
G-right, query and distractor roles under new graphs. Only a geometry that
predicts held-out *functional* substitutions or improves a finite edit earns
interpretive weight. No nearest-neighbor English gloss is inferred.

## Resource, multiplicity and audit admission

Implementation must benchmark the full finite-intervention grid on local MPS
before execution, cap wall time, sampled MPS allocation and artifacts, and
avoid paid APIs. It must keep TEACH-0014 source/checkpoints unchanged and
record hashes, exact seeds, generator rejection counts, software versions and
grouped denominators. The independent auditor must recompute visible oracles,
pair identities, target/control assignments, intervention outcomes, decision
order and numerical identity/restoration. Source and selection objects must
be committed before confirmation. Failure to qualify stops the confirmatory
path; exploratory traces can be archived under a distinct label. No TEACH-0014
or TEACH-0015 result licenses a manuscript decipherment claim without a
separate constrained decoder and manuscript holdout test.
