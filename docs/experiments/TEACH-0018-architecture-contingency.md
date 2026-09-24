# TEACH-0018 architecture contingency after TEACH-0014 audit

**Status:** prospective design, 2026-09-24, before final TEACH-0014 v3
neural results or any trained TEACH-0015--0017 intervention. No new model is
trained here. This is a decision plan, not an amendment to source-frozen
TEACH-0014/15/16 criteria or evidence that a larger network solves Voynich.

## Start with the observed failure mode, not parameter count

The current raw `CandidateEdgeWorkspace` has24,766,465 parameters (the
matched dense control has24,757,760). It uses a six-block bidirectional
raw-token encoder, builds every adjacent ordinary-token pair as a candidate
edge, processes those candidates with two parser blocks, learns a scalar
edge gate, and performs recurrent content-addressed reads. For marker-free
rows, the true pairs are the **even** adjacent ordinary-token pairs; false
cross-row candidates are odd pairs. The task therefore asks the network to
learn both a position-insensitive pairing grammar and a sharp, repeatable
two-hop lookup. Four-step recurrent/diffusion gate refiners test one route
to the former. The public-row oracle and one-read/mean-address/null controls
already separate several alternatives. These are facts about the source,
not yet neural outcome claims.

One hundred times the current parameter count is about2.48 **billion**
parameters. With float32 parameters, gradients and two Adam moments alone,
that is roughly39.6GB before activations, optimizer overhead or
checkpointing. It cannot fit the currently registered12GiB MPS allocation
cap. A scale increase therefore needs a new measured resource registration,
and cannot be smuggled into the ongoing campaign. More width also need not
teach an absolute-position encoder to pair marker-free operands or form a
query-dependent algorithm. Work on self-attention and parity gives reasons
to test grammar generalization, but does **not** prove this finite residual
model cannot learn it: [Chiang and Cholak (2022)](https://aclanthology.org/2022.acl-long.527/)
construct transformers that recognize parity and show architecture/normalization
details matter.

## Audit-triggered branch points

1. **Public-row oracle fails clean two-hop answers.** Treat the reader/state
   update, optimization, or output binding as the bottleneck. Do not call a
   raw parser failure from this. Before changing scale, test a tied
   identity-initialized value path and a residual key-state update against
   the existing oracle-row model at the same parameter count and training
   budget. Inspect first/second target-edge attention and exact answers on
   held-out graph families. A hard-coded graph oracle is a grammar-informed
   control, not a mathematical accuracy upper bound or decipherment evidence.
2. **Oracle succeeds but raw edge gates fail.** Treat row segmentation as the
   bottleneck. Compare a differentiable finite-state pair scanner over
   ordinary-token occurrences, a local relative-position parser, and the
   current absolute-position transformer, with the same reader and
   parameter/compute-matched controls. The scanner must learn or be told
   which tokens are ordinary; if told, label it a grammar-informed model.
   Test changed row count, marker style/dropout and independent row shuffles,
   not only in-distribution answer loss. The frozen false-candidate gate
   diagnostic can localize an edge-gating defect but cannot prove that a
   replacement architecture discovers grammar from Voynich text.
3. **Clean raw answers succeed but finite state transfer fails.** Preserve
   the working parser and vary only the state/update interface. Compare a
   direct value-carry residual, a slot bottleneck with explicit key-equivalent
   readout, and recurrent transition-predictive supervision. Require clean
   accuracy, matched full-state patch positive control, donor/recipient G
   crossing, reverse restoration and held-out family performance. A
   representation probe alone cannot rescue a failed intervention.
4. **TEACH-0016 succeeds but rank-shift TEACH-0017 fails.** The model may
   have learned a relative address. Train on controlled cross-distractor
   examples or add rank-invariance supervision only in a *new* campaign;
   then test fresh seeds and independent rank shifts. Report this as added
   causal supervision, not spontaneous world-model emergence.

These branches depend on independently audited **both-seed** behavior. A
single fluent output, good loss, attention picture, or parse gate correlation
is insufficient. No adaptive model selection uses a later confirmation
split; exploratory tuning uses new development draws and records all arms.

## Candidate architecture ladder

**A. Typed-edge recurrent workspace.** Keep raw visible tokens as input but
build a learned pair-boundary/role posterior, then a permutation-equivariant
set of edge slots with key/value projections and a recurrent query state.
Expose native hooks at boundary logits, edge keys/values, query before and
after each read. Require a source-only row-order permutation test: if the
parsed slot multiset changes under a pure row reorder, the proposed invariant
   mechanism is invalid. The current public-row oracle remains a
   grammar-informed control. Differentiable external-memory models demonstrate addressable
memory as a viable architectural idea, not a result on this corpus
([Graves et al., 2014](https://arxiv.org/abs/1410.5401)).

**B. Instruction-pointer graph process.** When the grammar is explicitly
known, encode each visible row as a directed edge and alternate a node/edge
message step with a pointer update. This is a fair *algorithmic competence*
teacher, not a blind manuscript solver. It tests whether the current reader
architecture is the main bottleneck and provides a causal source of known
key-state trajectories. Program-execution work motivates combining graph
structure with recurrent pointer dynamics
([Bieber et al., 2020](https://proceedings.neurips.cc/paper/2020/file/62326dc7c4f7b849d6f013ba46489d6c-Paper.pdf));
the CLRS benchmark emphasizes held-out algorithmic generalization rather
than memorized examples
([Veličković et al., 2022](https://proceedings.mlr.press/v162/velickovic22a.html)).
Neither transfers an accuracy guarantee to Voynich.

**C. Scaled raw model only after an optimization curve.** Candidate widths
512/768/1024 and more parser/reader depth should be benchmarked with the
*same* data, supervision, gradient-token budget and raw-input controls.
Width changes alter tied embedding/output capacity, so parameter-matched
and compute-matched baselines answer different questions. Require a
predeclared improvement on a held-out answer/parser gate and a measured MPS
projection before adding a larger tier. No 2.5B-parameter run on the user's
Mac is implied by enthusiasm alone. If a larger network merely improves
training loss or in-family answers while rank-breaking transfer stays at
chance, it has not solved the mechanistic question.

The current four-step diffusion-like edge refiner is a diagnostic arm, not a
generative diffusion model of the manuscript. A true world/action model
would need a defined state-transition objective and independent future
queries, with a prediction baseline and causal transition commutation test.
Calling a recurrent reader a “world model” without those tests would add
terminology, not evidence. VLA-inspired instruction/action conditioning is
similarly interesting only if the action variables and intervention labels
are specified and compared against simpler supervised graph execution.

## Validation contract for any selected arm

Freeze generator, train/dev/final graph-family splits, model source,
parameter count, seeds, optimizer, step count, hardware and measured
resource caps before training. Keep the original TEACH-0014 campaign and
all its outcomes immutable. Two seeds, answer-only, parser-supervised and
causal-supervised comparisons should make the effect of supervision clear.
Evaluate exact copy, one/two/longer hops, parser gates, clean transfer,
cross-G, cross-order, rank shift, and out-of-family graph/length tests as
separate gates. Use the previously demonstrated row-pointer and four-bit
rank programs as explicit shortcut baselines. Replay sampled logits and
intervention vectors from checkpoints; treat failed positive controls as
inconclusive. Only then pursue J-inspired geometry, SAE/transcoder or
neuron-level causal tracing, and compare to the tied output span. The local
`METHODS_REVIEW.md` gives the interpretation limits for those methods.

This architecture ladder advances a *synthetic competence and mechanism*
study. It remains upstream of the manuscript requirement in
`docs/research/PROTOCOL.md`: a constrained decoder with frozen rules,
external predictions and blind held-out folios. None of these synthetic
models supplies a Voynich translation by itself.

The closest decipherment analogue in the local
`docs/research/ARCHITECTURE_REVIEW.md` is the supervised substitution solver
[ALICE](https://arxiv.org/abs/2509.07282): its paired plaintext/ciphertext
training and one-to-one cipher assumption are far stronger than the evidence
available for the manuscript. A successful synthetic graph reader therefore
calibrates the analysis machinery; it does not establish that the manuscript
contains graph programs, a substitution cipher, or recoverable semantic
states. Any eventual manuscript-facing arm needs explicit null and
homophonic/finite-state alternatives at matched data scale.
