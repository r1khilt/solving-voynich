# TEACH-0017 design: separate key identity from program-relative rank

**Status:** prospective design, 2026-09-24, before TEACH-0014 v3 final neural
outcomes and before any TEACH-0015/16 trained intervention. The near-term
rank-shift suite seeds and validity floor below are fixed; an executable
causal assay still needs source, audit, numerical replay and a measured
resource gate. No neural result or new training change is claimed. The
longer cross-group triad remains a design without fixed suite seeds or
confirmation thresholds. The long-term target is manuscript decipherment.

## Why another crossed intervention is necessary

TEACH-0016 breaks the **absolute physical row** shortcut. It cannot break
an order-invariant **relative address**. The source-frozen TEACH-0016 visible
counterexample carries only rank0--15 of the first-hop key among sorted row
left operands; it passes every behavioral threshold on both fresh splits
while the distinct-group rank null remains below9%. The independent audit is
`results/TEACH-0016-rank-rival/audit.json`. This does not say the trained
model uses a rank. It says a future TEACH-0016 neural pass cannot identify
global token identity from behavior alone.

The existing fresh groups offer too few natural same-key cross-**group** pairs:
only3 unordered `key1` collision pairs among128 discovery groups and6
among128 confirmation groups. These counts come from the visible manifests,
not neural selection. A deliberate paired generator is needed; repeatedly
mining these suites after seeing scores would be a selection error.

## Near-term TEACH-0017: change distractor context within a group

The existing TEACH-0016 groups already contain two independent distractor
topologies `d=0,1` for the *same* key and three G tables. TEACH-0016 always
pairs donor and recipient at the same `d`, so the sorted-left-operand set is
identical. An exploratory **visible-only** scan of its now-exposed suites
paired source `d` with recipient `1-d`, and source row order `o` with
recipient `1-o`. It found that the key's relative rank shifts in704/1,024
discovery and720/1,024 confirmation surface pairs (4,224/6,144 and
4,320/6,144 off-diagonal attempts). The rank-address oracle hit0 targets
on those shifted subsets. These numbers are *development evidence*, not a
TEACH-0017 confirmation. The TEACH-0016 confirmation split is now exposed
to this design and must not be reused as TEACH-0017 confirmation.

Draw **128 fresh discovery groups with seed87111 and128 fresh confirmation
groups with seed87121** from the unchanged TEACH-0015/16 generator and
its existing `confirm` stage partition. Freeze the source bytes and exact
manifest IDs. Audit all66 visible episodes per group, leakage against the
exposed TEACH-0014--0016 suites, answer distinctness, left-set changes and
rank shifts. Do not redraw if the predeclared shifted-rank fraction is below
**60%** in either split; that is a suite-validity failure. Use all eight
`(source_d, marker, source_order)` surfaces and three source Gs crossed with
three recipient Gs, with recipient `d=1-source_d` and
`order=1-source_order`. Archive all9,216 attempts per split/arm/seed; the
shifted-rank off-diagonal subset is the primary denominator. This is still
the **same group** and may allow a group-specific code, so it addresses only
the four-bit relative-rank rival.

Condition trained inference on primary TEACH-0014 audit/replay, TEACH-0015
fresh competence and finite portable-state replay, TEACH-0016 fresh clean
competence, and a TEACH-0016 order-robust portable-state pass in both seeds.
If these fail, mark TEACH-0017 not entered without peeking at confirmation
neural outcomes. The new suite also needs its own all-cell clean competence
screen. Test full-state same-key cross-distractor transfer, a same-key
source/recipient F1 positive control, reverse F0 restoration, same-group F0
wrong-key, globally deranged distinct-group, norm-matched random, visible
rank-address and final-answer injection controls. Require the F1 positive
control≥90% on confirmation shifted-rank off-diagonal attempts before
interpreting negative transfer. Proposed numeric thresholds for the later
source-frozen assay mirror TEACH-0016's70/55/90/65/35-point conjunction;
freeze their exact field definitions and all denominators in a separate
execution registration before any TEACH-0017 trained inference. A passing
benchmark must keep the full assay within4h local MPS/12GiB/3GiB, or the
assay remains unlaunched. No paid API or corpus downloads are needed.

Even a success here only rejects relative rank under changed distractors;
a group-fingerprint/branch code could remain. That motivates the harder
cross-group triad below.

## Longer-term cross-group matched triad construction

Generate a recipient program `R` with distinct intermediate keys `k_A` and
`k_B` and unique outputs `G_R(k_A)`, `G_R(k_B)`, and base `G_R(k_0)`. Generate
source `A` on an independently sampled graph whose queried first hop is
`k_A`, but change unrelated left operands so
`rank_A(k_A) != rank_R(k_A)`. Generate source `B` whose queried first hop is
the **different** key `k_B`, but arrange unrelated left operands so
`rank_B(k_B) = rank_R(k_A)`. The source answers and recipient G outputs must
be distinct. Include several rank gaps and both directions of the gap.
Changing only one distractor can make the task too easy to fingerprint;
counterbalance the number, topology, and numeric ranges of changed
distractors while keeping sequence length and marker schedule comparable.

Use the *unchanged* frozen neural site `query.1` first. Source A → recipient
should produce `G_R(k_A)` under a key-equivalent state and generally fails
under the relative-rank program. Source B → recipient should produce
`G_R(k_B)` under key equivalence, while pure rank predicts `G_R(k_A)`.
This second collision makes a rank-based false positive directly visible;
one same-key transfer success alone could be an output or ordinal accident.
Cross at least two recipient G remaps so copying a source final answer cannot
explain success. Perturb row order and marker dropout independently in source
and recipient, with clean source/base/target predictions archived.

The raw grammar must remain solvable from visible rows and contain every
queried continuation. Before fitting/selection, audit stage-family and exact
graph/render separation from TEACH-0014--0016, answer uniqueness, key/rank
relations, left-set changes, ordinal-baseline expected outputs, and the
absence of a simpler preserved physical-slot cue. Positive controls need
native identity, same-program rerender patch, and same-context key transfer;
matched negative controls need wrong key, same rank/different key, unrelated
graph, norm-matched random delta and final-answer-state injection. A
cross-graph full-state patch may be off-manifold even if clean answers are
right. Therefore success is informative, while failure is **inconclusive**
unless a more specific cross-graph patch-validity control passes. A local
perturbation scale sweep can diagnose discontinuity but must be labeled
exploratory unless registered in advance.

## Representation geometry as a secondary measurement

Collect first-read states on *repeated keys in independent graph contexts*
and intentionally rank-matched different keys. This design makes a token
similarity question meaningful. Raw cosine between two state vectors is a
description, not a mechanism: shared means, norm variation, graph identity,
answer leakage and output-head directions can dominate it. Compare at least:

1. Raw, mean-centered and train-only shrinkage-whitened cosine; covariance
   and centering must be fit on development groups only.
2. A cross-validated key-equality versus rank-equality representational
   dissimilarity matrix, with logical groups as resampling units and matched
   graph/length/answer controls. Cross-validated Mahalanobis distances are
   useful when repeated measurements estimate noise covariance, but graph
   variation here is partly real signal, so do not import a neuroscience
   noise model unchanged.
3. Linear CKA for *whole representation matrices* across seeds/sites on the
   same held-out input set, not as a per-token similarity score. CKA can
   reveal broad geometry while leaving a causal-variable claim unresolved.
4. Held-out-context probes for key token, rank, source answer and graph
   fingerprint at equal capacity; a probe that decodes key does not show
   the continuation actually uses it. Test the selected direction with
   finite replacement patches and matched rotated/output-span baselines.

Choose one primary causal question before confirmation and account for all
tested sites, ranks and metrics. If the current model fails clean competence,
do not rescue a geometry claim with attractive nearest neighbors. A stronger
row-permutation-equivariant or typed-edge architecture could be trained in a
separate registered campaign, with its own answer-only baseline and null;
the present design does not authorize reading a failed backbone as evidence
against the abstract mechanism.

## What this could and could not identify

If same-key/different-rank donor states transfer across independent graphs
and same-rank/different-key states reliably select `G_R(k_B)`, the relative
rank rival is rejected on the tested population. This supports a globally
portable **key-equivalent** state. It still cannot identify a literal token
coordinate, because any invertible recoding `h(k)` supports the same
recipient computation. Without assumptions on the representation map and
intervention family, causal variables are generally underidentified
([Bing et al., 2024](https://proceedings.mlr.press/v236/bing24a.html)).
Distributed alignment can be found in non-neuron bases
([Geiger et al., 2024](https://proceedings.mlr.press/v236/geiger24a.html)).
Linear CKA measures whole-matrix similarity and has its own invariance
limits ([Kornblith et al., 2019](https://proceedings.mlr.press/v97/kornblith19a.html)).
These papers motivate controls; none proves a Voynich decipherment or that
the local synthetic backbone has Anthropic's J-space. The manuscript task
would still need a separately frozen decoder, held-out folios and external
constraints under `docs/research/PROTOCOL.md`.
