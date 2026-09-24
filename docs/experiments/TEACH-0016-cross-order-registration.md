# TEACH-0016: cross-order source/recipient state transfer

**Status:** prospective hypothesis and design, 2026-09-24, before any
TEACH-0014 v3 final neural outcome, TEACH-0015 trained screen, or finite
intervention. No TEACH-0016 model inference has run. A separate suite freeze,
independent auditor, implementation and source-matched MPS resource benchmark
are required before this becomes an executable confirmation.

## Why this exists

The frozen TEACH-0015 three-G intervention asks a good causal question but
cannot by itself identify a *symbol* `k=F(n)` at `query.1`. All three G tables
put `G_j(k1)` at the same physical row slot under the shared rendering seed.
A constructed state that carries only that slot passes **every registered
behavioral threshold** on both frozen splits; see
[`TEACH-0015-row-pointer-counterexample.md`](TEACH-0015-row-pointer-counterexample.md).
The exact row-pointer construction fails the opposite-marker control, but
same-key marker transfer is descriptive rather than part of the original
primary conjunction. A neural pass on that control would weaken this exact
counterexample but not prove a unique symbol coordinate.

Interchange interventions test whether a high-level variable can be aligned
with internal behavior, not whether the alignment is unique. Distributed
alignment search explicitly allows non-neuron bases
([Geiger et al., 2024](https://proceedings.mlr.press/v236/geiger24a.html));
general causal representation identification needs further assumptions or
interventions
([Bing et al., 2024](https://proceedings.mlr.press/v236/bing24a.html)).
This assay tests one concrete rival: an **absolute physical G-row pointer**.
It is not Anthropic's averaged-Jacobian J-space construction
([Gurnee et al., 2026](https://transformer-circuits.pub/2026/workspace/index.html)).

## Fresh visible population and selection

Draw128 discovery and128 confirmation groups from the unchanged TEACH-0015
two-hop generator with newly frozen Python `random.Random` seeds **86111** and
**86121** respectively. Reuse the generator's public `confirm` stage-family
partition and its TEACH-0015 group split-hash rule, but give these draws a
TEACH-0016 suite identity. This is a fresh draw from the **same** generator,
not a newly identified data-generating family. No changing
seeds after examining a model or a failed suite audit. Audit all66 visible
episodes per group, exact group/graph/logical/render IDs against both
TEACH-0015 splits and exposed TEACH-0014 development/final suites, stage
family separation from training, six-way distinct recipient answers,
preprocessing, row counts and token lengths. The corpus is synthetic with
public graph oracle labels; it is not a manuscript evaluation.

For each group, distractor topology `d∈{0,1}`, marker condition
`m∈{rich,free}`, and source row order `o∈{0,1}`, use the **opposite** order
`1-o` for each recipient. Source donor `(F1,G_a,d,m,o)` is captured once for
each `a∈{0,1,2}`; patch its unchanged `query.1` state into each base
recipient `(F0,G_b,d,m,1-o)` for `b∈{0,1,2}`. The desired output is
`G_b(k1)`. This is128×2×2×2=1,024 crossed-order surface pairs and
9,216 source/recipient attempts per split per arm/seed. The off-diagonal
`a≠b` subset has6,144 attempts. Define the **shifted-slot** subset before
inference using visible rows: the physical index of `(k1,G_a(k1))` in the
source differs from that of `(k1,G_b(k1))` in the recipient. The suite is
valid only if at least85% of crossed-order pairs shift that slot in each
split; report the exact count, do not redraw if this fails. Preserve all
attempts in the archive; shifted off-diagonal items are the fixed primary
denominator. The physical row index, not token offset, defines this test.

Changing the whole rendering order can also introduce nuisance differences.
Therefore this experiment uses a strong same-key positive control and cannot
interpret a negative transfer without its success. The source/recipient
pool is fresh; no TEACH-0016 group may be used to choose model arms, site,
rank or thresholds. The source G and recipient G are crossed fully, so success
cannot be explained by copying one fixed source answer.

## Entry and controls

First require TEACH-0014 v3 no-model artifact and sampled checkpoint replay
audits to pass. Then require the independently audited/replayed TEACH-0015
fresh clean screen and finite assay. A named arm enters TEACH-0016 only if
**both its seeds** pass the TEACH-0015 finite
`PORTABLE-INTERMEDIATE-STATE-SUPPORTED` criteria after replay. Keep the
answer-only, interchange-supervised and public-grammar-supervised arm labels
separate. If no arm enters, record `NOT ENTERED: PRIOR PORTABLE STATE`; no
TEACH-0016 trained inference occurs.

Before patches, run a source-hashed clean competence screen on *all*8,448
episodes of each fresh split, requiring the same ≥90% composed, ≥80% exact
recipient triples and marked/free pairs, ≥95% first-hop/direct and ≥98% copy
thresholds in **both seeds** of the named arm. Failure is `INCONCLUSIVE:
FRESH COMPETENCE`, not evidence against a key representation.

For every crossed pair and source G, archive:

- clean base/target/donor predictions and exact public answers;
- identity patch and same-key source-order donor into `(F1,G_b,1-o)`,
  which should preserve target behavior despite the order change;
- unchanged donor-state transfer into `(F0,G_b,1-o)`;
- matched wrong-key source `(F0,G_a,o)` into `(F0,G_b,1-o)`;
- deterministic norm-matched random delta and a distinct-group donor of
  matched surface, both with actual replacement vectors;
- reverse `(F0,G_a,o)` into `(F1,G_b,1-o)` to test restoration of `G_b(k0)`;
- final-answer-state injection, to expose simple source-answer copying.

The no-neural absolute-row-index baseline reads the recipient row at the
source's `G_a(k1)` slot. Its target rate on the shifted subset is a required
reported control. The benchmark must time the **complete** batched
intervention/archiving surface, not a single patch. Independent no-model
auditing must reconstruct visible targets, pairings, actual vector formulas,
all prediction counts and exact denominators; sampled CPU checkpoint replay
must recompute logits and vectors. No criterion may be selected after seeing
discovery or confirmation scores.

## Frozen confirmation decision and limits

Discovery is descriptive. On the confirmation split, each seed of the same
named arm must jointly achieve all of the following on the **shifted** subset:

1. ≥70% recipient-specific target accuracy on off-diagonal `(a,b)` attempts;
   ≥55% exact three-recipient triples for each source G, with all nine
   attempts and both order directions reported.
2. ≥90% same-key positive-control target accuracy on those off-diagonal
   attempts. If this fails, interpret the cross-order manipulation as
   incompatible with the full-state interface and mark the assay
   **inconclusive**, irrespective of target transfer.
3. ≥90% off-diagonal non-injection: transfer output differs from the source
   donor's fixed answer; ≥65% reverse restoration among both-clean-correct
   attempts, with full denominator and eligible count reported.
4. ≥65% target transfer separately under rich/free markers and each of the
   two source-order orientations.
5. Correct transfer beats each matched wrong-key, distinct-group,
   norm-matched random and visible absolute-row-pointer target rate by ≥35
   percentage points on the *same shifted off-diagonal denominator*.

Report all primary and nonprimary denominators, 4,000-resample intervals over
logical groups, per-G-source/recipient matrices and both seeds. The
registered label is `ORDER-ROBUST-PORTABLE-STATE-SUPPORTED`. It rejects the
specific absolute-row-pointer explanation on these unseen programs but does
not uniquely identify a symbol token, a coordinate axis, a J-space cone, or
a Voynich reading. A state could still carry a relational instruction,
permutation-invariant row identity or a distributed key-equivalent program.
If only the interchange-supervised arm passes, report that dependence on
causal training rather than calling the mechanism emergent.

## Resource and source gate

Worst-case model grid is three eligible arms×two seeds×two splits×1,024
crossed pairs=12,288 complete surface pairs, each with nine source/recipient
transfers plus registered controls. Before inference, freeze code/manifests,
run a source-matched random-weight MPS benchmark of all eight cross-order
surface types and every control, then require `1.75×slowest type median×12,288
+300s` below a **4h wall cap**. Sampled MPS allocation must stay below12GiB,
bulk artifacts below3GiB, and no paid API is used. Actual runtime and disk
caps are enforced throughout. If the benchmark fails, stop or prospectively
amend the design and rebenchmark; do not silently drop controls or groups.
No TEACH-0016 confirmation starts until these entry, numerical and resource
gates are independently audited and committed.
