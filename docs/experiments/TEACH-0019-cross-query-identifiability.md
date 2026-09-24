# TEACH-0019: cross-query transfer and offset-collision design

**Status:** mathematical design only, 2026-09-24. No suite is generated,
no training is run, no checkpoint is scored, and no neural threshold is
registered here. TEACH-0014 final audit and the conditional TEACH-0015--17
gates retain priority. This document does not alter their frozen sources.

## The actual identification question

For a donor context `c`, let the visible first hop produce intermediate
key `k`, an encoder place state `E_c(k)` at the frozen `query.1` site, and
the recipient continuation `D_r` use that state to choose an output.
The desired high-level rule is `D_r(E_c(k))=G_r(k)` when recipient `r`
has a visible continuation for `k`. Passing this equation on a finite
set of tested donor/recipient pairs does not uniquely identify the
coordinates of `E_c(k)`.

In all current TEACH-0015--17 paired surfaces, donor and recipient have
the same query `q`. The constructed state `E_c(k)=k-q (mod 2048)` and
recipient decoder `D_r(z)=G_r(z+q_r)` therefore satisfy the equation
exactly. The source-frozen TEACH-0017 visible-only audit confirms the
result on both complete test suites in
`results/TEACH-0017-query-relative-rival/`. This is our own algebraic
counterexample, not a trained-model finding. It is an instance of the
general causal-representation underidentification cautions in
[Bing et al. (2024)](https://proceedings.mlr.press/v236/bing24a.html) and
distributed alignment in
[Geiger et al. (2024)](https://proceedings.mlr.press/v236/geiger24a.html).

Think of each query value as a **context chart**. The current transfer
graph has edges only within a chart. Its interventions cannot determine
how the coordinates of one chart align with another. A globally portable
key-equivalent state requires transfer edges across charts. Even complete
cross-chart success cannot fix a unique neuronal basis: any global
invertible recoding `h(k)` with an inverse in the recipient is equivalent.
The attainable claim is therefore functional portability under the tested
intervention family, not literal token-ID neurons.

## A two-axis matched collision, before any neural outcome

Generate a recipient program `R` with one fixed query `q_R`, at least two
distinct second-stage keys `k_A` and `k_B`, and unique outputs
`G_R(k_A)`, `G_R(k_B)`, and `G_R(k_0)` for its native first-hop key `k_0`.
Generate donor programs with first-hop keys and queries arranged in five
paired conditions; `A`--`D` use independent graphs and `P` is a same-graph
patch-validity control:

| Donor relation to `R` | Key relation | Offset relation | What it isolates |
| --- | --- | --- | --- |
| `A`: independent graph, changed query | donor reaches `k_A`; `R` continues `k_A` | `k_A-q_A != k_A-q_R` | Does the same key cross query charts and graphs? |
| `B`: changed query, different key | `k_B != k_A` | `k_B-q_B = k_A-q_A` | Do equal offsets falsely collapse different keys? |
| `C`: same query, same key | donor reaches `k_A`; `R` continues `k_A` | same offset | Same-query cross-graph positive control. |
| `D`: changed query, different key and offset | `k_B != k_A` | different offset | Matched negative/control geometry. |
| `P`: same graph, changed query | donor reaches `k_A`; `R` natively reaches `k_0` but continues `k_A` | different offset | Cross-query patch-validity positive control. |

All offsets are modulo2048. `A` and `B` must be injected into the **same**
recipient graph and frozen site. A true key-equivalent state selects
`G_R(k_A)` for `A` and `G_R(k_B)` for `B`. Pure offset coding sends the
same code from `A` and `B`, so it cannot produce both distinct targets in
the same deterministic recipient. The `A` contrast alone might fail
because cross-graph patches are off-manifold; the `B` collision gives a
direct false-positive test. `P` uses an additional visible first-stage
root in the same graph and keeps the recipient G table fixed; it tests
whether a changed-query state can redirect a competent recipient at all.
Repeat with both directions of query/key
movement and multiple offset gaps; use at least two recipient G remaps
with distinct outputs so donor-final-answer injection cannot explain
success.

Also cross the **rank** axis from TEACH-0017: make `rank_A(k_A)` differ
from `rank_R(k_A)` and match `rank_B(k_B)` to a misleading recipient rank
on designated cells. This yields a factorial key × offset × rank design.
Report the actual joint support rather than presenting each factor as
independently balanced if generator constraints prevent it. Add an
absolute-row-slot perturbation and independent row order to reject the
earliest pointer rival.

## Generator and leakage contract for a later implementation

For `A`--`D`, source and recipient must be independently sampled in their
visible graph topology, with matched numbers of rows, token lengths, marker
schedule, answer frequencies and query/key numeric ranges. Preserve the
public strip-and-pair grammar and a unique continuation for every tested
key. Explicitly validate that no donor target can be inferred merely
from a shared final answer, a fixed physical row, a conserved ordinal,
or a memorized group fingerprint. Reject ambiguous outputs by a fixed
pre-outcome rule rather than after inspecting neural behavior.

Freeze generator source, seeds, paired-family partition and discovery/
confirmation assignments before fitting or querying a trained checkpoint.
Audit exact graph/logical/render/stage-family overlap against all exposed
TEACH-0014--17 suites. This is a **new synthetic distribution**, so old
clean competence cannot be assumed; screen every generated input natively
before causal transfer. All donor/recipient clean predictions, intermediate
keys, offsets, ranks, physical row indices and target outputs must be
archived under deterministic IDs. A failed ≥60% or other structural
validity criterion must not trigger adaptive redraw of the same seeds.
Numerical thresholds, denominators, confidence intervals, multiple-test
handling and resource limits require a separate execution registration.

The initial neural site remains the frozen full-state `query.1` hook.
Controls include native identity, same-program rerender, `C` same-query
transfer, `P` same-graph changed-query positive, F0 wrong-key, `B` equal-offset
wrong-key, rank-matched wrong-key, globally distinct donor, norm-matched
random state, reverse restoration and final-answer-state injection.
The **`P` cross-query patch-validity positive control** must pass before a failure
of `A` or `B` is interpreted as evidence against portability: the patch
may be invalid across independently sampled graphs even if each program
is cleanly solved. Sampled CPU replay must reproduce full logits and
intervention vectors from frozen checkpoints. Whole-state success precedes
neuron/SAE/transcoder interpretations; descriptive cosine or CKA cannot
substitute for the collision intervention.

This study calibrates a synthetic graph reader and tests a stronger
functional abstraction. It says nothing directly about whether the
Voynich manuscript encodes graph programs or meanings. The manuscript
claim boundary remains the frozen-decoder and blind-folio protocol in
`docs/research/PROTOCOL.md`.
