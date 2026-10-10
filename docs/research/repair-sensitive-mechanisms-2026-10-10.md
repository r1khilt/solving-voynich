# Interpret mechanisms that repair a reading, not a supplied dictionary

Derived analysis while the frozen fresh-key replica comparison runs. No neural measurement,
fit, new intervention, manuscript scoring or new recovery result is produced by this note.
It leaves every active search input and success criterion unchanged.

## Source review and prior evidence

Refreshed primary [Gurnee et al., methods, J-space definition, intervention methods and limitations](https://transformer-circuits.pub/2026/workspace/index.html).
Their lens averages downstream sensitivity across contexts/future outputs; their J-space is
a sparse nonnegative token frame, not merely the leading singular vectors of one Jacobian.
They test causal reuse beyond immediate readout and caution that the lens is incomplete.
These claims concern their studied language models, not our decoder. The official
[companion repository](https://github.com/anthropics/jacobian-lens) was inspected at README
level; no code was installed or executed. Selected sections were refreshed, not a full replication.

The published local [JSPACE-0001 result](../experiments/JSPACE-0001-results.md) failed its
causal gate despite stable country directions. [TEACH-0025](../experiments/TEACH-0025-results.md)
reports replicated synthetic hard-first retrieval repair, but depends on supplied public
row structure and task markers. Those are concrete methodological precedents and limits,
not a decipherment or evidence of an equivalent circuit in the interrupted96M model.
The [previous behavioral geometry note](behavioral-latent-geometry-2026-09-30.md) already
derives multi-step Fisher geometry and distinguishes probing from generated-sequence Fisher.
The following is our additional application to literal coding-state constraints.

## A dictionary readout can be an architectural artifact

Code inspection: SourceActionProposal packs the current23past bindings into explicit
row/unit-pair embeddings, averages them into every decoder query, and adds previous action,
selected record and offset. A separate symbolic environment supplies the legal-action mask.
This is a sensible causal proposal interface: no future/unused binding is supplied. But a
probe that recovers those past bindings may be reading the architecture's input. Likewise,
perfect literal compatibility may come entirely from the hard mask. Neither demonstrates
that the neural encoder inferred the unknown key from ciphertext.

The model outputs46append actions. A complete state has no append action. Consequently
this architecture cannot directly edit a wrong completed dictionary; its controller must
discard/regrow an earlier suffix. This is an action-space limitation, independent of model
size or whether hidden activations contain useful information. A future model specifically
proposing global repairs would be a different architecture/interface, not a larger instance
of the same teacher-forced append policy.

## Mask changes are not learned geometry

Let L(s) be the symbolic legal set and z(h) the46finite pre-mask logits. For fixed L, the
legal action distribution is p=softmax(z_L). Its local sensitivity is

    F_L(h) = J_L(h)^T [diag(p)-p p^T] J_L(h),
    rank(F_L) ≤ |L|-1 ≤45.

This rank ceiling follows from the readout and softmax, regardless of what the model learned.
It does not identify a conceptual workspace. With one remaining legal action F_L is zero:
the mask completely dictates the next action even if h stores rich, irrelevant information.

Comparing two symbolic binding states can change L. A positive action in one state may
have zero probability in the other, giving infinite directed KL. That discontinuity can
arise from the constraint program alone. Report legal-set overlap/cardinality separately
from learned sensitivity. If using common-support renormalized distributions, also report
the mass discarded on each side; conditioning away that mass changes the question. An
unmasked softmax comparison is another diagnostic, not the literal decoder's action law.

For a hidden intervention with the symbolic state/mask fixed, output changes can be
attributed to downstream neural computation rather than a changed mask. But the test must
evaluate a meaningful legal behavior, retain wrong-direction/norm/site controls and account
for paths that later overwrite the patch. Swapping an explicit binding while keeping its
already-consumed ciphertext inconsistent is not an on-manifold dictionary counterfactual.

## Future behavior should expose uncertainty, not just known bindings

Proposed targets include which unseen source row will bind a repeated unit, whether two
records demand one coherent interpretation, and which earlier commitment needs revision.
These are inference questions the model must compute, rather than simply copy from s.key.
Use disjoint synthetic keys and eligible counterfactuals with literal validity checked.
Counterfactual cases must state whether observations, prefixes, legal sets, source context
and target length change. A side-channel unit query is not equivalent to successful reading.

For multiple fixed recipients/continuations c and future steps t, define a candidate
behavior operator G=E_c sum_t J_c,t^T C(p_c,t) J_c,t. Stacking recipient effects and then
forming this quadratic operator preserves context-specific sensitivities that can cancel
if Jacobians are averaged first. These are different summaries. A direction with a large
G response can cause generic instability rather than correct repair; finite interventions
and held-out recovery remain the deciding evidence. No universal neuron basis follows.

The averaging distribution matters particularly here: truth-prefix teacher forcing and
the decoder's erroneous-prefix rollout visit different states. Geometry estimated on only
correct prefixes does not diagnose the free-running failure. Stratify first binding/reuse,
current inventory, ambiguity and failure-prefix stage; fit directions on discovery only;
test finite transfer and selective necessity/rescue on separate recipient data and seeds.
Character/action directions are not word meanings, and cosine alone does not fix this.

## Candidate architecture: learn a repair policy over an exact symbolic world

Speculation, not implemented or registered. Give an observation encoder and a candidate
reading/dictionary encoder a shared iterative state. A learned proposer chooses among
globally coherent, independently validated rewrites. A source critic scores complete
consequences; a consistency module preserves literal observations. This explicitly exposes
revision actions absent from the current append decoder and trains on erroneous hypotheses
as well as true prefixes. It is closer to a world/action model of a coding system than a
model that merely forecasts the next glyph.

For a partial-involution repair f_i and a positive quantized state-dependent selector q(i|s),
the warm acceptance would be T(f_i(s))^beta/T(s)^beta ×q(i|f_i(s))/q(i|s), clipped at1.
Do not silently use the uniform-eligible correction after replacing uniform selection with
a neural policy. Reverse support, identity mass, quantization, block caps and cost need
new qualification. A learned critic must not replace the audited final target without a
new model/identifiability registration. Hidden states could then be tested for causal
representation of a specific repair variable, not named a global workspace by analogy.

Such a model is justified only if the current comparison establishes what search misses
and the post-seal Gold scores/orbit diagnosis distinguish useful source guidance from a
bad objective. More capacity by itself would not address the interface/mask/exposure issues.
If it does become justified, compare a learned proposer against the same fixed target,
symbolic moves, work budget and nulls; demand actual exact text/used-key competence before
calling its internal features useful for decipherment.

## What finite qualification can prove

Own deduction under ideal independent random draws, positive source coefficients and
positive root-regrowth mass: every complete reading has a unique scheduled action path
with positive quantized proposal probability. Root regrowth therefore connects every pair
of complete states with positive MH acceptance, including a positive self-loop. The ideal
cold finite-state chain is irreducible and aperiodic. This establishes eventual asymptotic
behavior, not a practical hitting time; the minimum root probability can be exponentially
tiny in path length. The seeded PCG64 run is a reproducible simulation, not a proof about
the infinite ideal random stream.

The implemented radical sampler additionally has a finite abort cap. A cap terminates
the experiment; it does not round an acceptance or retain a state as a rejection. Thus
the finite flux qualification describes the intended non-aborting law, not a stationary
process over a new absorbing failure state. For an ideal uniform binary stream, at most
one1024-bit interval straddles a given nontrivial threshold, so the16-block uncertainty
event has probability at most2^-1024 per decision. Integer-work caps are separate and
can deterministically abort a large point. Actual absence of aborts and exact replay do
not turn a short optimizer run into an equilibrium or historical decipherment certificate.
