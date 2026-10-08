# Crossing structural barriers while preserving the final reading target

Own proposed extension, not implemented, registered, qualified or run. It follows the
completed global-rewrite admission; no earlier source, target, freeze or failure is changed.

## Measured motivation, with a sampling limitation

ADMIT001 checked1,544 deterministic legal-point quantiles on194 archived states. From
the already failed positive regrowth endpoints, only2/504 inspected candidates increased
the logged original target; the corresponding null states had1/264. Initial positives had12/512,
nulls6/264. These are post-outcome exploratory saved-point counts, not the mean acceptance
rate of the full uniform-legal proposal: only up to eight quantiles were inspected per state.
No Gold evaluation or chain was performed. The new move is correct and affordable, but
global coherence alone does not remove a score barrier. A restricted move or wrong objective
can also produce this pattern; it does not prove metastability or that tempering will help.

## Primary method and the distinction from the earlier key search

[Earl and Deem, arXiv physics/0508111v2, section2.1/equations1–2 and section2.3](https://arxiv.org/pdf/physics/0508111)
describes a product ensemble of replicas, corrected exchanges, and the need for overlapping
adjacent energy distributions. Fixed exchange schedules can preserve balance through kernel
composition. Selected theory sections were read directly; their molecular simulation results,
Gaussian energy assumptions and particular optimal acceptance rates do not establish our
discrete corpus behavior. Additional replicas cost computation and do not guarantee useful
transport between modes. This is a primary method review, not a new decipherment precedent.

The earlier TEMPERED-RECOVERY-001 revised full keys using a summed fixed-key likelihood;
it failed its recovery gates. This proposal instead exchanges complete source readings plus
visited dictionaries, with explicit regrowth and global structural changes. It inherits no
success from that earlier campaign and needs a separate complete finite/empirical registration.
The existing source-key/Voynich review covers the differing historical/channel assumptions.

## Choose the auxiliary target explicitly: collapsing and tempering differ

At the cold replica retain T(s)=Q(texts) U^(-m), including the SAME geometric continuation,
EOS, reset contexts and original binary64 source coefficients. Candidate warm replicas have
targets T(s)^(1/k), positive integer k. This deliberately tempers the entire collapsed visited
target. It does not claim a posterior interpretation for the warm replicas or change cold T.

Tempering first in the explicit full-key space and then integrating unvisited rows instead
gives, for beta=1/k and R full rows,

    sum_unused [Q(texts) U^(-R)]^beta
      = Q(texts)^beta U^(R(1-beta)-m).

Up to a state-independent constant this is Q^beta U^(-m), whereas the proposed collapsed-
then-tempered law is Q^beta U^(-beta*m). They differ whenever m changes and beta!=1.
This distinction is our counting derivation; neither auxiliary law is uniquely mandated.
Their cold targets agree. Avoid silently mixing them across merges, splits, regrowth or swaps.

## Keep proposal corrections outside the temperature power

For uniform-valid structural selection, write r=T(new)/T(old), h=V(old)/V(new). At1/k:

    acceptance = min(1, r^(1/k) h) = min(1, [r h^k]^(1/k)).

Raising the entire already-corrected cold ratio r*h to1/k is WRONG when h!=1. The
source-policy q likewise stays a proposal correction, not part of the tempered target.
For regrowth h is the unchanged reverse/forward marginal cut-and-suffix law. Draw the cut
and suffix under the original past-only quantized policy, retain failed draws as self-loops,
then apply r^(1/k)*h. Legacy SourceRegrowth.step cannot simply be reused: its acceptance
is fixed to the cold law. For qualified symmetric label transport h=1.

Exchanging a cold-state s at1/k with state t at1/l, k<l, has ratio

    [T(t)/T(s)]^((l-k)/(k*l)).

Reduce that rational exponent and represent the swap as an integer radical. Check the sign
against a two-state product-target reference; a sign reversal looks numerically plausible.
State-independent fixed kernel mixtures preserve the corresponding replica target, and
compositions preserve the product target even when a whole sweep is not reversible.

## Exact acceptance without clipped exponentials

Our existing exact Bernoulli routine accepts rational probabilities. Positive roots of
rationals can be irrational. A candidate extension can compare a uniform binary stream's
dyadic interval [B/2^b,(B+1)/2^b) against alpha=min(1,(N/D)^(1/k)) using integers:

    accept if (B+1)^k D <= N 2^(b*k);
    reject if B^k D >= N 2^(b*k);
    otherwise append another raw64 block and refine the interval.

For N>=D accept with an explicit recorded raw block, matching the existing receipt policy.
Bound blocks, exponent size, wall/CPU and integer work; a cap aborts rather than rounds.
Equality cases, k=1 parity, probability1, tiny probabilities and endpoints need independent
finite/dyadic reference tests. This is a proposed interval algorithm, not a qualified tool.

## What must precede a real recovery comparison

First qualify radical decisions and complete finite structural/label/regrowth/replica laws,
including altered m, changed path lengths, old-pair rejection, zero eligibility, marginal
cuts, failed draws, wrong powered-selector/q/prior and swap-sign negative controls. Then
measure original-source cost and transport under a fixed ladder; do not choose temperatures
using Gold. Compare cold-only and replica-assisted inference at declared matched resources,
retain all nulls, seal cold endpoints before known-answer metrics, use fresh keys/seeds and
the existing strict exact-record/used-key competence gates. A finite run is not an equilibrium
sample, a mixing certificate or a decipherment. The current temperature1 rewrite admission
continues to be its own completed result, with no retrofitted success criteria.
