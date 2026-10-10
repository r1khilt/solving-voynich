# Repair one occurrence without changing the observed cipher

Preparatory derivation and artificial implementation tests. This move is **not** in
TEMPERED-READING-RECOVERY-001 and cannot change that frozen comparison. No original-source
benchmark, fresh-key recovery, neural training or manuscript decipherment has been run with it.

## Why this is a different inference question

The active global label move preserves equality among source positions. Global pair
merge/split changes all matching pairs or all occurrences of one source row together.
Suffix regrowth can change an occurrence, but must regenerate everything after its cut.
The append-only neural proposal also lacks a direct completed-reading repair action.
These are limitations of the current actions, not evidence that larger networks cannot help.

Our synthetic channel maps each source letter to one fixed one/two-glyph unit and allows
several source letters to share the same unit. Hence identical observed units need not
decode to identical source letters. A single-occurrence repair can split that equality
while preserving literal ciphertext and every existing boundary. It can also remove a
now-unused binding. This specifically targets one obstruction left by pure label search;
it does not fix wrong segmentation or guarantee a useful source score.

Primary review: [Ravi and Knight2011, selected section3.1, PDF pages6–7](https://aclanthology.org/P11-1025.pdf)
argue for coordinated type updates because their main channel has a single plaintext
letter per ciphertext type. They also discuss point updates and word-boundary moves.
Our duplicate-allowing deterministic **enciphering** direction differs: a ciphertext unit
can have several legal source letters. Their rejection of inconsistent token assignments
does not forbid those assignments here. We do not adopt their cache/exchangeability,
English dictionary, sparse channel prior, annealing or reported recovery performance.
Prior [contextual whole-key row Gibbs](contextual-row-gibbs-2026-10-01.md) instead changes
a dictionary entry while marginalizing all compatible source paths. It is not this move.
The [prior Voynich/source-key review](blind-channel-fresh-qualification-review.md) remains
the historical-method context; none establishes that this synthetic channel generated Voynich.

## Exact symmetric component law

Let s be a complete visited-only reading with texts x, bindings K, N total source
positions and target T(s)=Q(x) U^(-m(s)). Here Q includes the unchanged source,
continuation and end-of-record factors. Choose a source occurrence i uniformly from N.
If its current row is a and observed unit is u=K[a], define

    C(s,i) = {b : K[b] is unbound or K[b]=u}.

Choose b uniformly from C, including a. Replace only x[i]=a by b, bind b to u if
needed, and release a only if it has no other occurrence anywhere in any record.
All other bindings and positions remain unchanged. Source length N and every unit
boundary remain fixed, so direct literal reconstruction must still equal the observation.

The compatible set is identical in the reverse state: b changes from unbound to u
at most, and a changes from u to unbound at most. Both statuses are members of C.
No incompatible binding changes. Therefore the forward and reverse component
probabilities are both 1/(N |C|). Replacing that same position by a recovers the
entire original state, including released bindings. Nonidentity transitions change
one unique position; identity probability is the sum of the identity components.
The MH acceptance for inverse temperature1/k is consequently

    min(1, [T(s')/T(s)]^(1/k)).

There is no legal-list size correction, source-reference action probability or cut
probability in this proposal. If m changes, U^(-delta_m) remains part of the target,
inside its temperature power. Dropping it would target another distribution. The
new scorer rescans complete source contexts and checks accepted points against the
qualified forced reference path. Integer radical decisions use the existing bounded
law; cap aborts are failures, not ordinary MH rejections or convergence certificates.

This is **not** an ergodicity proof. If every row is used with a distinct incompatible
unit, each C can contain only its current row and the entire kernel is identity.
Even nontrivial components preserve boundaries and cannot cross all decoding states.
It belongs in a separately qualified mixture with structural/global moves. A learned
position/row selector would generally need its actual reverse probability; the uniform
symmetry proof would no longer suffice.

## Preparation and limits

New reading_occurrence.py provides literal candidate construction, explicit component
probabilities and a warm MH wrapper; it does not modify the active controller. Eight
artificial tests passed in0.63seconds: six complete tiny panels independently rebuild
bindings from observed slices, enumerate every candidate, verify reverse support and
exact rational cold detailed balance, normalization and stationarity. They include
binding births/deaths, cross-record repeated source positions and a fully frozen
all-rows-used witness. A seeded80step warm fixture checks all four existing temperatures,
exact candidate source targets and accepted reference replay. These are preparation
tests, **not** a registered source admission, independent expert replication, recovery
result or proof of warm full-kernel stationarity. Initial scoped lint found one unused
test import, removed before any scientific use.

Before adding this to an empirical method: freeze a finite qualification with an
alternate map/source-target audit, omission-of-prior negative control and warm radical
flux checks; then measure original-source proposal/replay cost under explicit caps.
Only a new sealed fresh-key/control comparison can determine whether the extra action
improves inference. The current132cell comparison is unchanged and cannot be rerun
with a revised move mixture under its original registration.
