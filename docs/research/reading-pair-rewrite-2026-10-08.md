# Global structural revision with exact selection correction

2026-10-08. Own derived adaptation for the restricted synthetic channel. The manuscript
remains unsolved. This repairs an inference limitation measured on exposed synthetic
readings; it is not evidence for Latin, manuscript ligatures, semantics or neural circuits.

## Evidence motivating the change

The audited [landscape](../experiments/READING-LABEL-LANDSCAPE-001-results.md) found all
320 positive states outside the generating reading's global-label orbit. Every tested
state also had an insufficient unit inventory for complete used-dictionary recovery.
Local-regrowth endpoints allowed at most about34% correct row bindings under arbitrary
relabeling. Renaming a state cannot repair its source length, segmentation or partition
of positions into repeated letters. A different structural operator is needed.

The [earlier proposal](global-pair-rewrite-proposal-2026-10-02.md) gives the global
distinct-triplet partial involution. This implementation retains its literal conditions.
Every adjacent b,c is replaced by a when a is unbound and b/c have one-glyph units;
the inverse splits every a when its two-glyph unit is compatible with b/c and no old
b,c adjacency exists. Distinct letters prevent overlapping pair matches. Release only
rows that become unused. Rebuild the normalized-consumption action schedule from the
complete transformed texts, then rescore all source contexts and reconstruct policy counts.

## Primary review and what is our own derivation

[Neklyudov et al., ICML2020 main text, section2 and Tricks2–3](https://proceedings.mlr.press/v119/neklyudov20a/neklyudov20a.pdf)
provides the auxiliary-map/involution framework, including reverse selection probabilities.
Selected main-text derivations were reviewed earlier and the primary publication refreshed
this turn. The separately referenced supplementary B.5 was not reviewed. No continuous
flow performance or experimental acceleration from that paper is imported here.

[Wingate et al., author Revision3, February8,2014, section2–2.1/Algorithm2](https://web.stanford.edu/~ngoodman/papers/lightweight-mcmc-aistats2011.pdf)
requires selection and fresh/stale corrections for its stochastic trace proposal. Algorithm2
and the supporting text were refreshed directly this turn. This differs from a deterministic
counting-measure rewrite: our reverse selection law must be derived for this map rather than
copying generic trace factors. No continuous Jacobian enters this discrete bijection.

The [prior source-key review](source-key-revision-2026-10-01.md) covers Hauer et al.'s2014
substitution key modification, the unaccepted2016 Voynich/anagram interpretation, and the
shared-dictionary obstacle to naive local particle splices. Their permutation channel and
language supervision differ from our duplicate variable-length units. This is an inference
operator for an existing artificial model, not a new historical encoding hypothesis.

## Two selectors with different acceptance rules

Let f_i denote the partial involution for ordered distinct triplet i. The unchanged target
is T(s)=Q(texts) U^(-m), with the original positive binary64 source potential, independent
reset contexts, geometric continuation/EOS and m visited rows. Source coefficients are
not renormalized. Full forced replay is used initially, favoring correctness over speed.

For a uniform rank among all R(R−1)(R−2) triplets, i has the same probability forward
and backward. The ratio is T(f_i(s))/T(s); invalid i gives a self-loop. An explicit rank
bijection supplies this law without state-dependent rejection sampling.

For a uniform rank only among V(s) legal nonidentity triplets, f_i(f_i(s))=s ensures that
the same i is legal in reverse. The auxiliary proposal ratio is

    [1/V(new)]/[1/V(old)] = V(old)/V(new).
    acceptance = min(1, T(new)/T(old) * V(old)/V(new)).

Each augmented triplet edge satisfies detailed balance; summing over multiple triplets
with the same endpoint preserves it. A zero-degree state is a self-loop. Omitting V(old)/
V(new) generally breaks balance. This ratio and the efficient eligibility construction
are our derivation, prospectively qualified by the new finite record below.

Eligibility is computed from the current reading's source-row adjacencies and binding inventory:
merges combine one-glyph adjacent rows with each free row; splits combine each two-glyph
row with every compatible single/free pair, rejecting existing adjacencies. The resulting
sorted list is compared against independently testing every triplet. Neither selector uses
Gold, old future bindings, a neural model or action-policy q in acceptance. Those q counts
are reconstructed only to allow later regrowth moves under their separate correction.

## Qualification, limits and next experiment

[READING-PAIR-REWRITE-THEORY-001](../experiments/READING-PAIR-REWRITE-THEORY-001.md) checks
all complete states and all triplets in twelve short binary multi-record panels. Its direct
recordwise enumeration, second rewrite implementation, fractional scheduler and per-record
target arithmetic are separate from the production logic. It checks both selectors' full
stationarity and row normalization, every triplet's flux and involution, exact ratios,
eligibility, reference-path reconstruction and negative controls. A second invocation only
closes receipt/hash/counter arithmetic; it does not re-enumerate or provide expert review.

Preparation: fourteen focused tests pass, including repeated replacements across records,
released/retained single rows, duplicate glyph codes, existing-pair rejection, record-boundary
separation, foreign/incomplete paths, seed identity, zero-degree behavior and selector factors.
One larger artificial boundary panel was inspected before scientific freezing, not as recovery.
No source archive, manuscript, learned weights or exposed Gold is opened for this qualification.

Correctness does not imply useful acceptance, fast mixing, irreducibility or recovery. Some
complete states have no legal rewrite. The distinct-row map cannot directly merge an identical
letter pair. Source contexts may make coherent global changes very improbable. A fixed mixture
with regrowth/labels still needs its own bounded cost and known-answer experiment. The valid
selector avoids invalid proposals, but its actual original-source speed has not been measured.
The next step is cost admission with frozen source/target and retained nulls, followed by a
separately registered recovery comparison whose candidates are sealed before Gold evaluation.
