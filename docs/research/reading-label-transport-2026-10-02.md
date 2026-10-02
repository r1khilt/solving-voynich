# Coordinated source-letter reassignment in a revisable reading

Exploratory adaptation of the existing [joint swap derivation](auxiliary-key-swap-2026-10-01.md).
This is not a new MCMC theorem, neural architecture or identified circuit. No original-source
label experiment has run. The sustained recovery and original GPU training remain frozen.

## Motivation and source review

The completed recovery producer reports some improvement from suffix regrowth, but neither
seed recovers a complete known-answer record or used dictionary. Its registered full audit is
running. A local suffix change often keeps the early labels of repeated codes. A different
operation can revise those labels coherently throughout all records without throwing away
the segmentation. Whether that operation improves inference needs an empirical comparison.

[Neklyudov et al. (2020), section 2 and section 3.1 Tricks 1–2](https://proceedings.mlr.press/v119/neklyudov20a/neklyudov20a.pdf)
provides the extended-state involution framework and correct state-dependent selection
factor. It separates invariance from convergence; auxiliary dimensions can impede mixing.
This review reread the selected derivations, not its complete experiments or appendix proofs.
Our transformation is discrete, so there is no continuous Jacobian computation.

[Chiang et al. (2010), sections 3.2–3.4 and 4](https://aclanthology.org/N10-1068.pdf)
reviews local and whole-derivation revision and approximate proposal lattices. Its method
discusses, then skips, an MH correction under an empirical acceptance observation. We retain
our exact correction and do not transfer its accuracy or run-selection claim. Full-history
dictionary dependence prevents a source-order-only shortcut.

The [source revision review](source-key-revision-2026-10-01.md) covers Hauer, Hayward and
Kondrak's key modification for substitution ciphers, and prior Voynich hypotheses. Our
duplicate one/two-glyph units differ from a substitution permutation. Nothing here establishes
Latin, a fixed channel, or plaintext meaning in the manuscript.

## Joint target and a partial dictionary

The unchanged terminal positive potential for a literal complete state s=(K,X) is

    T(s) = U^(-m) ∏records [ρ (1−ρ)^|x| ∏letters p(letter | complete source context)].

Only the m visited rows are bound; others are −1. U=42 and ρ=1/225 on the original panel.
The source coefficients remain their original binary64 values; the exact-integer arithmetic
represents that positive potential rather than silently renormalizing the coefficients.

For distinct rows a,b, transpose their dictionary entries, including −1, and transpose those
letters everywhere in all source texts and in every action. Offsets, each action's unit length,
record resets, literal ciphertext, and the multiset of bound codes are unchanged. The same
operation twice restores the entire state. Duplicate codes remain distinct rows: swapping
their labels can change the reading and source potential even when the key tuple is unchanged.

The visited count and source lengths stay fixed. Thus all prior and EOS factors cancel in
the ratio T(s')/T(s), but the implementation independently constructs the complete target
integers for comparison with the existing reference path. Full transformed contexts must be
rescored, including subsequent positions whose emitted letter did not change.

## State-dependent pair selection that remains symmetric

Select a uniformly random visited row, then a uniformly random different row among all R−1
others. Sort the pair. Its marginal unordered probability is

    w_s({a,b}) = [1(a visited)+1(b visited)] / [m(R−1)].

An unused/unused pair has probability zero; visited/unused has one orientation, and
visited/visited has two. The transformation preserves m and the number of visited rows in
the selected pair, so w_s({a,b})=w_s'({a,b}). The forward sampled orientation is not a
separate fixed auxiliary orientation for acceptance: after exchanging visited and unused
rows that particular orientation could have zero reverse probability. Marginalizing the
two orientations is essential to this implemented transition law.

The label acceptance is therefore min(1,T(s')/T(s)). Exact integer targets and the existing
bounded lazy raw64 Bernoulli avoid rounding tiny ratios to zero. Reaching the bit-work cap
aborts; it does not approximate acceptance. PCG64 is required. No Gold, neural network or
future source text enters pair selection.

## Reference policy counts are bookkeeping, not a selection density

The deterministic label candidate is not sampled from the source action policy q. Its
acceptance must not acquire q(s)/q(s'). A finite negative control demonstrates that this
extra factor violates the intended target's detailed balance.

After acceptance, the new actions are replayed through the fixed source policy to reconstruct
counts needed by a later suffix-regrowth correction. That reference q is not the probability
law of the retained Markov-chain state. The target-only scorer avoids that replay for rejected
candidates. Source terms must be saved in action order: normalized-consumption scheduling
can interleave records, so concatenating per-record terms gives the same target product but
wrong reference bookkeeping. The interleaved-record fixture checks the actual order.

## Combining complementary moves

Labels alone cannot change the encoding inventory, source lengths, glyph boundaries, or the
equality pattern of repeated source letters. They cannot solve examples whose generating
reading needs those changes. Regrowth can change those features, while coordinated labels
can revise long-established assignments. A state-independent mixture of the two invariant
reversible kernels is reversible for the same target. Their sequential composition preserves
the target but need not be reversible. The finite checker tests both statements exactly.
No production mixture controller is implemented here, and neither invariance nor positive
support supplies a useful finite-time recovery guarantee.

An own reachability lemma makes the limitation testable. Consider two complete literal
readings of the SAME records with visited-only dictionaries. A global label permutation
maps the first exactly to the second if and only if (1) each record has the same number
of source letters; (2) each corresponding position emits a unit of the same glyph length;
and (3) the cross-record equality pattern of source letters is identical. Condition3 means
two positions carry the same source letter in one reading exactly when they do in the
other, including positions in different records.

Necessity follows because permutation changes labels, not unit lengths or equality. For
sufficiency, define the map from each used old row to the other reading's row at any one
of its occurrences. Equality-pattern agreement makes that map well-defined and injective,
with the same number of used rows on both sides. Equal per-position unit lengths give the
same ciphertext boundaries; literal re-encoding then gives the same unit at every mapped
row. Extend this used-row bijection to a permutation of all rows; unused entries are−1 in
both collapsed keys, so that extension also maps the dictionaries. Transpositions generate
the permutation. This proves orbit membership, not that a finite sampler finds it.

Thus code-inventory coverage alone is an insufficient oracle diagnostic: even the right
multiset of codes may coexist with the wrong segmentation or repeated-letter partition.
After candidate sealing, known answers can quantify these separate obstructions without
being supplied to proposals. Per-record source-length discrepancy also supplies an edit
lower bound that labels cannot remove. No such original-panel oracle measurement has yet
been made; this is a derived criterion for the planned diagnostic.

Before a new recovery run, a separately frozen development diagnostic should examine every
unordered pair at initial and both-arm/both-seed endpoints, preserving all null controls.
The planned panel is 97×5 states×253 pairs=122705 deterministic transformations. It should
separate remaining uphill target moves from actual known-answer effects, and measure
whether any label permutation could in principle repair the current segmentation/equality
pattern. Existence of a better neighboring score does not show semantic recovery; no best
Gold-selected candidate may become a reported solver output. This is a proposal, not a
registered or launched empirical workload. Actual bounds and input hashes must precede it.

## Preparation qualification and limitations

Independent exact terminal enumeration uses two and three rows, binary glyphs, two declared
dyadic sources, stop1/3, grid2^8, bias1/6, and root-cut mass1/8. Eight panels reconstruct
proposal counts independently, check every transposition and inverse, independently score
the target, and check all flux/stationarity equations for the label kernel, regrowth, fixed
half mixture, and composition. Counters cover unbound swaps, duplicate-code changes,
context effects at unchanged actions, wrong-policy-ratio failures, and nonreversible
composition witnesses. Runtime fixtures check PCG replay, future regrowth counts, foreign
instance rejection, literal truncation, and interleaved term order.

Preparation failures were corrected before scientific use: the first panel had no unused
row for its asserted witness; a symmetric source hid the asserted context effect. Those
were fixture failures, not empirical results. The scorer's initial per-record term order
was an implementation defect caught and corrected with an interleaved-record fixture.

The separate [finite registration](../experiments/READING-LABEL-TRANSPORT-THEORY-001.md)
fixes one scientific checker and one receipt closure after exact remote publication. Small
finite correctness does not establish original-scale cost, mixing, decipherment or a neural
mechanism. The old full-key involution and new inventory regrowth remain distinct prior work.
