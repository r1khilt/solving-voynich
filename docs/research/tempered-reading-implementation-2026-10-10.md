# Complete-reading replica exploration

This implements the separately recorded [proposal](tempered-structural-reading-proposal-2026-10-08.md).
It does not resume the interrupted neural campaign or change any earlier scientific freeze.
Implementation, mathematical qualification, original-source admission, and recovery are separate stages.

The prior [source-key review](source-key-revision-2026-10-01.md) distinguishes Hauer et al.'s
substitution-decipherment setting and Hauer/Kondrak's Voynich anagram assumptions from our
persistent one/two-glyph dictionary hypothesis. None establishes that our channel explains
the manuscript. The earlier full-key tempered recovery failed; it is not evidence that the
new complete-reading search works. We keep that failure and the structural inventory diagnosis.

The primary methodological review is [Earl and Deem, physics/0508111v2](https://arxiv.org/pdf/physics/0508111),
selected section 2.1 equations 1–2 and section 2.3. It motivates a product ensemble and corrected
exchange kernels, with adjacent overlap a practical concern. Molecular results, Gaussian
energies and proposed optimal acceptance rates do not transfer automatically. The arithmetic
and collapsed-target distinctions below are our derivations, not claims established by that paper.

For each complete reading s, the original collapsed cold target is T(s)=Q(s) U^(-m(s)),
including unchanged source resets, geometric continuation, EOS and dyadic binary64 coefficients.
The ladder is k=(1,2,4,16), with auxiliary law T(s)^(1/k). Collapse then temper differs from
tempering full keys then summing unused rows whenever m changes; the cold law agrees.

Fixed local mixture: half past-only quantized suffix regrowth, quarter uniform-eligible global
pair rewrite, quarter used-first symmetric label transport. Every proposal correction h remains
outside the root: acceptance=min(1,(r h^k)^(1/k)), r=T(new)/T(old). For pairs h=V(old)/V(new).
For regrowth h is the reverse/forward marginal cut-and-suffix probability; failed suffixes are
self-loops. Zero eligible pairs are self-loops. Deterministic candidates score full source terms
in the normalized-consumption action order; accepted points reconstruct the full reference
path/counts. Exchange adjacent k<l with ratio r^((l-k)/(kl)). Local updates followed by a
fixed alternating adjacent exchange schedule preserve the product law; the whole sweep need
not satisfy detailed balance. No adaptive ladder, Gold-based selection or annealed final target.

The new radical Bernoulli sampler refines a raw64 PCG64 dyadic interval and compares integer
powers. It never clips probabilities or uses a floating exponential for a decision. At k=1
its decision and consumed RNG stream match the existing rational Bernoulli sampler. A block
cap or integer-work cap aborts rather than selecting an approximate decision. Limit 16 blocks,
degree 256, integer-work bound 4,194,304 bits; allocations are bounded before powers/products.
These generous mathematical bounds are not an authorization for an unbounded empirical run.

The [finite registration](../experiments/TEMPERED-READING-THEORY-001.md) uses independent
terminal-policy enumeration, direct complete-state targets and alternate proposal laws. Each
positive edge has exact rational equality after raising its flux to the positive integer k;
injectivity of positive powers proves equality of the original real-valued flux. Row construction
retains rejection mass and normalized proposal laws. Separate 100-digit Decimal calculations
check row sums/stationarity below 1e-75; these diagnostics alone are not exact certificates.
An independent integer-root quantile oracle checks radical decisions and the consumed stream.
A 27-state perfect-power product ensemble checks full two-sweep stationarity with exact Fractions.
Wrongly powering h, using the other warm prior, and reversing swap direction must fail.

Preparation fixtures exercise a 39-state artificial panel, k=1 legacy parity, both replica
sweep parities, all local kernels and full replay of retained paths. A temporary test-only
aliasing defect (`changed = kernels = set()`) was fixed before registration; it did not affect
the search module or produce a scientific result. No original source, known-answer metrics,
neural inference, manuscript holdout or recovery was used for these fixtures.

Correct mathematics does not prove mixing or identify a good target. Warm readings can be
less informative, swaps can fail for poor overlap, and the cold model can prefer incorrect
readings. Next requires original-source cost/RNG/literal/target replay, followed by a separately
registered comparison on fresh synthetic keys/nulls with every endpoint sealed before Gold.
Only exact-record and complete-used-key recovery establish synthetic competence; neither
would establish historical decipherment or a causal neural circuit.
