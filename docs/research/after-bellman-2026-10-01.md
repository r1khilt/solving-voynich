# After one-step lookahead: preserve a dictionary across future observations

2026-10-01. **Own post-outcome mathematical analysis and proposed next method.** The [registered one-step experiment](../experiments/SOURCE-BELLMAN-001-results.md) failed its improvement criterion. Nothing here changes that result or its threshold. No new whole-text recovery experiment, neural fit or manuscript scoring is reported.

## The specific approximation to investigate

The original law samples a dictionary once and reuses each row throughout all records. The cheap remaining-text guide instead redraws an unassigned row's unit per future occurrence, and multiplies separately computed passage likelihoods. It retains bindings already made, but drops dependencies induced by future reuse of currently unknown rows. One Bellman step restores only the next action and its resulting binding; its tail still makes this approximation.

There are other mismatches: root-context rather than full Markov probabilities in the tail, smoothing, finite pruning and fixed-key Viterbi decisions. The current experiment does not isolate dictionary reuse as the dominant empirical cause. The following calculation isolates that approximation mathematically.

## Two one-glyph remainders: an exact covariance formula

Condition on a supplied partial dictionary and source contexts. Assume independent uniform remaining rows over U possible units, which can include two-glyph units. Two unfinished records each have exactly one observed glyph remaining. Only one source letter emitting that one-glyph unit can finish each record; there are no zero-length emissions. Let p_A,r and p_B,r be their context-specific next-source-letter probabilities. For a glyph u define the random compatible source mass

`W_A,u = sum_r p_A,r * 1(K_r = u)`.

Known rows contribute constants. Write F for free rows, and

`mu_A,u = sum_known p_A,r * 1(K_r=u) + sum_free p_A,r/U`.

Independence between different dictionary rows gives exactly

`Cov(W_A,u, W_B,v) = sum_free p_A,r*p_B,r * [1(u=v)/U - 1/U^2]`.

The full future completion mass is

`H = rho^2*(1-rho)^2 * [mu_A,u*mu_B,v + Cov(W_A,u,W_B,v)]`.

This is conditional future mass, so previously paid binding priors are not paid again. The geometric factor includes a continuation and EOS for each unfinished record. It is not a formula for arbitrary-length future text.

With two source rows having probabilities 1/3 and 2/3, six possible units, both rows free and rho 1/4:

| Remainder glyphs | Exact future mass | Independent mean-product mass | Exact / independent |
| --- | ---: | ---: | ---: |
| Same glyph |17/4608|1/1024|34/9|
| Different glyphs |1/2304|1/1024|4/9|

Thus the independence approximation can underestimate same-glyph completion by about 3.78 times and overestimate different-glyph completion by 2.25 times in this example. No lower/upper-bound interpretation is available. [Rational checker](../../scripts/check_shared_key_moments001.py) compares 1764 context/partial-key/observation combinations with explicit full dictionaries and source-letter pairs; [receipt](../../results/SOURCE-BELLMAN-001/shared-key-moments-post-outcome.json).

This recovers a precise version of the user's similarity intuition: overlap between probability profiles on still-unknown rows matters. The mathematical coefficient is a weighted **dot product**, not cosine similarity; normalization would discard how much uncertain source mass remains. This concerns compatibility under a stipulated code law. It does not establish word meaning, vocabulary clusters or a neural global workspace.

## Three remainders: pair corrections are insufficient

For three one-glyph records, expand around their conditional means:

`E[W1*W2*W3] = mu1*mu2*mu3 + mu1*Cov23 + mu2*Cov13 + mu3*Cov12 + kappa123`.

The remaining third centered moment is exactly

`kappa123 = sum_free p1,r*p2,r*p3,r * [1(u1=u2=u3)/U - (1(u1=u2)+1(u1=u3)+1(u2=u3))/U^2 + 2/U^3]`.

Only a shared free row can contribute: distinct rows are independent and centered singleton factors vanish. Multiply the entire expectation by `rho^3*(1-rho)^3` for future mass. Known rows affect means but not the centered corrections.

For three equal glyphs using the same numerical fixture, exact mass is 1/2048. Mean-product plus all three pair corrections gives 7/24576; the missing third contribution is 5/24576. Exact mass is 12/7 times the pair-only value. This is an explicit counterexample to an exact pair-only correction even before long histories or context changes enter.

[Separate rational checker](../../scripts/check_shared_key_third_moments001.py) passed 1512 cases: 27 ordered context triples, seven stated partial dictionaries and eight glyph triples. It enumerates every compatible full dictionary and source-letter triple independently of the moment expression. Pair-only differs from exact in 1080 of these specified cases; that count is not a frequency estimate for actual search states. [Receipt](../../results/SOURCE-BELLMAN-001/shared-key-third-moments-post-outcome.json). All arithmetic is exact Fraction arithmetic, not floating tolerance.

## General dependence and a better surrogate family

For m one-glyph records, condition on source-row choices r1..rm. If a known row disagrees with any assigned observed glyph, that tuple has zero mass. If a free row is reused with two different observed glyphs, it also has zero mass. Otherwise the dictionary compatibility factor is exactly `U^(-d)`, where d counts distinct free rows used by the tuple. Summing source probability products times this factor, and multiplying by `rho^m*(1-rho)^m`, gives exact future mass. This is the same once-binding principle as the original channel, not a new historical hypothesis. Higher-order reuse is why pairwise similarity summaries cannot preserve the entire law.

A prospective whole-text surrogate should preserve one sampled completion of the dictionary across **both records and all future occurrences**. For example, average conditional remaining-text IID likelihoods over a fixed set of complete dictionaries drawn conditional on already assigned rows. Each fixed dictionary supports ordinary variable-unit dynamic programming. The finite-sample mean is unbiased for that specified shared-dictionary IID surrogate if sampling and conditional priors are correct; it is not unbiased for the full context-dependent source model, and its log is not unbiased. Rare compatibility can produce many zeros and huge variance. Stable sampling across candidate states also needs an explicit coupling rule; independent redraws can scramble rankings.

A stronger version substitutes context-dependent fixed-key future likelihoods, retaining the current source context. Averaging over complete remaining dictionaries then targets the actual conditional future mass, but each fixed-key computation can be expensive. Monte Carlo does not remove the exponential rarity problem; bounded tiny enumeration must first measure error and zero-support behavior. No implementation or training of either proposed guide is queued by this memo.

Next registration should distinguish shared-dictionary IID from the current occurrence-redrawing IID, validate multi-glyph exact tiny futures, include incompatible observations and repeated-source-row controls, bound dictionary samples/table work/memory, and compare matched compute before new-key qualification. A learned future critic is justified only after a competent target and costs are established. Mechanistic claims would then require selective binding/context/cross-record interventions with matched controls and disjoint recovery evidence. Current toy identities alone do not qualify a decoder, causal circuit or Voynich interpretation.

## Provenance and failure record

These are post-outcome own derivations following the primary-source review in [the original method memo](source-bellman-2026-10-01.md). They are not attributed to those papers and are not empirical evidence at their frontier scales. The first two-record checker encountered a Python empty-sum type-promotion error: integer0division produced float 0 in a fully bound case. Explicit Fraction zero initializers fixed the checker before its first successful saved receipt. The identity was not changed and no source-panel call was repeated. The three-record checker completed its first invocation without a failure. Total new empirical panel calls, training and paid spend: 0.
