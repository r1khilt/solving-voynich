# Observation-conditioned inventory counting

Own exploratory mathematics developed while the fixed COORDINATED-DICTIONARY-001 run was underway; its shorter-case survival statuses had been observed. No registered code, criteria, seeds or running experiment were changed. These are finite exact counting checks and a proposed successor, not another sampler run or a decipherment claim.

## Why guaranteeing every single-glyph code can be inefficient

The registered all-single-code event A has prior probability p=0.003927233618484433 under23 independent42-code rows. A32-particle prior bank contains0.12567 covered keys in expectation; the half-covered proposal contains16.06284. This is a substantial redistribution, and correct importance weights can undo much of it.

Own arithmetic for the mixture q=π/2+π(·|A)/2 gives E_q[(π/q)²]=2/(1+p). Ignoring any observation, the usual large-population importance ESS fraction is (1+p)/2≈0.501964. This is not the finite realized ESS and not an estimate of posterior coverage.

For any nonnegative likelihood h, the difference between the mixture and prior second moments is:

E_q[(πh/q)²]−E_π[h²] = (1−p) [ E_π(h²|Aᶜ) − p/(1+p) E_π(h²|A) ].

Thus the mixture reduces this particular importance-weight variance only when the covered event's conditional h² expectation exceeds the uncovered event's by at least(1+p)/p≈255.632. Literal support alone does not guarantee that. This calculation does not predict the registered resample–move panel's outcome or justify changing it midway.

## Condition on the actual first observation support

With strictly positive source rows, valid transitions, nonempty units and0<ρ<1, a fixed dictionary has positive prefix likelihood exactly when its image can encode that visible prefix, allowing the final unit to cross the cut. Let B mean this event for the first fixed observation stage. This condition uses ciphertext and the available code pool, not the hidden key or a guessed source reading.

All later prefix/closed targets have support within B. Therefore an initial q=π(·|B) is sufficient for the first actual target, even though q excludes prior keys: those excluded keys have zero G_1 and zero final likelihood. The first weight is Pπ(B)G_1, not merely G_1. Later old-key bridge ratios and invariant MH remain valid. This is a special support fact, not permission to restrict arbitrary proposal support or silently change targets.

Let M be the units that match at some position of the first visible prefix (including a crossing emission), and let s be their currently present availability mask. All other unit identities are irrelevant to B. Count completions with:

C_0(s)=1_B(s),

C_n(s)=(U−M)C_(n−1)(s)+Σ_(j=1..M) C_(n−1)(s∪{j}).

Known bound rows provide the initial mask; n is the remaining free-row count. Pπ(B)=C_n(s)/U^n. A selected next code receives C_(n−1)(s∪{code}) completions when relevant and C_(n−1)(s) otherwise. Those integer weights sum to C_n(s); their telescoping conditional path law is uniform over every supported dictionary completion. No stochastic repair or source-length input is required.

For a four-glyph prefix under the42-unit pool, at most13 units are relevant: up to4 single codes,3 fully visible pairs, and6 pairs crossing the final glyph. Dense availability DP therefore needs at most8192 masks and24 count layers for23 free rows. This is not dense enumeration of42^23 keys.

The own [finite checker](../../scripts/check_observation_initialization001.py) and [receipt](../../results/OBSERVATION-INITIALIZATION-THEORY-001/result.json) verify56 known-binding/prefix profiles,686 conditional completion paths and168 exact source-evidence identities against independent rational source-string enumeration. A crossing-emission example retains support where incorrectly closing the prefix would give zero. A source-zero control proves why strict positivity is essential.

For the **generic invented prefix**0123, not a fitted case or actual Voynich observation, the exact count is9732841398180148689892306910289985536 of42^23 dictionaries, probability≈0.4503041684. This demonstrates a tractable first-prefix count; it says nothing about long-record posterior mass, decoding speed or recovery. No actual conditioned initializer has been implemented or run yet.

## A route toward conditioning on whole records

Long-record support can involve all42 unit variables, so dense2^42 availability enumeration is not acceptable. Represent closed word-break support as a Boolean function of code presence. At each cipher offset, its function is the OR of `(required code is present AND suffix is encodable)` over matching one/two-glyph units. Each record shares the same unit variables; join records by AND. Compile this into a reduced ordered binary decision diagram (ROBDD), with explicit node/apply/memory caps. This compiler is only proposed here.

[Bryant (1986), author-hosted annotated paper](https://www.cs.cmu.edu/~bryant/pubdir/ieeetc86.pdf), introduction and Apply algorithm, supports canonical ordered Boolean graphs and memoized graph operations. Apply cost depends on graph sizes; representation size can still be exponential and variable ordering matters. We cannot promise a small graph for these cipher constraints.

[Darwiche and Marquis (2002), A Knowledge Compilation Map](https://www.cs.cmu.edu/afs/cs.cmu.edu/project/jair/pub/volume17/darwiche02a.pdf), §4/Table5 and §5, distinguishes tractable queries and transformations after compilation. Tractable model counting in representation size does not make the compilation itself tractable. Neither paper establishes a Voynich application or the following dictionary occupancy law.

**Critical distinction:** code-presence variables are not independent Bernoulli draws. They are occupied categories after a fixed number of uniform dictionary-row assignments. Even with one free row, two distinct code-presence events cannot both occur; multiplying their marginal probabilities would incorrectly give a positive answer.

Own occupancy identity: after forcing known-present relevant units, let m still-unfixed presence variables remain and b_k count satisfying availability subsets of size k. For n remaining free rows and a particular such k-subset, all its k named codes must occur, all other m−k relevant codes must be absent, and U−m other/known-present codes are allowed. Its number of assignments is:

T_n(k;U−m)=Σ_(j=0..k) (−1)^j binom(k,j) (U−m+k−j)^n.

Consequently the supported dictionary count is Σ_k b_k T_n(k;U−m). An ROBDD could supply the cardinality polynomial Σ_k b_k z^k using low-branch plus z×high-branch recursion, with `(1+z)` factors for every skipped unfixed variable. Forced-present variables are removed, not counted as extra uncertain bits. Counts retain arbitrary-size exact integers; no independent-bit approximation.

The separate [occupancy checker](../../scripts/check_occupancy_cardinality001.py) and [receipt](../../results/OCCUPANCY-CARDINALITY-THEORY-001/result.json) verify224 profile/layer identities against the original dense completion recurrence, reject the independent-presence null, and reproduce the generic23-row dictionary count. A bounded compiler, its cardinality traversal, conditional sampler and full-target SMC still require their own implementation, exhaustive controls and prospective empirical registration.

For conditioning on **complete-record support** B_full at initialization, intermediate targets must explicitly become π(K)1_B_full(K)G_t(K); MH must reject keys outside B_full at every stage. Initial weights include Pπ(B_full). The final target is unchanged because B_full is exactly the final likelihood support. Using the original unconstrained intermediate MH after restricting initial support would not satisfy this proposed law. With positive source tables, this construction would eliminate finite population extinction from dictionary encoding zeros, while graph/work caps and extremely concentrated source likelihoods could still prevent recovery. It addresses structural support, not source-language labeling or posterior mixing.

The prior Voynich/method limitations in [the coordinated-search review](coordinated-dictionary-2026-10-01.md) still apply. No evidence here identifies Latin, any historical channel, a correct semantic reading or an interpretable competent neural solver. The next concrete test would qualify bounded compilation and exact conditional sampling, then compare source recovery at fixed work on new development keys before historical application.
