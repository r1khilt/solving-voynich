# Compile the code inventory, then assign its source labels

Own mathematical and engineering successor to [observation-conditioned counting](observation-conditioned-inventory-2026-10-01.md). Prior recovery panels failed. This method addresses their structural-support problem; it does not establish a historical encoding or competent neural solver.

## Method review and reason for the architecture

The [prior decipherment/Voynich review](coordinated-dictionary-2026-10-01.md) remains applicable: Hauer–Kondrak's substitution/anagram/abjad assumptions and Chiang et al.'s English-bigram substitution example differ from our synthetic unspaced, duplicate-allowing, variable-unit channel. Their accuracy cannot be transferred. This is a conditional engineering experiment on exposed synthetic ciphertext, not a new claim about manuscript glyphs, language or words.

[Bryant (1986), author-hosted annotated paper](https://www.cs.cmu.edu/~bryant/pubdir/ieeetc86.pdf), introduction and Apply sections, motivates a canonical reduced ordered binary decision diagram (ROBDD) with shared subfunctions. Variable ordering and intermediate graphs can dominate cost. The annotation's footnote7 corrects the older conjecture about output-sensitive Apply complexity: a small result does not imply cheap construction. We therefore bound Apply calls, retained cache entries, nodes and memory separately; we do not rely on that conjecture.

[Darwiche and Marquis (2002)](https://www.cs.cmu.edu/afs/cs.cmu.edu/project/jair/pub/volume17/darwiche02a.pdf), §4/Table5 and §5, separates tractable queries after compilation from compilation cost. Our cardinality/occupancy extension below is our own derivation. Neither cited paper demonstrates Voynich recovery. All mathematical identities here are finite exact integer or rational checks, not externally peer-reviewed proofs.

## Boolean constraint and the correct dictionary prior

The original dictionary prior assigns each of R free source-letter rows independently and uniformly to one of U one/two-glyph codes. For six glyphs, U=42; duplicate codes are allowed. Positive source transition probabilities, valid deterministic state transitions, nonempty emissions and EOS probability strictly between zero and one imply: a dictionary's closed-record likelihood is positive exactly when its set of codes permits a complete word break of every record. Source labels affect likelihood values but not this structural support. Zero-probability source entries would invalidate that equivalence.

Compile the word-break recurrence `F_i = OR_c (present_c AND F_(i+len(c)))` for matching codes, with F_end=true and crossing emissions forbidden for closed records. AND the record roots. One shared Boolean variable per code represents its presence in the dictionary, rather than a separate presence variable for each occurrence. Reduced nodes merge equal low/high branches and identical ordered triples; memoized Boolean Apply shares intermediate work. Nonclosed finite controls allow the final unit to cross the visible cut.

The prior on these presence bits is **dependent occupancy**, not independent Bernoulli. One source row cannot occupy two distinct code categories simultaneously. Let J be the distinct codes already fixed by known source rows; they are forced present. Let b_k be the number of satisfying subsets S of the other U−|J| codes having cardinality k. The generating polynomial accounts for every unknown variable, including irrelevant or skipped variables, through factors `(1+z)`. A forced-known variable follows its high branch without contributing z.

For a particular subset S, the free rows may use J∪S; every code in S must occur at least once. The number of assignments is

`T_R(k; |J|) = sum_(j=0..k) (-1)^j binom(k,j) (|J|+k−j)^R`.

Consequently `H = sum_k b_k T_R(k;|J|)` is the exact supported completion count and `P_prior(B)=H/U^R`. This convention includes all code variables, whereas the previous dense-prefix proof groups irrelevant codes into its base category; their counts agree.

## Two-stage exact draw

Avoid recompiling or recounting after every source-row binding:

1. Draw cardinality k with integer weight `b_k T_R(k;|J|)`.
2. Uniformly unrank one of the b_k satisfying subsets S, by exact counts of low/high completions in the diagram.
3. Assign the free source rows uniformly over completions using J∪S, with each S code occurring at least once. For the next code, weight it by the number of completions of the remaining rows that cover the still-missing codes.

Every supported labelled dictionary has one S and k. Its probability telescopes to `(b_k T_R/H) × (1/b_k) × (1/T_R) = 1/H`. No likelihood, guessed source string, generator key or model output selects S. Label ambiguity remains: a supported inventory can correspond to many source-letter permutations, and its prior need not concentrate near the generator dictionary.

The byte-based integer draw has a fixed64-rejection cap and raises without fallback. The ideal draw is uniform. Under ideal byte randomness, its capped implementation couples to that draw except with probability at most2^−64 per call. At most25 draws per full23-row dictionary give a per-bank bound800×2^−64, and eight32-key banks give6400×2^−64<3.5×10^−16. Conditioning on completion can introduce this explicit negligible censoring; no aborted draw is replaced or retried.

## Correct later target, if used in a solver

Whole-record conditioning is stronger than conditioning on the first observed prefix. With `q=prior(·|B_full)`, every intermediate target must be `prior(K) 1_B_full(K) G_t(K)`. First weights include `P_prior(B_full) G_1(K)`; later old-key ratios and MH mutations must retain the indicator. The final target remains the original one because its likelihood is zero outside B_full. Using unrestricted intermediate MH after this initialization is incorrect. The exact finite checker supplies both a successful unnormalized-mass transport proof and an explicit incorrect-normalizer witness for that shortcut.

This could remove encoding-zero population extinction under the stated positive-source assumptions. It cannot prevent concentrated source likelihoods, poor mode exploration, numerical errors or graph/time caps. A finite particle population is still approximate. This checkpoint implements the compiler/count/sampler only; it does not yet implement or register the new restricted SMC solver.

## Verification and prospective engineering scope

[Finite tests](../../tests/test_inventory_bdd.py) exhaust small Boolean inventories, known bindings, all satisfying subset ranks and labelled dictionary completions, and check exact SMC mass transport. Six-glyph generic-prefix tests compare against the previous independent dense availability recurrence under natural/reverse variable orders. Equal-label and impossible-known-key controls separate support from identification. [Panel transport tests](../../tests/test_compiled_inventory001.py) cover the full eight-cell tiny native/Python replay, separate compiler/native caps, no-gold admission, count corruption and duplicate-run refusal. Early transport preparation caught a list-versus-tuple bug in the independent literal checker and a helper import error; both were fixed before registration and corpus computation.

Preparation also discovered that the inherited admission helper opened earlier answer-containing fixtures and fitted banks while checking provenance. Its preliminary read-only check did not perform inference, but it was unsuitable for the promised archive boundary. The new admission uses closed source/native/contextual qualification metadata directly; a test ensures its only bound data archive is the ciphertext query. The parent result's gold path is metadata and its contents are not opened in this panel.

The [engineering registration](../experiments/COMPILED-INVENTORY-001.md) uses only four exposed ciphertext pairs and the original source, with two fixed variable orders. Actual compilation, exact counts, supported draws and first-draw native/Python score agreement remain unmeasured at registration. The audit independently traverses stored graphs and computes cardinality coefficients bottom-up plus onto counts by inclusion-exclusion, as well as replaying every compiler/RNG result. Shared compiler/native replay is explicitly same-author, not independent expert review. Historical analysis and neural mechanistic interpretation remain unqualified until a competent solver exists.
