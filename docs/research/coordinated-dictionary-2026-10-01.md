# Coordinated code changes and corrected initialization

Status: own finite mathematics qualified; fresh-key empirical recovery untested at registration. This addresses a search defect under a restrictive synthetic channel, not evidence about the Voynich plaintext language or a neural circuit.

## Evidence that motivates a different search

The closed CONTEXTUAL-GIBBS-001 panel found no complete used generator key. Its 160 surviving dictionaries all lack enough instances of necessary generator codes to recover the used key by permuting source-letter labels. That inventory ceiling is an oracle diagnostic calculated after fitting, not a rule available to a blind solver. All nine particle extinctions happened at early observation cuts before EOS. Positive source probabilities mean such zeros are failures of the available codes to encode the observed prefix. They do not identify which inventory modification would help.

The separate exact two-letter witness has four supported dictionaries under two closed records, `01` and `10`: `(0,1)`, `(1,0)`, `(01,10)`, `(10,01)`. Single-row replacements leave four components; adding swaps leaves two. This is a counterexample to general connectivity, not a proof that the 23-row failures share exactly this topology. Spare rows and shorter observation prefixes can alter it.

## What the methods literature actually supports

[Neklyudov et al., Involutive MCMC (2020)](https://proceedings.mlr.press/v119/neklyudov20a/neklyudov20a.pdf), §2/Algorithm 1 and §3.1, gives a framework for reversible auxiliary-variable moves. Invariance alone does not establish ergodicity or practical mixing. Our finite code-pair transformations are elementary discrete involutions; there is no continuous Jacobian or neural flow approximation. We retain state-independent operation and ordered-row selection to avoid uncomputed reverse selection probabilities.

[Del Moral, Doucet and Jasra, Sequential Monte Carlo Samplers (2006)](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf), §3.1/initialization and §3.3.2.3, supports properly weighted initialization and invariant moves within a sequence of targets. It does not make a small finite population exact. Our first weight includes the actual prior/proposal density ratio; later bridge ratios are evaluated before mutation.

[Chiang et al., Bayesian Inference for Finite-State Transducers (2010)](https://aclanthology.org/N10-1068.pdf), §3.2 and §5, discusses blocked derivation proposals and their reversibility/correction difficulties. Its decipherment example uses a fixed English bigram source and a 414-letter substitution cipher. Its reported accuracy cannot transfer to our shared, variable-length, potentially homophonic dictionaries, much less Voynich. We do not adopt its skipped MH correction.

For prior Voynich application, [Hauer and Kondrak (2016)](https://aclanthology.org/Q16-1006.pdf), §5/conclusion, explores substitution, within-word anagrams and abjad hypotheses. Its language suggestions remain method-dependent proposals. Our unspaced two-length channel is different; recovering its generated keys would qualify only that restricted synthetic search, not authenticate a manuscript translation.

## Actual target and initial proposal

A dictionary has R source-letter rows. Each free row independently selects one of U = g + g² one- or two-glyph units under the original uniform prior π. Known rows, if supplied, stay fixed. Each observation bridge G_t(K) is the exact original contextual source probability of the observed prefixes given K, allowing the last emission to cross an intermediate cut; only the final stage closes with EOS. The target is proportional to π(K) G_t(K).

Let A mean that the dictionary contains every one-glyph code. With a strictly positive source, A suffices to encode every observation prefix and complete record, but many true prior keys lie outside A. We therefore do **not** condition the target on A. Instead:

q(K) = π(K) / 2 + π(K | A) / 2,

q(K) / π(K) = 1/2 + 1_A(K) / (2 Pπ(A)).

Every prior key keeps positive proposal probability. The first unnormalized particle weight is G_1(K) π(K) / q(K). Later weights are G_t(K)/G_(t−1)(K) at the old key. Finite samples, extinction and numerical/resource failures remain possible. Dropping the first correction silently changes the target/evidence; the exact small-case checker exhibits this discrepancy.

Conditional sampling is exact over integer counts. If n free rows remain and m particular required one-glyph codes are still absent, the number of completions is:

H(n,m) = (U−m) H(n−1,m) + m H(n−1,m−1), with H(0,0)=1 and H(0,m>0)=0.

The selected next code receives H(n−1,m−1) completions when it covers a missing requirement and H(n−1,m) otherwise. Summing these code weights gives H(n,m), and the telescoping path probability is uniform over covered completions. Counts can exceed uint64, so the sampler uses arbitrary-size integers and unbiased byte-based rejection. Its 64-attempt bound raises an explicit failure; there is no modulo bias or fallback. A zero-probability coverage event is rejected as invalid configuration.

The exact-law proof concerns the ideal uncapped rejection draw. Each actual bounded draw couples to that ideal draw unless all64 attempts reject; its failure probability is at most2^−64. At most736 conditional draws occur per initialization, giving a union bound736×2^−64<4×10^−17 per call. Conditioning on a completed capped run is not mathematically identical to the ideal law; this negligible resource-censoring event is explicit, never hidden by a replacement draw.

## Coordinated reversible moves

An ordered pair of distinct free rows is selected uniformly. Each following transformation is self-inverse on the complete finite unit pool; configurations outside a stated pattern remain unchanged:

- Swap the two entire units.
- Swap their first glyphs, keeping unit lengths.
- Swap their last glyphs, keeping unit lengths.
- For one unit of length one and another of length two, concatenate in row order and switch the cut from 1+2 to 2+1 or vice versa.
- Replace single units `(a),(b)` with `(a,b),(b,a)`, and reverse that replacement for two reversed two-glyph units.

The last move crosses the verified inventory barrier directly. Combined with swapping, it connects the four supported witness states without a random full-key refresh. This is a local finite proof, not an efficiency result on 23 rows.

The registered mixture has 32 equally likely operation tickets: 16 single-row replacements, 4 uniform two-row replacements, 3 swaps, 2 first-glyph swaps, 2 last-glyph swaps, 2 boundary shifts, 2 reversed motifs and 1 whole-free-key uniform refresh. Replacements include the old code. With fewer than two free rows, pair involutions are identities rather than a renormalized mixture. Each component is symmetric, hence the mixture is symmetric. The acceptance probability is min(1,G_t(K′)/G_t(K)); zero-target proposals are rejected even when the random uniform is zero. Full refresh supplies formal finite-support connectivity, with no claim that its acceptance is useful at scale.

## Qualification and empirical separation

The exact checker and tests enumerate code pairs for g=1,2,6; exhaust completion counts and conditional path probabilities; construct full small proposal matrices; verify detailed balance at every observation stage; and propagate the weighted N=1 law to recover the exact unnormalized target and evidence. They include an unidentifiable equal-source-label null. Full-panel transport tests distinguish graph caps from cumulative work caps and reject gold-containing solver inputs.

The next experiment generates four fresh development keys only after code/design publication. These are fresh keys within an adaptively studied generator/source family, not untouched confirmation data. Three arms isolate initialization from coordinated moves. Each gets the same particle count, observation schedule and maximum number of key tables. Completed calls have matched table budgets; CPU and actual work of failed calls need not match. Cipher-only predictions are sealed across all 24 calls before ground-truth recovery metrics are computed. Fixed-key Viterbi readings are conditional on the selected sampled key, not exact global plaintext MAP.

Even an improvement would require disjoint mechanism/language tests, ambiguity accounting, model adequacy checks and independent manuscript predictions before any historical claim. Mechanistic interpretation of a learned solver remains deferred until one demonstrates competent recovery; increasing model size alone has not met that standard here.
