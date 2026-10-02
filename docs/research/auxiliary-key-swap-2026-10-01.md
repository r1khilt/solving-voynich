# A cheaper terminal move in the joint key and reading space

Own proposed connection and finite qualification, while TEMPERED-RECOVERY-001 runs unchanged. The long-term objective is decipherment; this does not infer a historical language or channel. The new unpruned suffix-source posterior-path module supports arbitrary dense source order but has only artificial finite qualification here. No original-model cost, speed gain, mixing or recovery result is claimed.

## Method review and prior boundary

[Neklyudov et al. 2020, §2 Algorithm1/Proposition2 and §3.1 Trick1](https://proceedings.mlr.press/v119/neklyudov20a/neklyudov20a.pdf) motivates auxiliary-variable involutions: preserve the augmented density, retain the correct acceptance ratio, and distinguish invariance from ergodicity. Extra latent dimensions can worsen convergence. [Chiang et al. 2010, §3.2 and §5](https://aclanthology.org/N10-1068.pdf) discusses derivation-lattice proposals, dependence lost by an approximation, and an MH correction. We do not adopt its skipped correction or transfer its substitution-cipher accuracy. Read depth is those selected sections, not independent replication.

Existing [coordinated dictionary work](coordinated-dictionary-2026-10-01.md) and [source revision review](source-key-revision-2026-10-01.md) already cover Voynich applications, shared-key barriers, blocked proposals and why a naive particle-Gibbs reference is invalid. Existing row swaps score the marginal likelihood over all compatible texts. This proposal instead keeps a compatible text as an auxiliary state. It does not make unused entries identifiable, repair a missing inventory or qualify a neural circuit. The [posterior-reading conversion](posterior-key-reading-2026-10-01.md) concerns output decisions; the move here concerns invariant search. Neither is inserted into the live campaign.

## One full key, all records, and a sampled compatible reading

Under the declared iid uniform row prior π, let Q(X) include source resets and geometric EOS for every record, I_K(X,C) indicate literal re-encoding, and L(K)=Σ_X Q(X)I_K(X,C). The terminal joint target is

    Γ_1(K,X) = π(K) Q(X) I_K(X,C).

Its key marginal is the intended π(K)L(K); X|K,C has probabilities Q(X)/L(K) over compatible tuples. Draw paths separately conditional on one shared K, not by choosing separate convenient keys per record. A Gibbs path refresh preserves Γ_1.

Pick a transposition of source-letter indices a,b by a state-independent rule. Swap the corresponding dictionary rows and simultaneously swap a,b everywhere in all source records. Applying this transformation twice returns the original state. Every observed glyph string, record length and reset remains unchanged; duplicate units and identity key changes are retained. Under the uniform prior π(K')=π(K). The final-temperature MH ratio is simply

    α = min(1, Q(X')/Q(X)).

Detailed balance follows from min(Γ_1(K,X),Γ_1(K',X')); no language-marginal integration is needed for this swap. Source point scores still require all affected contexts, not just positions containing a,b: changing a letter changes subsequent suffix states. An unqualified local n-gram shortcut would be wrong for the original interpolated order12 source.

Multiple MH swaps may be applied while retaining the auxiliary state, with occasional valid Gibbs path refreshes. Each operation preserves the target; their composition need not be reversible on the joint state, but remains invariant. A finite run starting from approximate particles is still approximate. Taking an arbitrary Viterbi path as if it were a conditional posterior draw does not qualify the Gibbs initialization or refresh. Formal irreducibility is not supplied by these swaps.

## Why the shortcut cannot enter intermediate temperatures unchanged

The frozen tempering target is π(K)L(K)^β, restricted to full support. Augment it with the conditional path law Q(X)I/L(K):

    Γ_β(K,X) = π(K) L(K)^(β−1) Q(X) I_K(X,C).

Its correct transposition ratio is

    [π(K')/π(K)] [L(K')/L(K)]^(β−1) [Q(X')/Q(X)].

The marginal factor cancels only at β1. At β0 or1/2, a point-only rule is generally wrong. Instead tempering Q(X)^β would give key marginal π(K)Σ_X Q(X)^βI, different from π(K)L(K)^β. At β0 it weights keys by their number of compatible paths rather than the uniform supported prior. Finite negative witnesses check both mistakes. The new acceptance helper refuses β other than1 and nonuniform-prior use. A nonexchangeable row prior would require its ratio; exchangeable nonuniform priors may still cancel under a swap, but that broader implementation is not claimed.

## A concrete tradeoff, not a promised win

For a strictly positive source, the label transposition is a bijection of compatible source-path sets. If a fresh conditional path is drawn, the expected raw ratio is L(K')/L(K). Concavity of min(1,r) gives

    E[α_aux | K] ≤ min(1, L(K')/L(K)) = α_collapsed.

Thus the existing fully marginalized swap has at least as much average acceptance for the same transposition. The new move can only be attractive if its cheaper point-score work compensates for this loss, or if persistent auxiliary updates provide a useful cost tradeoff. That requires an actual equal-time or measured-cost comparison; no theorem here promises faster recovery. The finite checker verifies the inequality and strict cases, rather than hiding this limitation.

Inventory is preserved: the multiset of one/two-glyph dictionary units cannot change through swaps. If required codes are missing, label search alone cannot recover the true used mapping. Keeping a separate qualified inventory-changing kernel or adding a correctly derived joint block transformation would be necessary. The old inventory ceilings and source inadequacy remain distinct causes of failure.

## Unpruned dense-source path sampling

The new SuffixPosteriorPaths borrows immutable float64 source rows and uint32 transitions. Nodes are (observed offset, source state). A matching nonempty unit emits one source letter, adds log(1−ρ)+log p(letter|state), and advances both offset and source state. At the record's exact end the backward value is logρ; earlier backward values are log-sums of all complete continuations. A sampled transition has conditional probability proportional to its edge weight times the backward probability of the following node. These ratios telescope to Q(x)/L(K,c). Duplicate units are distinct letter edges; source zeros, dead continuations and ambiguous segmentations remain explicit.

Nonempty emissions make this graph acyclic. No beam, top-k, greedy path, unseen-path renormalization or fallback is used. Empty observations have only the empty source path. Impossible observations return no conditional paths. Node/edge/owned-work caps raise errors before partial results can escape. Sampling accepts0–4096 draws and a separate seed, with a prospective conservative output byte check. The finite PRNG and floating probabilities are approximate; endpoint rounding is constrained below the final CDF value, never an arbitrary refill.

Defaults:100knodes/400kedges/128MiB conservative lattice envelope/64MiB sample-output bound. The envelope reserves offset-map space and1024bytes/node plus512bytes/edge for graph/backward/choice objects. It excludes borrowed source arrays, caller inputs and returned paths, whose separate bound is checked; whole-process RSS still needs a driver limit. Samples and point probabilities are reproducible. Entire source/cipher/key identity admission and original-scale timing remain future empirical responsibilities. The module does not create a training job or read Gold.

## Qualification plan and remaining work

Finite check: all36 two-row dictionaries from six one/two-glyph units,15 binary observations of length0–3, both positive IID and contextual sources. The1080 single-record lattices are checked against complete literal Fraction sums and full conditional path distributions; seven seeded paths per supported case are re-encoded and source-scored. All450 ordered two-record panels then check the real coupled transformation, involution/re-encoding, exact joint detailed balance and Gibbs-refreshed key stationarity, float acceptance, and collapsed-vs-auxiliary acceptance dominance. Identical-record panels make L=G² so β1/2 counterexamples can use exact rational arithmetic. A deliberately nonexchangeable toy prior witnesses the omitted prior factor; no weighted-prior kernel is implemented. Unequal path-count witnesses reject replacing marginal tempering by path tempering at β0.

One bounded saved receipt binds both new modules/tests, checker/memo and unchanged tempering target implementation.120wall/100absoluteCPU/512MiB/CPU1/$0/no original corpus/model/empirical archives/Gold. Focused tests include cap refusal, immutable/invalid source guards, duplicate emissions, empty/dead/impossible cases, zero source probabilities, deterministic streams, invalid-temperature/prior refusal and sample-output caps. No original-source path draw or coupled move is qualified by a finite PASS alone; separately register a bounded actual cost/transport experiment before empirical use. The current fresh recovery run and its one full audit must close without adaptive changes. Future comparisons must preserve support/inventory controls, full failed denominators and message metrics; a cheaper valid kernel is useful only if it improves practical inference.
