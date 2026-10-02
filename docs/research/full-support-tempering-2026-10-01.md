# Search the whole record without discarding the dictionary prior

This is an implemented, locally qualified search direction, not a successful recovery result. It was developed after the fixed003 engineering run, without changing any240 registered inputs or repeating that run. [Inventory mathematics and prior Voynich/decipherment review](compiled-inventory-sampling-2026-10-01.md), [coordinated kernel review](coordinated-dictionary-2026-10-01.md), and [003 outcome](../experiments/COMPILED-INVENTORY-003-results.md) define the restricted synthetic channel and its current limits. Exposed synthetic engineering is distinct from fresh recovery and historical interpretation.

## Source-based choice

[Del Moral, Doucet and Jasra (2006), Sequential Monte Carlo Samplers](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf), §3.1 remark1, §3.3.2.3 equations30–31, §3.5 and §4.2.2–4.2.4, derives old-state incremental weights for invariant MCMC moves and discusses resampling before mutation. Gradual likelihood powers provide intermediate targets. Their mixture example improves particle posterior exploration without guaranteeing better normalizer estimates; few stages can leave large target discrepancies. It uses Gaussian mixtures, not these dictionaries, so its accuracy/runtime cannot transfer. Fixed schedules avoid silently applying fixed-schedule evidence identities to an adaptively chosen temperature path.

Hauer–Kondrak's Voynich substitution/anagram/abjad assumptions and Chiang et al.'s finite-state English-bigram substitution experiment, reviewed with primary sources in the linked earlier methods, differ from our unspaced duplicate-allowing one/two-unit channel. We do not import their recovery claims. No novelty claim is made for tempering or resample-move SMC. The following specialization, controls and exact finite identities are our own mathematics.

## Fixed full-support path

Let π(K) be the original independent uniform prior over free source-letter code rows, with any known rows fixed. Let L(K) be the unchanged contextual source probability of both complete observations, including original EOS. Under strictly positive source entries, nonempty deterministic emissions and valid source transitions, let B be the set of dictionaries permitting a full closed word break. Then L>0 exactly on B.

For a fixed schedule0=β0<…<βT=1 define

```
γβ(K) = π(K) 1_B(K) L(K)^β,
Zβ = sum_K γβ(K),
q0(K) = π(K | B),   Z0 = π(B).
```

At β0, γ0=π1_B explicitly. Never evaluate `0**0` as1 outside B. The already qualified cardinality/occupancy/onto-row sampler supplies q0 and the exact Fraction π(B). Conditioning on encoding support removes no positive final-likelihood dictionary. It does not preferentially supply the correct labels.

For each old particle key, first calculate `log_increment=(βnew−βold)*log L(old)`. Add `logmeanexp(log_increment)` to the normalizer estimate initialized at `log π(B)`. Resample, then mutate at βnew. The final target is the original πL, because L is zero outside B; it is not a newly constrained language model. Reversing weighting/mutation or omitting π(B) changes the transport/evidence and is rejected by the finite witnesses.

Each mutation draws a fixed, state-independent operation ticket. The original row/pair/involution moves remain symmetric; candidates outside B are rejected without source scoring. Supported candidates use acceptance `min(1, exp(β*(log L(new)−log L(old))))`. Source zeros, scorer disagreement, work/graph exceptions, nonfinite supported scores and integer rejection exhaustion abort; they are never treated as ordinary rejection or repaired.

The new `supported_refresh` option replaces only the1/32 whole-prior refresh ticket with an independent q0 draw. For the implemented uniform prior, q0 is uniform on B and that component is symmetric on B. Hence it has the same likelihood-only acceptance ratio. It reaches every supported dictionary with positive ideal probability, unlike a swap-only kernel. This is irreducibility, not a quantitative mixing guarantee: extreme likelihood ratios can make fresh proposals almost never accepted. Other operation tickets retain their original probabilities and identity moves, including one-free-row cases. `row` and original `coordinated` are available controls.

A future nonuniform prior needs its π ratio in symmetric MH moves and the appropriate proposal correction for a prior-conditioned refresh. The finite abstract weighted-prior cases test the general target identities using explicitly enumerated initial mass and corrected MH ratios; their refresh proposal remains uniform on B. They do not test a weighted-prior conditional refresh. The actual sampler/core supports **only the uniform free-row prior**. No arbitrary weighted sampler is implemented.

## Own qualification and falsification

[Exact receipt](../../results/FULL-SUPPORT-TEMPERING-THEORY-001/result.json), [tests](../../tests/test_tempered_inventory.py), [implementation](../../src/voynich/tempered_inventory.py), [finite checker](../../scripts/check_full_support_tempering001.py). All36 two-row/six-unit keys are enumerated. Two identical closed binary records make L=G², allowing the genuine fractional β=1/2 stage to be checked using exact rational arithmetic. Thirteen dictionaries have full support. Two positive sources (contextual and equal-label null), two priors (uniform and abstract weighted), and two coordinated proposal matrices (original and supported-refresh) produce eight cases.

The checker verifies31,104 detailed-balance pairs;864 exact unnormalized state-transport checks across β0,1/2,1;5,408 literal ordered two-particle first-increment ancestor paths with multinomial resampling and analytically integrated invariant mutation. It propagates the expected unnormalized N1 law, not an exact finite particle posterior. The N2 check concerns the first increment, not a fully enumerated multistage two-particle run. Initial-normalizer omission, mutation-before-weighting, unrestricted β0 support leakage and dropped nonuniform-prior ratios each fail exact witnesses. The equal-label null stays ambiguous at β1.

Nineteen local tests pass, including actual native C++ and independent Python full search trajectories for all three kernels; keys, ancestry, acceptance counters and bank hashes agree, returned log scores/evidence within2e−14. Tests check fixed rows, independent final source scores, malformed schedules, observation/source-zero refusal, unsupported candidates never entering source calls, explicit work/integer/nonfinite failures and actual conditional refresh draws. These are tiny fixtures, not the order12 corpus or historical manuscript.

Finite correctness does not make finite particle ratios or log evidence unbiased, certify posterior concentration, or imply accurate message recovery. The ideal integer sampler remains subject to the explicit64-rejection censoring event: at most R+2 integer calls per supporting dictionary draw, recorded union bound `conditional_draws*(R+2)*2^-64`. There is no biased fallback. Floating resampling/MH arithmetic is approximate even when rational finite transition laws are exact. Always-resample fixed schedules can cause genealogical collapse; trace ESS, maximum weight, ancestry, distinct keys, accepted changes and supported refreshes rather than mistaking mere survival for competence.

## Next empirical decision

The actual full-support tempering runner is implemented; **no original-source search, fresh recovery panel, training or historical decoding has run under it**. Before a fresh campaign, separately register original-source closed-table/initial-bank/mutation cost and cap behavior, fixed schedules and all stop conditions. Then freeze fresh synthetic keys and positive/null controls, seal predictions before Gold evaluation, compare supported initialization without moves, local/coordinated moves and supported refresh at bounded resources, and retain failed recovery/calibration gates. Frequency order is an exposed engineering choice supported by003, not a guarantee of unseen compilation. Improve source/channel assumptions if supported search still favors systematically wrong messages; correct probability algebra alone cannot resolve unidentifiability.
