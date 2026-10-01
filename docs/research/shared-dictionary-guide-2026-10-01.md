# Whole-dictionary future compatibility and conditional integration

2026-10-01. Own method, under calibration. The [previous lookahead experiment](../experiments/SOURCE-BELLMAN-001-results.md) failed; [exact second/third moments](after-bellman-2026-10-01.md) show that occurrence-wise renewal misses shared-dictionary dependencies. This motivates evaluating a specified replacement, without assuming it will decode anything.

## Primary sources reviewed before empirical evaluation

Ravi and Knight (2011), introduction and selected section 3.1 through type sampling and boundary inference: their Bayesian decipherment updates every occurrence of a cipher type together, rather than treating token assignments independently. Their English letter/word model, sparse probabilistic channel, and homophonic deciphering direction differ from our duplicate-allowing deterministic source-row-to-one/two-glyph mapping. We do not import their CRP prior, language model, word dictionary, annealing or reported recovery guarantees. The shared-type principle is relevant; their solved-cipher results do not qualify this guide. [Paper](https://aclanthology.org/P11-1025.pdf).

Doucet, de Freitas, Murphy and Russell (2000), introduction and selected sections 2–3 and 4.1: integrate tractable conditional structure rather than sampling every latent variable. Their variance comparisons condition on matching proposals; they explicitly discuss the computational expense and inefficiency of prior proposals. Our fixed-key IID dynamic program integrates source strings/segmentations, while an additional exact average over one free dictionary row is a conditional-expectation estimator. We implement no particle-filter posterior, resampling or MCMC kernel, and assert no matched-CPU variance guarantee. [Paper](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/optreadings/DoucetdeFreitasMurphyRussell_RaoBlackwellizedParticleFilters.pdf).

Hauer and Kondrak (2016), refreshed selected section 5.4 and conclusion: their Voynich anagram/abjad decoding outputs were not syntactically and semantically coherent, and they discuss the alternative of language-model/anagram artifacts. Their language and pure-substitution assumptions differ from this artificial shared-variable-unit channel. A better conditional likelihood here would identify neither the manuscript's language nor its historical encoding. [Paper](https://aclanthology.org/Q16-1006.pdf).

Reading depths are selected sections, not full theorem audits or reproductions. One mistaken anthology URL returned unrelated referring-expression material; that content was not used. The relevant papers above were inspected directly.

## Specified surrogate

Freeze smoothed root-context row probabilities `q_r=(p_root,r+epsilon)/(1+M*epsilon)`, with epsilon=1e-8 to match the old guide. Draw free dictionary rows independently and uniformly from the U=g+g² one/two-glyph units; override assigned rows with their conditional known values. Each complete sampled dictionary K is reused everywhere in both remaining records.

For a fixed K, let `w_u(K)=sum_r q_r*1(K_r=u)`. For record t with remaining observed glyphs y, compute a backward table

`D_t[end]=rho`

`D_t[j]=(1-rho)*(w_single(y_j)*D_t[j+1] + w_pair(y_j,y_(j+1))*D_t[j+2])`.

The pair term is absent at the last glyph. For an already completed record, its factor is 1: EOS was already paid in the prefix. For unfinished records use the entry at its current offset. The conditional fixed-dictionary likelihood `L(K)` is the product over unfinished records. Computation uses log sums, with no floor or invented nonzero value for impossible sampled dictionaries.

The target of this guide is `H_iid_shared(s)=E_K[L(K)|partial dictionary]`. This preserves dictionary sharing but still replaces the context-dependent source by root IID probabilities. It is not the original future H, an admissible bound, or a translation. No binding prior is paid twice; already assigned rows are conditional, not newly random.

## Estimators and variance

The arithmetic mean of L over N independent conditional-prior dictionaries is unbiased for the stated surrogate. A finite bank can miss all compatible dictionaries; estimated zero does not establish impossible actual future text. Log estimates are biased and may be absent. Common random numbers use the same full draw bank for all queries, with known rows overridden; marginally the still-free rows remain independent uniform. This deterministic coupling aims to make comparisons interpretable, without asserting it reduces every contrast's variance.

The conditional-integration variant selects the highest root-probability free row, with lowest-index tie breaking. For each base sample of other rows, average L over all U assignments of that row. This is a conditional expectation of the raw likelihood random variable, so its variance is no greater at the same number of independent base samples. It costs U times as many fixed-dictionary tables; a same-base variance inequality is not a matched-work guarantee. No adaptive row selection uses outcomes or known answers.

The calibration compares MC16, MC672, and RB16 with U42. MC672 and RB16 each evaluate672 fixed dictionaries per query/seed; MC16 costs16. Base contribution ESS and maximum contribution share quantify concentration of likelihood contributions, not posterior accuracy or full-key diversity. RB16 ESS groups integrate42 completions each; MC672 ESS uses672 individual samples and is not directly the same resolution.

## Before increasing the solver budget

The exact tiny reference sums full dictionaries and source strings with rational arithmetic, separate from fixed-key DP. Tests cover all36 one/two-glyph observation pairs and49 partial bindings, suffix offsets, completed records, impossible observations, variance at fixed base count, and the prior covariance witness. A separate auditor reconstructs every sampled completion and evaluates likelihoods with outgoing forward edges. This is same-author code with an alternate computational direction, not independent expert review.

The real-size calibration uses only twelve predetermined conditional queries from four exposed artificial cases: root, archived correct-prefix state, and one explicit altered binding. The alteration is a hypothetical conditional state, not a proven reachable competing beam state or a causal intervention in a neural model. No search frontier or reader is rerun. Measured stability, support and cost decide whether prior sampling merits integration; prefix-versus-altered preference is exploratory and cannot establish recovery.

A finite successful stability result would still require actual recovery, matched total CPU, new keys, null controls and then a historical channel/transcription justification. A failure would motivate observation-informed dictionary proposals or exact short-block constraints, rather than increasing prior-draw count without a variance argument. No open-ended training, cloud expense or model-size expansion is authorized by the calculation itself.
