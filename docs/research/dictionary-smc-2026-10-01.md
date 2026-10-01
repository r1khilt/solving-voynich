# Progressive observation conditioning with whole-key revisions

2026-10-01. Own implementation following the failed [prior-key calibration](../experiments/SHARED-KEY-GUIDE-001-results.md). Whole-key guessing almost never supplies the prescribed future path, and positive estimates are usually dominated by one draw. More uninformed samples are an expensive response to a proposal problem.

## Reviewed sources and limits

Del Moral, Doucet and Jasra2006, introduction/selected§§2.1–2.3,3.1 and3.3.2.3, revisited the algorithm, remark1 and equations30–31 directly. Their common-space targets and target-invariant MCMC construction justify weight-before-move resample–move. The reverse-kernel simplification excludes expanding-support sequences; our exact observation events shrink support. This is a selected-section review, not a full theorem audit or reproduction. [Primary paper](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf).

Ravi and Knight2011, selected§3.1 type sampling revisited: coordinated changes must affect all occurrences of a cipher type. Our deterministic source-row-to-variable-unit direction differs from their English homophonic letter/word model, CRP/sparse probabilistic channel and inference over derivations. We integrate source paths given a fixed key rather than applying their cache updates. Their solved-cipher results do not establish performance here. [Primary paper](https://aclanthology.org/P11-1025.pdf).

Hauer and Kondrak2016,§5.4/conclusion and the incoherent-output/anagram-LM-artifact discussion revisited. Their anagram/abjad language identification assumptions do not qualify our channel or establish a historical language. [Primary paper](https://aclanthology.org/Q16-1006.pdf). Prior [method comparisons](shared-dictionary-guide-2026-10-01.md) and [joint-constraint diagnosis](source-key-coupling-2026-10-01.md) remain relevant. Source review preceded the new empirical calls.

## Implemented law

For a supplied partial dictionary, unknown rows have an independent uniform prior over U=g+g² units. Prior particles are complete keys with known rows overridden. No hidden plaintext, segmentation or source length is supplied. A whole dictionary is shared by all future occurrences in both records.

For each unfinished record, G_c(K) is the source probability of reaching c glyphs with the observed prefix. A crossing two-glyph emission compares only its visible part. G_0=1; an already closed record has factor1, with EOS paid earlier. Both-record factors multiply given the same K. Balanced relative progress adds stride4 observed glyphs in one selected record at a time. Even after all glyphs have been added, a separate final stage requires exact boundaries and EOS. Every positive final key belongs to all preceding prefix supports.

The source-aware evaluator preserves the supplied Markov probability and transition tables, starting contexts and conditional offsets. It sums outgoing source edges with log probabilities and explicit state/edge/table/allocation caps. Only an actual one-state source uses the faster vectorized IID suffix calculation. It never automatically discards source context, introduces a likelihood floor, or truncates source paths with a beam.

At each stage compute G_new/G_old at the OLD dictionary. Multiply its arithmetic population mean into the evidence estimate. Multinomial-resample parents, then perform the registered number of target-invariant symmetric MH proposals. Reject every zero-target proposal. No refill on extinction. Finite posterior estimates and log evidence are biased in general; no interval/unbiased-log/complete-support certificate is claimed. For a conditional prefix, this estimates the conditional remaining likelihood, not the likelihood of the entire original prefix-and-suffix event.

The row kernel chooses one free row uniformly and its code uniformly, including self. The joint mixture chooses one row with probability3/4, two distinct free rows with probability3/16, or all free rows with probability1/16; each selected row receives an independent uniform code. Subset selection is independent of the current key, so proposals are symmetric and the uniform conditional prior cancels in the MH ratio. The all-free component gives a positive proposal route between every pair of supported finite keys. It does not provide a useful mixing-time bound; high-dimensional prior refreshes may almost never succeed. Assigned rows never change.

## Correctness evidence before calibration

[Core](../../src/voynich/dictionary_smc.py), [tests](../../tests/test_dictionary_smc.py) and the [exact arithmetic checker](../../scripts/check_dictionary_smc_theory001.py) use independent source-string enumeration, contextual/zero-transition sources, partial offsets, paid EOS and mid-unit cuts. The checker passed1960target/kernel matrices,70560detailed-balance pairs,5760weighted transport coordinates and392exact one-particle/two-stage evidence expectations across all49conditional two-row keys, two observations, five stages and two proposal kernels. [Receipt](../../results/DICTIONARY-SMC-THEORY-001/result.json).

Wrong operation order is demonstrably biased: under the contextual fixture the first observation probability is3/8. Moving with the first-stage invariant kernel before weighting instead gives77/192(row) or619/1536(joint). These are finite mathematical counterexamples, not measured real-corpus errors. Initial floating matrix test tolerance2e−16 missed2.22e−16rounding; changed to1e−15and backed with exact Fraction equalities. No empirical threshold changed.

## Bounded empirical question

[DICTIONARY-SMC-001](../experiments/DICTIONARY-SMC-001.md) calibrates only the explicitly supplied one-state IID surrogate, matching the previous shared-key target. The general Markov evaluator has finite exact correctness checks but no realistic-scale resource qualification yet. The experiment is not a contextual recovery run, a fresh-key test, a null qualification or mechanistic interpretation of neurons. Successful stability would justify measuring contextual cost and testing actual recovery next; failure would require inspecting remaining proposal/target/mixing deficits rather than calling the historical source model wrong.
