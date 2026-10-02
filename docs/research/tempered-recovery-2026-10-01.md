# Fresh message recovery after original-source transport qualification

Prospective exploratory method; no fresh result exists yet. [Qualified full-support law and implementation](full-support-tempering-2026-10-01.md), [actual original-source engineering result](../experiments/TEMPERED-INVENTORY-001-results.md), [support-only prior uncertainty](inventory-entropy-2026-10-01.md). Preserve earlier recovery/calibration failures; a higher source score is not an authenticated reading.

## Why this comparison

[Del Moral, Doucet and Jasra (2006)](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf), §3.5 and §4.2.3, distinguishes weight degeneracy from mutation mixing. More temperature steps reduced adjacent-target discrepancies in their mixture example; that motivates a test, not a promise for these keys. Retain the already qualified old-state weighting and full-support target.

[Chiang et al. (2010)](https://aclanthology.org/N10-1068.pdf), §5, tests fixed English-bigram substitution on414letters. [Hauer and Kondrak (2016)](https://aclanthology.org/Q16-1006.pdf), §4–5, investigates substitution/anagram/abjad decoding and Voynich language hypotheses. Those model families and data differ from our Latin-source, unspaced, duplicate-allowing variable-unit cipher. Their accuracy cannot qualify ours. No frontier-model/mechanistic or historical result is inferred.

Own source-based hypothesis:128particles and64fixed cubic temperature increments should reduce some collapse seen with32/16. That changes both population and schedule together; the new experiment does not causally isolate either scaling intervention. Four mutation moves per increment stay fixed. Original π(K)L(K), source arrays/EOS, compiler and integer sampler remain immutable.

Four arms isolate different remaining questions. `supported_prior` scores an initial supported bank then anneals/resamples without mutation, measuring initialization alone. `row` permits local replacements. `coordinated` uses the original state-independent row/pair/involution/uniform-refresh tickets. `supported_refresh` changes only the1/32whole-key refresh to a uniform supported-prior draw. The last comparison isolates that refresh proposal within the same coordinated mixture. All moving arms have identical maximum proposal slots/table allowances, but actual costs differ when structural rejection avoids scoring; initialization-only is intentionally cheaper. None uses neural proposals or oracle warm starts.

## What fresh means

Generate four new keys and two original-source messages each, using fixed generation seeds and lengths only after registration publication. Use the same positive original model and channel family already studied adaptively. Each pair also gets independently shuffled records preserving exact glyph lengths and unigram counts. There are eight queries, two inference seeds and four arms:64fixed calls. No generation key/source text/length/kind/seed enters a fitter or reader argument. Shared driver RAM does contain the generator fixtures; strict interface and seal tests audit this boundary, not cryptographic separation of processes or human blind review.

These are fresh **development keys** within the same source/channel family, not a language/mechanism holdout. The independent units for generalization are four generated keys; two seeds on the same key do not double the data sample size. Shuffles are a limited order-destroying null, not every structured nonlinguistic alternative. Fixed64/224source-length generation differs from the geometric EOS length prior used for scoring; both arms share that known specification, and true lengths never enter inference. Eventual source/channel changes need separate controls and new experiments.

## Recovery, evidence and failure

Select the highest-likelihood final key by a fixed first-index tie rule, then decode with the already qualified bounded conditional Viterbi reader. This is conditional decoding from a finite bank, not exact global plaintext MAP. Seal every output and the entire output manifest before computing any Gold accuracy.

Positive records measure complete used key, bank coverage of the complete used key, selected used-row matches, full-string edit errors and exact source records. Failed fitting/reading calls remain in their original denominators. Shuffled records have no labelled plaintext accuracy; only their bounded output/status and evidence diagnostics are compared. Multi-seed evidence spreads and paired positive-minus-shuffle margins are reported separately from message quality; correct normalization is not an uncertainty certificate.

The fixed exploratory recovery gate requires all positive calls complete for the new and coordinated arms, at least one complete used key and exact record, ≥10percentage-point used-row improvement and ≥10%edit-error reduction over coordinated. Calibration requires all16new-arm calls complete, each of eight two-seed log-evidence spreads≤2nats and every largest incremental weight≤.5. Null separation requires all eight paired margins>0. Combined qualification needs all three. No threshold is changed after outcome; negative results remain useful. Passing would qualify only this restricted developmental generator/search, not Voynich.

The [experiment design](../experiments/TEMPERED-RECOVERY-001.md) fixes every seed, bound, order, archive and command. Actual scoring caps fail whole calls; they never supply zero/partial likelihood to MH. Time/global/reference/unexpected/integer failures abort without retry. Fullsame-author replay plus independent literal/first-eight source checks and reader path checks distinguish computational validity from recovery evidence.
