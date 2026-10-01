# Exact contextual prefixes using the existing native marginal kernel

2026-10-01. Own implementation following [unstable IID inference](../experiments/DICTIONARY-SMC-001-results.md). The next question is whether the actual order12source can be used in prefix-conditioned dictionary inference with verified costs. This is a contextual evaluator, not a recovered cipher or a new language model.

## Literature reviewed before the benchmark

Nuhn, Schamper and Ney2013, abstract/introduction and related-work/definition passages: their letter substitution experiments motivate distinguishing a stronger language model from more exact optimization of a weaker one. The supplied 1:1/homophonic unit assumptions differ from duplicate deterministic source-row emissions of length1or2, and we do not transfer their reported accuracies or beam scoring function. [Primary paper](https://aclanthology.org/P13-1154.pdf). Previous fixed-source comparisons remain in [the source-context note](fixed-channel-source-context.md).

Del Moral, Doucet and Jasra2006, selected common-space targets and§3.3.2.3eq30–31 revisited: prefix target supports must contain the final target support, and invariant move weights are evaluated before moving. This motivates checking the observation law before launching another sampler. No new convergence or variance theorem is asserted for our decoder. [Primary paper](https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf).

Hauer and Kondrak2016, original Voynich§5.4/conclusion discussion already reviewed for this pipeline, refreshed direct PDF: incoherent outputs and possible anagram/LM artifacts restrict historical interpretation. Their language/anagram/abjad assumptions do not establish Latin or this variable-unit channel. [Primary paper](https://aclanthology.org/Q16-1006.pdf). Selected-section reviews, not complete theorem audits or reproductions.

## Censoring identity, our derivation

For a fixed dictionary and starting source context, enumerate every source prefix that reaches c observed glyphs for the first time. Each step emits its whole unit, but the observation compares only the visible part before the cut. Unseen glyphs after the cut are not required to match. The likelihood is the sum of the products of source-row probabilities and continuation factors(1−rho) for these minimal-crossing source paths. G0=1. The geometric source terminates almost surely, so the conditional sum over all unobserved continuation tails is1.

The existing native suffix function receives matching source-letter edges and their destination offsets, not dictionary strings themselves. Construct the normal matched edges before c. For a unit crossing c, check its visible part and clamp its destination to c. Run the SAME kernel with the actual starting context, original source probabilities and transitions. It sums each minimal-crossing path and multiplies the result by its usual terminal rho. Subtract log(rho) from the returned log likelihood to obtain logG_c. No probabilities, source rows, goto states or dictionary values are approximated or changed.

For final exact closed likelihood, require the full unit to fit inside the observation and retain the native terminal rho. An offset at the original record end means EOS was paid before this conditional future; its factor is1. Shared fixed keys multiply both remaining-record factors. Intermediate full-observation prefixes may allow another hidden glyph from a crossing unit; the separate closed stage correctly rules those paths out.

This reuses the already qualified C++ kernel and binary unchanged. Closed likelihoods agree with its original string matcher. That comparison alone shares a backend; independent Python source-state DP and full rational source-string enumeration provide separate correctness evidence. The old C++ kernel requires strictly positive source probabilities, and the wrapper rejects source zeros explicitly. The general Python evaluator supports zeros; it is not silently substituted.

## Memory and finite work

The original context count archive lists1,447,724states and23letters. Dense float64probabilities+uint32gotos occupy399,571,824bytes. The new [wrapper](../../src/voynich/native_censored_bridge.py) pins those immutable arrays and their matching native pointers without copying them. It detects pointer/array mismatches and captures stable function/pointer references. Its128MiB owned bank/match/native graph envelope excludes the pinned source; the separate2GiB process peak-RSS guard includes it and loader overhead. The Python reference explicitly receives768MiB to cover its copied source and scratch, unlike the earlier128MiB IID calibration budget.

Tables, native edges, per-record nodes/edges and conservative native graph/match allocations are bounded. Graph caps raise errors, never return an invented zero or approximate partial marginal. Exactly one backend comparison and complete replay per registered input; no runtime-driven extensions. The original compiler/binary/source hashes stay bound by the existing source qualification.

## Evidence and next state

Native tests cover all36two-row keys, allthree supplied contextual rows, every binary observation length1..3and every nonempty prefix/final stage, plus two-record conditional offsets/paidEOS, the mid-pair witness, original closed graph/node/edge equality at every context, source pinning/corruption, state/edge/table/allocation guards and actual contextual SMC trajectory parity. Independent rational source-string comparisons supply5184native values; zero cuts are also covered by conditional-offset cases.

[CENSORED-CONTEXT-001](../experiments/CENSORED-CONTEXT-001.md) will benchmark the original full source on fixed previously saved dictionary particles. No new SMC, key search, reader, training or manuscript inference is launched. Adequate throughput is an empirical question: a valid native evaluator may still be too costly for millions of contextual proposals. Successful engineering would permit a bounded contextual recovery design; it would not repair the failed IID calibration or establish decipherment.
