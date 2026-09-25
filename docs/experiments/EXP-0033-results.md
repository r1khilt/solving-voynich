# EXP-0033 result — exact masks are nonunique, but most positions are identifiable with the true source

**Registered descriptive decision: not substantial by the prespecified joint criterion.** All **198/198** Polish copy-mutate examples have more than one keep-mask alignment even when the true pre-null cipher stream is supplied. The median has **48** valid alignments, but **91.7%** of nonspace positions are forced to the same keep/delete status across *all* alignments (row-bootstrap 95% interval **[91.13%, 92.30%]**). Thus literal full-mask uniqueness fails universally, while the stricter registered substantial-ambiguity rule fails because mean nonspace forced fraction is **above** its 90% cutoff. This is an exposed synthetic diagnostic, not evidence of Voynich nulls.

## Frozen setup and validation

The [pre-output registration](EXP-0033.md), exact-integer dynamic program, independent recursive auditor and exhaustive small-string tests were committed as `9c7437ea35dd1fd1e99e2c1b1c2bf9be09d76e49` and verified on `origin/main` before the first alignment count. Input was the 600-row Polish holdout used in EXP-0032, SHA-256 `ee78f08814a993c42e0558e5028cfdaa4d1c1e2812c9e56d82585a1e8b255664`; the independently checked true-source file was pinned by source-audit SHA-256 `b8f44e509a759d2661fdfde383ddf5a5fa4390b6a054d7c6c6c6b7cd0ab6b2a0`. The gold keep mask was verified to spell the supplied source in every row. Exact count means the number of positional subsequences of the observed 128-character string that spell this same source, without imposing a probability model on the generator. No model was trained, no paid API used, and no manuscript text scored.

The run completed in 1.84 seconds, below its 180-second cap. A separately written recursive prefix/suffix auditor replayed **all 600 rows**, agreed on every exact integer alignment and position marginal, recomputed aggregates/bootstrap/decision, and passed in 4.37 seconds. Result JSON SHA-256 `4f6ac9db7343ac11a6d9e0d94b329b0fc4982cdbe00c5e5f4faa4e2c19d21626`; audit JSON SHA-256 `aceed8ecff0b8e5f32c5879bb51596e6349dbca1fb46c9a387870a7bc8bf1c26`. The source-frozen full suite had passed **1,126 tests, eight skipped and 23 subtests**; the exact DP also matched exhaustive enumeration on small strings.

## Observations

| Synthetic filler family | Rows | Unique exact alignment | Median alignments | Nonspace positions forced | Rows with adjacent equal nonspace signal/null swap | True-source uniform-alignment top-count null F1 | Matched-count random null F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random character | 200 | 26.0% | 2 | 97.4% | 63.5% | 0.981 | 0.293 |
| Periodic | 202 | 61.4% | 1 | 97.9% | 38.6% | 0.988 | 0.293 |
| Copy-mutate | 198 | **0.0%** | **48** | **91.7%** | **87.9%** | **0.933** | **0.318** |

Copy alignments range from **2 to 12,288**. Every copy row has an alternative complete mask, and 174/198 have an immediately visible alternative: adjacent identical *nonspace* glyphs on opposite sides of a signal/null boundary can trade labels without changing the retained string. The 198-row bootstrap interval for exact uniqueness is **[0, 0]**. Stripping terminal spaces changes neither the copy unique fraction nor its mean nonspace forced fraction, so padding does not cause the main observation. This sensitivity may itself remove real terminal spaces and is not the primary definition.

Giving an oracle the true source stream and choosing the highest uniform-alignment keep marginals at the true retained length produces copy null F1 **0.933**, versus **0.318** for a count-matched random mask. This is a useful *descriptive* localization check, but the distribution over valid masks is artificial and the source stream is supplied. It is neither the true channel-weighted posterior nor an upper bound comparable to a model that sees only the observed glyphs.

## Implication and limit

The zero exact-mask successes in EXP-0032 cannot be interpreted alone as complete failure to recover useful structure: exact-mask scoring demands one particular mask in a case where the observed string plus correct source usually admits many. Yet the high forced-position fraction means alignment ambiguity by itself is **not** a sufficient explanation for EXP-0032's weak copy F1 near 0.47–0.48. The gap between that source-free model and this true-source calculation plausibly involves source inference, a different channel posterior, or model capacity/objective/decoder limitations; this experiment does not allocate blame among them.

The next defensible test is a **channel-weighted source oracle**: condition on the true source but evaluate the actual copy-mutate generator's path probabilities and posterior mask uncertainty, then separately study how well a model can infer the source string without being given it. That requires a new registration, independent generator replay, and a fresh source if used for prospective model selection. No historical decipherment claim follows from this synthetic alignment exercise.
