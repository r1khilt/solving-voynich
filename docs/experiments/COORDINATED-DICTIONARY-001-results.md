# COORDINATED-DICTIONARY-001 results

**Recovery and calibration failed.** Corrected support-oriented initialization reduced population extinction; coordinated moves gave somewhat better incorrect guesses. No arm recovered a complete used generator key, a bank containing such a key, or an exact source message. The computational replay audit passed. This is fresh-key synthetic development evidence, not decipherment or a competent neural solver.

Registration commit `6d19bea29c4baa0b28cf62fdc813d655d9596e06` was pushed and exact origin/main verified before the single run. All207 frozen paths stayed unchanged through the single full audit. See [fixed design](COORDINATED-DICTIONARY-001.md), [method](../research/coordinated-dictionary-2026-10-01.md), [machine-readable result](../../results/COORDINATED-DICTIONARY-001/result.json) and [audit](../../results/COORDINATED-DICTIONARY-001/audit.json).

| Arm | Complete /8 | Extinct | Call work caps | Correct used rows /152 | Reading edits /2304 true letters | Exact messages /16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prior + row moves | 3 | 4 | 1 | 6 | 2233 | 0 |
| Corrected covered start + row moves | 5 | 0 | 3 | 4 | 2373 | 0 |
| Corrected covered start + coordinated moves | 5 | 2 | 1 | 13 | 1937 | 0 |

Failed fits/readers contribute zero correct used rows and full true-message length edit errors, as registered. Successful edit distance can exceed true length due to insertions; the edit totals are not token accuracy. The hidden-key denominator counts source rows actually used by the generator, separately from plaintext character errors. All13 completed calls obtained literal, scalar-score and marginal-checked conditional Viterbi readings; none was exact. All24 predictions were sealed before gold metrics.

The coordinated arm has18.37% fewer edits than the covered-row control and an absolute used-row-match gain of5.92 percentage points (8.55% vs2.63%). The unchanged support criterion also requires all eight calls in each arm complete, a10-point used-row gain and at least one exact used key. These clauses failed. Incomplete calls have different work; this is neither a matched-CPU win nor a successful decoder. Prior-row is a separate initialization control on the same new keys, not a comparison with old exposed-key results.

All arms failed calibration: maximum incremental weight reached1. The two short-case evidence spreads were60.2143/8.48486nats for covered-row and83.4564/56.8850 for coordinated, above the≤2 criterion. Long-case spreads are undefined because calls failed; every prior-row case has at least one failed seed. We cannot interpret these populations as stable posterior coverage or their log-evidence estimates as dependable language evidence.

## Why this still narrows the problem

The covered-row start avoided extinction but three long calls consumed their full fitting work allowance. Better support can create expensive incorrect source graphs rather than a solution. Coordinated moves crossed the separate exact finite support barrier, but did not select the necessary code inventory in these actual searches.

The [read-only post-outcome inventory diagnostic](../../results/COORDINATED-DICTIONARY-001/post-outcome.json) found that **none of416 final particles** contains enough instances of the actual used generator codes to match every used source row by any permutation. Selected keys need at least3–12 code replacements before such a permutation would be possible. These counts use gold after fitting; they are not blind proposals or proof of a particular large-case causal barrier. All six extinctions occurred before final EOS.

The source, exact likelihood calculation, correctly weighted initialization and elementary invariant moves have separate qualifications. Their correctness does not establish that a32-particle search reaches the important modes, that the generator key is uniquely identifiable, or that this channel describes Voynich. No new learned-model training or neuron interpretation was performed.

## Resources, stop semantics and audit

The actual run took598.567174wall/597.698177CPU seconds with866,320,384bytes peak host RSS. It booked368,800 dictionary-table requests, made733,517 native record calls and accumulated7,926,280,873 fitting edges. Cumulative nodes across all calls are3,065,756,026; maximum single-record nodes39,314; maximum owned bridge envelope78,125,184bytes. A bank request books its tables before scoring individual keys, so failed requests can contain unevaluated keys.

All five fitting work caps were explicitly typed per-call cumulative exhaustion, each charging1,000,000,001edges including the first rejected edge. The registration explicitly allows these failed calls and continuation of remaining predetermined calls; the global fitting bound16,000,000,016edges was not reached. No per-record graph cap, global abort, retry, retuning, refilling or manufactured partial likelihood occurred. Reader caps were not reached. The conservative reader edge-work bound is separate from fitting counters.

The one full audit took623.757732wall/622.827526CPU seconds with1,256,538,112bytes peak host RSS. It replayed fresh fixture generation, all24 RNG/native/status/trace/counter/archive/selection/reader/recovery/gate results. Every416 final key from completed calls received an independent Python contextual dynamic-program score; maximum absolute log-score difference6.36646291241×10^−12nats, below1×10^−7. Same-author replay shares search/native code; it is not independent expert review.

Both processes stayed within1800wall/1600CPU/2GiB host bounds. Total ignored compressed gold and banks/traces282,182bytes, below128MiB. Python3.12.13/NumPy2.5.3/macOS arm64; thread-limit environment variables were unset and recorded, not claimed explicitly fixed. Paid spend$0; no GPU allocation, downloads or new training. Full preregistration suite2756tests+23subtests passed,13skipped,175.40s. Nine new code/test files pass scoped lint; the same five unrelated full-tree lint findings remain.

## More targeted successor, not yet implemented

[Observation-conditioned inventory counting](../research/observation-conditioned-inventory-2026-10-01.md) develops exact prior conditioning on the actual visible-prefix support. Own56profile/686conditional-path/168source-evidence checks passed. A generic four-glyph example counts supported23-row dictionaries with8192 availability masks, rather than enumerating42^23 keys. A separate224-check occupancy identity avoids treating code-presence variables as independent Bernoulli events.

A bounded Boolean-diagram compiler may extend the counts to whole records. It would need exact occupancy/cardinality weighting, verified conditional sampling, explicit intermediate targets, caps and new empirical registration. No such compiler or successor sampler is implemented or queued. Even eliminating structural extinction would leave concentrated language likelihoods and incorrect label assignments to solve. Voynich remains unsolved.
