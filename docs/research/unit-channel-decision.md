# Fixed-channel posterior samples and edit-risk decisions

2026-09-29. Implementation and artificial validation only. This module does not
fit channels, change source probabilities, inspect empirical ciphertext or
answers, or execute the registered diagnostic. See
[DEV-004](../experiments/BLIND-CHANNEL-DEV-004.md) for the separate empirical plan.

## Method and source basis

[Kumar and Byrne 2004](https://aclanthology.org/N04-1022.pdf), §3, distinguishes
the most probable complete output from a decision minimizing expected task
loss. Their practical translation decoder uses an n-best approximation and
different task losses. [Eikema and Aziz 2022](https://aclanthology.org/2022.emnlp-main.754.pdf),
§§2–3, separates candidate exploration from Monte Carlo estimation of utility.
Both primary PDFs were opened. Their translation results do not predict cipher
accuracy; the useful principles are loss alignment, posterior sampling, and
separate candidate/reference budgets. The finite-channel derivation below is
specific to this repository, not a claim copied from these papers.

Let `u(a)` be one fixed nonempty unit per source character, `h+a` the updated
source context, `c=1-rho`, and `y` the observed record of length `n`. Define:

```text
beta[n,h] = rho
beta[i,h] = c * sum_a P(a|h) * 1{y starts with u(a) at i}
                         * beta[i+len(u(a)), h+a].
```

An overrun cannot match. Nonempty emissions make this an acyclic backward
recurrence; `beta[0,start]=p(y)`. The implementation evaluates it in log space,
including zero source probabilities, and retains every source-letter context
when different letters emit the same unit. Order zero has one context; order
one has the start context plus each source letter.

At a supported node, sample a letter with probability proportional to its
summand, emit its unit, and advance. Multiplying these conditional probabilities
telescopes the backward factors, leaving the full path probability divided by
`p(y)`. The final `rho` is already in the endpoint. Sampling stops only at the
record end: no glyph truncation, supplied source length, or intermediate stop
is allowed. Empty ciphertext has probability `rho` and the sole plaintext `""`.
Impossible observations have no posterior and return no samples.

This is unpruned posterior sampling, not beam or top-k sampling. Floating-point
log sums, categorical conversion, and Python's finite-resolution PRNG remain
numerical limitations; probabilities are not symbolic rationals at runtime.
Tiny conditional outcomes can be below representable sampling resolution.

## Decision and public interface

`posterior_sample_plaintexts(source, units, record, stop_probability, draws=...,
seed=...)` returns `PosteriorSamples`. `sample_mbr_decode` accepts the same
fixed model and keyword `candidate_draws=32, risk_draws=256, seed=0`, returning
`MBRDecision`. Inputs are ordinary order-zero/one `SourceModel` objects and
unit strings in `source.alphabet` order. Zero candidate draws are allowed;
the reference bank must contain at least one requested draw.

MAP comes from the unchanged original `viterbi_decode`, preserving its tie
behavior. Each plaintext determines exactly one path in this deterministic
one-state family, so joint MAP is also plaintext MAP. The candidate set is MAP
followed by unique candidate-bank samples in first-occurrence order. Every
candidate is scored against a separately generated reference bank using mean
**unnormalized Unicode-codepoint Levenshtein distance**. Reference duplicates
retain their frequencies. Integer total distance selects the winner, with
first-candidate ties; therefore a tie favors MAP. No normalization by sampled
length or extra model-probability weighting is applied.

The Myers bit-vector recurrence already used in
`scripts/evaluate_naibbe001.py` is reused locally with cached candidate masks,
avoiding a dependency on evaluator scripts. Equal reference strings are counted
once computationally and weighted by their exact multiplicities.

The winner minimizes estimated risk over this finite candidate set. Its
estimated risk cannot exceed MAP's on the same bank; that is a construction,
not evidence of better true decoding. For a fixed candidate, independent
reference draws support Monte Carlo risk estimation. Choosing the minimum on
that bank introduces selection optimism. This is not an exact global Bayes
optimum, a calibrated historical reading, or a posterior over unknown channels.

## Reproduction and serialization

Two RNG seeds are SHA-256 hex digests of UTF-8 strings
`unit-channel-decision/v1:candidates:{seed}` and
`unit-channel-decision/v1:risk:{seed}`. Each digest, converted to an integer,
seeds a distinct `random.Random` instance. Changing one bank's size cannot
change the other bank. Samples remain in draw order.

`MBRDecision.to_dict()` includes `status`, `plaintext`, `map_plaintext`,
`log_likelihood`, `map_log_probability`, `estimated_risk`, `map_estimated_risk`,
requested draw counts, input seed, both derived seed hex strings, full
`candidate_samples`/`risk_samples`, and both bank hashes. `candidate_risks`
contains ordered `{plaintext, total_edit_distance, mean_edit_distance}` rows;
`tie_break` is `map_first_then_candidate_first_occurrence`. Bank hashes cover
UTF-8 `json.dumps(list(samples), ensure_ascii=True, separators=(",", ":"))`.
Impossible results use null decoded strings/log scores and empty banks/risks,
so JSON serialization needs no nonfinite values.

The backward array requires `O(n*A)` memory for order one, plus transition and
choice caches bounded by `O(n*A*A)` and retained sample strings. Draw counts
are explicit but not hard-coded caps. Callers own external CPU/memory limits;
there is no cooperative mid-call deadline or empirical runner in this module.

## Artificial validation and bounded timing

Author tests compare every sampled path's product of conditional edge
probabilities with independently enumerated Fraction posteriors, for both
source orders, variable units, duplicates, gaps, Unicode/NUL, and empty records.
They also cover impossible observations/source zeros, a 2,000-glyph underflow
control, seeded sampling frequencies, opposite-bank-size invariance, hashes,
risk multiplicities/ties, and scalar dynamic-programming edit-distance parity.
An explicit three-reading distribution shows MAP can differ from edit-risk
selection. A rare aliased context survives 1,100 positions before uniquely
supporting the suffix. **36 author tests pass; Ruff passes.** Independent
audit coverage is reported separately by its author.

One artificial timing process used Python 3.12.13, NumPy 2.5.3, macOS 26.6.2
arm64, single-thread settings, a 60 CPU-second
hard limit, no paid resources, and no files from any experiment. Alphabet was
the first 23 lowercase ASCII letters; start weights were `1..23` divided by
276. Context letter index `k` assigned next-letter index `j` weight
`((j-k)%23+1)/276`. A `random.Random(773)` sequential weighted draw generated
225 source characters with these rows. This fixed length is a timing fixture,
not a sample from the stopping law. Inference used `rho=1/225` and seed 773,
32 candidate draws and 256 reference draws.

The variable dictionary took index `(13*i)%42` for each row from the complete
length-one/two Cartesian pool over `ABCDEF`, then replaced its last three units
with its first three. The collision dictionary emitted `X` from every row.
Both timings include lattice construction, original MAP, sampling, distances,
and serialization; imports and fixture generation are excluded.

| Fixture | Glyphs | Unique candidates / references | Wall / CPU seconds | Process peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Variable units | 398 | 33 / 256 | 1.1499 / 1.1475 | 31,211,520 bytes |
| All units identical | 225 | 33 / 256 | 1.2954 / 1.2947 | 50,282,496 bytes |

Peak RSS is the cumulative macOS process high-water mark, not an incremental
allocation. These are artificial throughput observations, not a guarantee for
the empirical records. The input SHA-256 values (sorted-key default JSON of
source, units, record, rho) were respectively
`d9570f41b13b4518cb81d4826efcd1ba5c533673f31230ef26e4af63c9b0c735` and
`649867ccc175243d1efc0227e74dc9d398cf657159d36c62d9fbc1e6612210e0`.
