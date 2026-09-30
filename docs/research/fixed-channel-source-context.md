# Isolating language-model errors from dictionary errors

Prepared 2026-09-28–29 for [DEV004](../experiments/BLIND-CHANNEL-DEV-004.md).
This implementation note contains mathematics and artificial verification,
not empirical recovery results.

The previous search returned a dictionary whose normalized marginal likelihood
beats the generating dictionary but whose decoded text has more errors. Two
different interventions are meaningful: improve the source probability law,
or choose a reading to minimize a different loss. Neither can restore a true
text that the fixed dictionary cannot encode. Three separate controls therefore
hold the dictionary constant.

## Source law and exact inference

For maximum context length d in 0–3, the source alphabet A has a probability
row for every history h of length at most d. Let C(h,a) count occurrences of
the complete string ha inside an individual training record, with C('',a)
the unigram count. The smoothed base distribution is

    p(a|'') = [C('',a) + 1/2] / [N + |A|/2].

For nonempty h, recursively use

    p(a|h) = [C(h,a) + tau p(a|suffix(h))] / [sum_b C(h,b) + tau].

Here suffix(h) removes the oldest character. Summing over a proves each row
normalizes; a positive base and tau make every probability positive. Unobserved
contexts reduce to their suffix row. Short histories at the beginning of each
record are retained explicitly. Records are not concatenated for counts or
validation. At d=1 this exactly reproduces the existing Dirichlet-backoff model.
This is a simple hierarchical interpolation choice, not an assertion that it
is an optimal Latin model or a Bayesian integration over its parameters.

A fixed dictionary u maps each source character to a nonempty glyph string.
For plaintext x that concatenates to ciphertext y, its joint mass is

    rho * product_i [(1-rho) p(x_i | preceding history of length <=d)].

Stopping is geometrically distributed and almost sure; this gives a normalized
law over finite plaintexts and therefore over their deterministically encoded
ciphertexts. The artificial fixed-length excerpts are not generated from this
stopping law, a pre-existing and disclosed mismatch.

The forward state (glyph offset, source history) is sufficient for future
probabilities. Add every matching dictionary edge, combining prefixes by
log-sum-exp for the marginal and maximum for the most likely reading. Nonempty
units strictly increase glyph offset, making the graph acyclic. Duplicate units
remain separate source-letter edges. No candidate reading is pruned. The state
cap raises an error rather than silently approximating. Because one plaintext
determines one path in this family, best joint path also means best plaintext.
This equality does not generally hold for stochastic or multi-state channels.

The implementation uses log arithmetic, including tiny contexts that would
underflow in an unscaled probability recurrence. Exact means unpruned finite
inference, not exact real arithmetic or a global optimum over dictionaries.
An artificial 23-letter/order3 source, random six-glyph 400-character record
and 23 one/two-glyph units took 0.00298 seconds for 1,224 reachable nodes and
1,780 edges. This is a single non-empirical throughput check, not a worst-case
runtime guarantee.

## The best possible reading under a fixed dictionary

In an answer-only evaluation, let t be the true text. We can compute

    min edit(x,t) over all x with encode_u(x)=y

without consulting source probabilities. The product graph has state (i,j),
where i glyphs and j true characters have been consumed. A deletion advances j
at cost 1. For every letter a whose unit matches at i, an insertion advances
i by its unit length at cost 1; a match/substitution advances both i and j at
cost 0 or 1. Each edge increases i+j, so integer dynamic programming reaches
the global minimum over all compatible readings and all their edit alignments.
An impossible ciphertext is distinct from an empty decoded string.

A positive minimum proves that this dictionary excludes exact truth, even if
a perfect reading rule were available. A zero minimum does not prove the truth
should have maximal source probability. With the positive trained source every
compatible reading has positive mass; with source zeros this dictionary-only
minimum is merely a relaxed lower bound. This oracle diagnostic is never an
answer-free algorithm and is forbidden as a channel-selection objective here.

## Why these tests precede a larger cipher search

[Nuhn et al. 2013](https://aclanthology.org/P13-1154.pdf) demonstrates, for its
supplied-unit substitution setting, that stronger language context can matter
more than exact optimization of a weaker objective. Our source-only selection
and fixed-dictionary comparison tests the relevant mechanism without importing
their cipher assumptions or reported accuracies. The separate
[sampling decision note](unit-channel-decision.md) treats expected edit loss.
The [prior Voynich/decipherment review](blind-channel-recovery-design.md) remains
the broader rationale and limitation record. No source-language or historical
encoding inference is made by fitting Latin to these known Latin controls.
