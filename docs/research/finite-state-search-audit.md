# Independent audit of bounded channel search

2026-09-27. This audits the implementation proposed in the
[blind-channel design](blind-channel-recovery-design.md), building on
[fixed-support fitting and coding](finite-state-fitting.md). Fixtures are
hand-specified tiny strings and probability models. No source corpus,
recovery benchmark, supplied segmentation or hidden answers are used.

## Objective and assistance

The intended score for a candidate is

```text
J = number of bits in the channel description
    - sum_records log2 p(record | fixed source, candidate channel).
```

The likelihood sums all compatible source strings, channel paths and emission
boundaries. The final score must use the same integer-grid probabilities as
the actual coded candidate. A better pre-quantization EM likelihood does not
establish a better code score. Every complete emitted string is accounted for;
there is no best-path replacement for the sum or unmatched suffix exception.

The source model, source and glyph alphabets, stopping law, code denominator,
maximum states, emission length and alternatives are supplied. Their shared
context is fixed throughout comparisons. The source index is charged under
the declared catalogue but association of that index with the source model
remains the caller's responsibility. The search has no language catalogue
learning, null/deletion process, transposition or manuscript observation model.

Initialization uses every singleton glyph in the declared public alphabet,
including glyphs absent from fitting records, assigned across source-letter
rows with coverage in each state. Its necessary capacity bound is

```text
size of declared glyph alphabet <= source alphabet size * min(denominator, max alternatives).
```

This limits the initializer, not the existence of a valid channel. A channel
using multi-glyph units could fit observations while violating this singleton
initialization requirement. Even when capacity suffices, zero-probability
source letters or context transitions can make a proposed channel impossible;
support must be verified by full observation likelihood and bounded retries.
Keeping declared singletons in the proposal pool allows later candidates to
include absent glyphs; it does not guarantee a fitted candidate retains them
or decodes them correctly. Empty-only fitting records are valid and supply
only stopping-probability evidence.

An optional frequency heuristic initializes the first attempt of the first
one-state restart. It sorts glyphs by fitting-record frequency and source
letters by fixed-source frequency. For an order-one source, this uses the
first 256 survival-weighted marginal distributions, normalized by their sum;
it is a source-only ranking approximation, not the candidate likelihood.
Declared order breaks frequency ties. Remaining starts use the recorded random
seeds, and the heuristic can be disabled. EM is skipped when every row has
one alternative because those row probabilities are fixed at one.

## Search and budget claims

The algorithm uses bounded initializations, state-count choices, restarts and
local proposals. It accepts strict improvements to the conditional two-part
score. Such greedy acceptance supports only the best score found under that
schedule. It does not certify all neighboring models were examined, that the
optimum is global, or that competing channels are identifiable. There is no
integration over model parameters or structures, so the result is not a Bayes
factor or posterior probability of a decipherment.

Count budgets and a fixed random seed define the reproducible proposal
schedule when the deadline does not truncate it. A wall-clock deadline can
stop at different places on different machines. Completed candidates and
proposals must be recorded separately, including failed or incomplete work.
Checks between records do not preempt one exact record computation; EM also
performs an initial full scoring pass, and visible-substring preprocessing is
not interrupted inside its counting pass. The incumbent must always have a
complete score, and interruption must not pair a partial score with a newly
fitted model. A completely scored raw candidate can remain eligible even if
subsequent refinement is interrupted.

## Independent verification method

`tests/test_finite_state_channel_search_independent.py` implements its own
recursive rational path sum using public model probabilities. It independently
counts every model-description field from its cardinality: source index,
state count, initial weak composition, alternative counts, destinations,
emission lengths and literal glyphs, and positive row-weight compositions.
Neither oracle calls production inference, decoder or model-coding helpers.

Every audited returned candidate must have normalized complete rows, unique
retained alternatives, positive grid emission weights, grid-compatible initial
weights, declared destinations and either declared singleton glyphs or bounded
visible multi-glyph emission strings. These
are correctness requirements; passing them is not an empirical recovery result.

Twelve independent tests verify the following:

- All completed raw/refined stages in four seeded restarts are replayed against
  the independent likelihood and bit-count formulas. Each selected stage has
  the smaller actual code score, and the returned global best equals the
  smallest score among all completed stages. This includes empty records.
- Two identical seed/count-budget calls reproduce every candidate and decision
  after excluding elapsed time. Restart ordering interleaves the declared
  state counts. Count budgets include inapplicable proposals.
- The unit pool counts overlapping occurrences, preserves all declared
  singletons, and does not concatenate across record boundaries. Every raw
  initialization covers the declared singletons in every state with self-loop
  emissions, including an explicitly tested glyph absent from fitting data.
- Empty-only records receive their exact stop likelihood. The frequency
  initializer's order-one calculation matches an independent rational sum for
  a source that starts with `a` and then emits only `b`. Ties respect declared
  order, the source remains unchanged, disabling the heuristic uses random
  starts, and singleton rows skip a redundant EM call.
- Insufficient singleton capacity raises an explicit initializer error, while
  impossible zero-source initializations exhaust exactly the declared attempt
  budget and return no fallback channel or score.
- Scripted clocks stop a partially scored initialization, a partially scored
  proposal and refinement before its re-score. The corresponding results are
  no completed model, the previous incumbent, and the completely scored raw
  candidate. No partial score is promoted, and no real sleeping is required.

Validation after the pre-score full-alphabet/frequency-initializer amendment:
`.venv/bin/pytest -q tests/test_finite_state_channel_search_independent.py
tests/test_finite_state_channel_search.py` passed **31 tests in 0.08 seconds**,
including **12 independent tests**. The audit file's Ruff check passed. No
production-code discrepancy was found and no production file was edited.
These tests establish only the checked implementation properties. They do not
measure unknown-channel recovery, source-language identification, search
completeness, nonlanguage rejection or manuscript decipherment.
