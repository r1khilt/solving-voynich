# Independent audit of the restricted bijective channel search

2026-09-27. This concerns `src/voynich/bijective_channel_search.py` as a separate
possible follow-up to the original development pilot. No pilot ciphertext,
corpus, fitted pilot model, transfer material or answers were opened, and none
of the original eight fits was modified. All checks use tiny hand-specified
models and strings. The [blind-channel design](blind-channel-recovery-design.md)
and [general search audit](finite-state-search-audit.md) remain the broader
scope and evidence standards.

## Exact score within the supplied restricted family

The family supplies one channel state and a deterministic bijection between
the complete source and glyph alphabets, with exactly one emitted glyph per
source character. Thus each supported ciphertext has exactly one compatible
source path. This restriction removes segmentation and channel-state ambiguity;
it is substantial assistance and does not discover an arbitrary verbose cipher.

Let `f(g)` map glyph `g` to a source character, `R` count records including
empty records, `N` count all glyphs, `B_g` count record starts, and `C_gh` count
within-record directed glyph pairs. For a fixed order-one source and stop
probability `rho`, the exact log likelihood is

```text
R log rho + N log(1-rho)
  + sum_g B_g log p(f(g) | start)
  + sum_(g,h) C_gh log p(f(h) | f(g)).
```

For an order-zero source, replace the start/pair terms by the total glyph
counts times their mapped source log probabilities. Empty records contribute
their stopping factors. Boundaries never create a pair between separate
records. Complete alphabets, including glyphs absent in fitting data, remain
part of the bijection and the model description.

Swapping two mapped source labels affects only their start/unigram terms and
directed pairs touching either glyph. Counting their two full rows followed
by their columns in every other row includes each affected pair exactly once,
including both self-pairs and both directed cross-pairs. The stopping terms
cancel. This gives the exact score difference using O(alphabet size) terms.
Source-zero factors make some permutations unsupported; these are skipped as
initializations. A supported-to-unsupported swap has negative infinite gain.
An undefined difference between two zero-probability affected products raises
instead of producing a NaN.

Every bijection has the same finite model-code length under one fixed coding
context: row counts, state count, unit lengths and probability compositions
are identical, and each literal glyph index has the same field width. The
actual code is still charged. A four-letter fixture with three source options,
at most two states, at most three alternatives and emission length at most two
costs exactly 23 bits for every permutation, including its absent fourth glyph.
Returned scores are finally recalculated with the original exact channel
engine and compared with the sufficient-statistic likelihood.

## What completed and interrupted neighborhoods establish

A completed sweep evaluates each unordered pair once and can establish only
that its parent has no pair-swap improvement above the configured tolerance.
A sweep interrupted by the clock may retain its best finished neighbor, but
cannot certify a complete neighborhood. Preprocessing, one swap calculation
and the final whole-engine score check are cooperative atomic operations and
may overrun the nominal time budget.

The best finished key is retained across restarts, even when an improvement
falls below the move-acceptance threshold. Consequently, the returned key can
differ from the parent of the last completed no-sufficient-gain sweep.
`pair_local_optima` counts those certified parents;
`best_is_certified_pair_local_optimum` is true only if the exact returned
mapping matches such a parent. A zero-sweep budget grants no certificate.
Locality is always relative to pair swaps and the declared numerical tolerance.

Pair-local optimality is not global optimality for an order-one source. An
explicit three-letter counterexample starts with probabilities
`p(a,b,c)=(1/4,1/2,1/4)` and transitions around `a -> b -> c -> a` with
probability `3/4`, with each other transition having probability `1/8`.
For observed `XYZ` and `rho=1/2`, map `XYZ -> abc` has probability `9/1024`.
Every single pair swap lowers that probability, yet the three-cycle mapping
`XYZ -> bca` has probability `9/512`, twice as large. A completely checked
pair neighborhood therefore cannot license a global optimum claim.

## Independent tests and outcome

`tests/test_bijective_channel_search_independent.py` obtains its reference
likelihood by directly multiplying rational source and stopping probabilities
along each decoded record. It does not use production sufficient statistics,
swap deltas or inference to obtain expected values.

Its 11 tests cover all 24 permutations and all six swaps for both order-zero
and asymmetric order-one four-letter fixtures, including repeated self-pairs,
directed cross-pairs, an absent glyph, multiple records and empty records.
Every recorded search neighbor and final score is replayed independently.
Other checks cover constant family code cost, impossible swaps, zero-sweep
and empty-only certificates, and the non-global three-letter example above.

Two scripted-clock cases interrupt after the first of three neighbors, with
and without a finished improvement. Both retain the best completed key, grant
no local certificate, and execute the final original-engine check. A separate
large-tolerance case verifies that a better evaluated neighbor can be returned
even though the sweep rejects the move, without attaching its parent's local
certificate to the returned key.

Validation command:

```text
PYTHONPATH=.:src .venv/bin/python -m pytest -q tests/test_bijective_channel_search_independent.py tests/test_bijective_channel_search.py
```

Result: **32 passed in 0.08 seconds**, including **11 independent tests**.
The independent file's Ruff and diff checks passed. No production discrepancy
was found. These are implementation and mathematical controls, not pilot
recovery results, new empirical evidence about the manuscript, Bayesian model
probabilities, or a decipherment.
