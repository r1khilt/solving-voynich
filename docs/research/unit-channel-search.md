# Complete-neighborhood search of unknown deterministic units

`src/voynich/unit_channel_search.py` searches a restricted but nontrivial
ciphertext-only channel family: one state and one nonempty deterministic
literal unit per source letter. Units can have different lengths and can repeat
across letters. The source model, source/alphabet order, glyph inventory,
maximum unit length, stop law, and finite-grid coding context are supplied and
fixed. Source order is zero or one, as in the unchanged original engine.

The algorithm discovers both units and source-row assignments. It receives no
true dictionary, dictionary size, injectivity restriction, segmentation,
plaintext, target-derived source model, or assisted warm start. The shared
declared glyph alphabet is explicit side information, including glyphs absent
from fitting records. This is a bounded local optimizer, not a posterior
sampler, exhaustive key optimizer, or identifiability result.

## API and complete candidate universe

```python
config = UnitSearchConfig(
    seed=0, restarts=16, max_sweeps=80, max_seconds=300,
    batch_size=256, max_units=256, improvement_tolerance_bits=1e-8,
)
result = search_unit_channel(source, records, context, config=config, source_index=0)
```

The public helper `literal_unit_pool(context, max_units=256)` enumerates every
literal string of lengths 1 through the shared maximum L over the declared
glyph alphabet G. Order is length, then Cartesian product in declared glyph
order. The count sum(|G|^l, l=1..L) is checked incrementally against the cap
before any product is materialized. An oversized universe raises a clear
error; it is not truncated or replaced by observed substrings. The example
|G|=6,L=2 gives 42 units. The channel family has U^A assignments for U units
and A source letters, allowing duplicates. The cap bounds the unit pool, not
that much larger key universe.

`channel_from_units(source, context, units)` produces a normalized original
`Channel` from a tuple in source-alphabet order. All row probabilities and the
single initial-state probability are one, exactly on any positive-denominator
grid. `unit_neighbors(units, pool)` exposes the ordered neighborhood.

## Initialization and moves

Every initialization assigns all declared glyph singletons to distinct source
rows. Consequently this initializer requires A>=|G| and rejects smaller source
alphabets. That restriction belongs to the initializer: it is not a claim that
all channels with A<|G| are impossible. Singleton coverage ensures every
ciphertext can be segmented at an initial key when source paths have positive
probability; sources with zeros may still make a start unsupported. Such starts
are traced and skipped, with no hidden repair or oracle call.

Restart zero ranks source letters by **source start probability** and glyphs
by their total fitting-record counts, with declared-order ties. Ranked rows
receive ranked singleton glyphs; when A>|G|, additional rows cycle through the
ranked glyph list. This frequency assignment is a disclosed heuristic, not a
unit-recovery guarantee or an estimate of source stationary occupancy. It
retains absent glyphs. Other restarts independently shuffle source-row and
glyph order, assign each glyph singleton to a distinct row, and fill remaining
rows independently uniformly from the entire unit pool. A master RNG with the
explicit seed supplies a separate recorded 64-bit seed for each restart.

A sweep holds its parent fixed and evaluates:

1. All unordered source-row pairs with unequal units, in increasing index order.
2. Every source row paired with every pool unit other than its current unit, in
   source-row then pool order.

Swaps change two positions and replacements change one. Distinct unequal swaps
change distinct pairs, so these ordered operations produce no duplicate keys.
The public helper additionally removes duplicate pool entries in first-seen
order. With U distinct units the neighborhood has A(U−1) plus the number of
unequal row pairs. For A=23,U=42 the maximum is 1196 candidates, reduced by
equal-unit pairs. Proposals may remove singleton coverage or become impossible;
they are scored as such. Coverage is an initializer property, not a concealed
restriction on the searched family.

Candidates are scored in configurable batches, selecting the strict smallest
score with first-occurrence tie-breaking. The parent moves only when the best
score improves by more than the configured tolerance in bits. There are no
perturbations, annealing, EM steps, stochastic emissions, extra states, or
multirow coordinated dictionary edits in this version.

## Objective and verification

Every comparison minimizes the unchanged conditional two-part score

`actual model-description bits - corpus log marginal / log(2)`.

The likelihood comes from the [batched exact-path kernel](batched-unit-channel.md),
including all segmentations, distinct aliased source contexts, continuation
factors, a fresh source start for each independent record, and each final stop.
Duplicate units are summed probabilistically rather than collapsed into a
single source letter. Impossible keys receive no finite score and remain in
the trace as `status="unsupported", score=null`.

Code lengths use an exact shortcut within this family. First encode an
all-singleton deterministic channel with the unchanged `channel_code_bits`.
Every later key has the same source-choice width, state-count width, length
headers, alternative-count headers, state destination widths, and finite-grid
composition widths. Only the count of literal glyph fields changes. Therefore

`model_bits = encoded_singleton_baseline + ceil(log2|G|)*(sum(unit_lengths)-A)`.

In particular, the per-unit length header does not grow with its value under
the shared bounded code. The formula retains its headers even for unused
source rows. Tests enumerate all small candidates and compare it to actual
`encode_channel` strings. It is not a new model penalty or a free dictionary.

The best completely scored candidate across **all** restarts and completed
batches is retained, including an improvement smaller than the move tolerance.
Before return its `Channel` is scored by the original `two_part_score`, checking
both literal model bits and original-engine marginal against the batched
score. A mismatch beyond numerical slack raises an error. The result exposes
this replayed score. “Exact” means the full normalized path sum; floating-point
rounding still applies, and a zero tolerance is not a symbolic arithmetic
certificate.

## Budgets, traces, and local certificates

Limits are restart count, sweeps per restart, batch size, and elapsed time.
Deadline checks occur before initial scoring and between neighborhood batches.
Each kernel call scores an entire batch atomically. If it crosses the deadline,
every candidate in that completed batch remains eligible for the returned
best model. A partially evaluated neighborhood is logged as incomplete; no
unscored candidate or partial likelihood can displace the incumbent.
Preprocessing, one batch (including numerical fallback), trace construction,
and the final original-engine replay can exceed the cooperative time budget.
Hard external CPU/process deadlines remain the caller's responsibility and can
terminate the process before a result is returned. No approximate beam or
resource-triggered score is substituted.

A completed sweep with no improvement greater than tolerance certifies only
its exact parent against these two neighborhoods. The result field
`best_is_certified_local_optimum` is true only when the returned unit tuple is
identical to a parent with such a completed certificate. A better subtolerance
neighbor that was retained globally does not inherit its parent's certificate.
A partial sweep creates no certificate, though a completed certificate for
the same returned key from an earlier sweep remains valid. Exhausting time or
sweeps is not proof of local or global optimality.

`UnitSearchResult` includes `channel`, replayed `score`, `units`, `unit_pool`,
`trace`, `stop_reason`, `evaluated_neighbors`, `scored_initializations`,
`completed_sweeps`, `local_optima`, `best_is_certified_local_optimum`, `seconds`,
`config`, `source_index`, and aggregate `kernel_fallbacks`. `to_dict()` produces
JSON-compatible data without nonfinite scores. With no supported completed
initialization, channel/score/units are null. Empty record collections are
rejected; one or more empty records are valid independent observations.

Restart traces store initial units, method, restart seed, score/status, and
kernel diagnostics. Each sweep stores its parent and score; every completed
neighbor's move, unit tuple, score/status; batch counts/diagnostics; expected
and completed neighbor counts; completion, acceptance, selection, and stopping
reason. This deliberately retains rejected and unsupported proposals. The
kernel's rolling-buffer memory bound **does not bound trace memory**: trace
storage grows with all scored proposals, their A units, and score/move fields.
Large budgets may produce large in-memory and serialized traces. Chunking
bounds numerical batch working memory, not the accumulated audit record.

## Artificial validation and limitations

Own tests cover exhaustive tiny unit sets and encoded objectives, complete
neighbor-set equality/deduplication, replay of every traced score, deterministic
count-budget runs, batch-size invariance, initial singleton coverage including
absent glyphs, random duplicate units, source zeros, empty records, empty
neighborhoods, subnormal-source fallback counts, cap rejection before product
materialization, and fake-clock interruption. The latter verifies that the
best candidate from a finished batch survives timeout and still receives the
original-engine check. A deliberately corrupted kernel score is rejected at
that check. Independent coverage is recorded separately in
[the unit-search audit](unit-channel-search-independent-audit.md).

A small exact counterexample supplied by the independent artificial audit
illustrates the limit of even a complete certificate. Let source starts be
(a,b)=(3/4,1/4), transitions after a=(1/8,7/8), after b=(5/8,3/8), rho=1/4,
glyphs=(x,y), maximum unit length two, and observe `xxxx`. The duplicate-unit
key `(xx,xx)` has marginal 9/64 and is strictly better in the coded objective
than every unequal swap or one-row replacement. Yet changing both rows to
`(x,x)` gives marginal 81/1024, loses only 0.830075 data bits, and saves two
literal-model bits: the total improves by 1.169925 bits. This coordinated
change is absent from the current neighborhood. A local optimum therefore
does not establish that a dictionary or source is unlearnable.

The kernel's bounded synthetic timing is documented in its linked note; no
empirical files, target scores, or actual search throughput were used to design
this implementation. Real throughput also depends on trace allocation,
neighborhood length distributions, numerical fallbacks and the final replay.
Higher-order sources, coordinated dictionary moves, dense auxiliary support
and explicit projection, and source-versus-optimization controls remain
separate interventions described in
[the inference review](blind-channel-next-inference-review.md). This adapter
specializes existing exact forward scoring and elementary full-neighborhood
hill climbing; it makes no new global convergence claim.
