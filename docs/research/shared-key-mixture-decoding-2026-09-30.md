# One uncertain key shared across every record

2026-09-30. Implemented and validated on artificial inputs; no empirical key bank used yet. Motivation:
the exposed CONFIRM001 support failure and the new qualified supplied-key reader.
This is a proposed bridge toward unknown-key recovery, not evidence that Voynich
uses any of these alphabets, languages or memoryless emissions.

## Prior work and the specific problem

[Ravi and Knight (2011), section3.1](https://aclanthology.org/P11-1025.pdf)
use a Bayesian source/channel formulation and type-level sampling to impose
consistency across repeated cipher symbols. Their supplied-symbol substitution
model and word resources differ from our ambiguous variable-length units. Their
reported final sample is not the finite-bank marginal decoder derived here.
The paper motivates handling shared uncertainty, not a claim that our posterior
approximation is already calibrated.

[Mohri, Pereira and Riley's weighted-automata discussion, section2.4](https://cs.nyu.edu/~mohri/pub/hbka.pdf)
distinguishes summing probability over paths with the same string from choosing
one best path. Determinization can preserve string weights, but may expand the
state space substantially. The construction below is a specialized finite,
acyclic consistency automaton; no new weighted-determinization theorem is claimed.
The Voynich scope limits and source/search review in
[KEY-SOURCE-DIAG-001](../experiments/KEY-SOURCE-DIAG-001.md) also apply.

Simply retaining the best text under each candidate key is insufficient. A text
that is second-best under several keys can have more *combined* probability than
any individual key's best text. Likewise, selecting a different convenient key
for every letter or record changes the generative model. We need to sum over one
global key while keeping all its repeated assignments consistent.

## Fixed finite-bank model

Let distinct deterministic keys be `K_1...K_M`, with normalized weights `w_i`,
all chosen/frozen from fitting data. Each key maps every source letter to one
nonempty observed unit. Let `Q(x)` include the normalized geometric stopping
law, and reset the source at each record boundary. For R observed records:

```
P(x_1...x_R, y_1...y_R)
  = product_r Q(x_r) * sum_i [w_i * product_r 1{C_i(x_r)=y_r}].

P(y_1...y_R) = sum_i w_i * product_r Z_i(y_r),
Z_i(y) = sum_{x: C_i(x)=y} Q(x).
```

The same key index must survive every record. The MAP *text tuple* maximizes
the first expression, summing compatible key weights. It differs from choosing
the MAP key-and-text path, which replaces the key sum with a maximum.

For a bank specified independently of fitting data, ordinary posterior weights
are proportional to `prior_i * product_fit Z_i(y)`. If search selected the bank
from those data, that same formula is a restricted, data-dependent approximation;
do not claim the exact posterior over the full key family. Once normalized and
frozen it still specifies a proper conditional predictive mixture. Reporting
its training evidence as if its selected bank were a free prior is invalid.
Bank duplicates must be removed or given explicitly justified prior mass; search
visitation frequency is not automatically posterior probability.

## Deterministic consistency states

For a finite-state source, use state `(record_index, source_state, offsets)`.
`offsets[i]` is the number of observed glyphs consumed by key i under the
*same plaintext prefix*, or -1 if it has become incompatible with any current
or earlier record. A key never revives. This vector is determined by the text,
so there is one path for each text tuple, not one path per candidate key.

On source letter a, check each live key's literal unit at its current observed
offset, advance that offset on a match, and kill that key otherwise. Delete a
transition if no key survives. Multiply by `(1-rho) * q(a|source_state)` and
advance the source state. Keys that reached the end cannot consume another
letter; they die on that transition.

A record boundary may be taken whenever some live keys have consumed exactly
the whole record. Kill all keys not at its end, reset surviving offsets and the
source state for the next record, and multiply by rho. Apply no key weight yet.
At the final record's stop, multiply by rho times the sum of weights of keys
that reached the end. Thus key mass is counted exactly once and shared across
all records. Empty records and overlapping units follow the same rule.

Within a record, order states lexicographically by `(dead_key_count,
sum_live_offsets)`: a letter either kills more keys or strictly increases the
second component. Across boundaries, record_index increases. The graph is
acyclic. Identical full states can merge even when reached by different text
histories because the source's retained state is sufficient and future key
compatibility is identical. Forward log-sums give evidence; Viterbi scores give
the best text tuple, with compatible key weights added only at terminal stops.

This is exact enumeration with state merging, not a beam. Float arithmetic is
not exact real arithmetic. Worst-case size can be exponential in bank size;
an explicit node cap must stop rather than silently prune or fall back to a
best-key answer. Neural hidden states cannot be collapsed to a finite suffix
state; this construction does not justify applying that merge to the recurrent
source. The implementation uses the finite-state statistical-source interface.

## Required checks before use

- Enumerate all text tuples and key assignments on tiny artificial examples;
  compare evidence, MAP text score and re-encoding under a globally shared key.
- Independently compare evidence with `sum_i w_i * product_r Z_i(y_r)`.
- Include an example where summing keys changes the winning text, and one where
  allowing separate keys per record would wrongly permit an impossible tuple.
- Check a one-key bank reduces to the existing exact decoder; reject duplicate
  keys, invalid weights, empty units and cap overruns; retain unsupported cases.
- Freeze any empirical bank-construction policy separately. Candidate banks,
  weights, alphabets and sources must not be chosen from gold or transfer accuracy.

A finite bank can still miss the right key or all supporting keys. This method
manages uncertainty *within* its declared bank; search coverage, historical
model adequacy and semantic truth remain separate problems.

## Implemented representation and validation

`src/voynich/shared_key_mixture.py` groups equal live offsets into `(offset,
key_bitmask)` pairs. Intersecting these masks with the keys that assign a given
unit is exactly the same transition as updating every key separately. No key
is removed for having a small positive weight. Zero-prior keys are inactive.
The acyclic rank counts live bits and sums offsets weighted by their bit counts.
This representation saves memory when keys share behavior; it does not remove
the exponential worst case.

Twenty-five artificial tests cover the checks above, including 48 randomly
generated tiny panels compared with complete enumeration. A 128-key example
with identical behavior on the observed records occupies the same number of
states as one key, with one offset group. This is a representation check, not
a performance claim for ambiguous empirical banks. Very small positive weights
remain in the log domain; enormous common log-weight offsets are normalized
after subtracting their maximum, and unrepresentable relative ranges fail.

Bank construction and full-size performance still require separate bounded
experiments. No unknown key, Latin passage or Voynich text has been recovered
by this module at this checkpoint.
