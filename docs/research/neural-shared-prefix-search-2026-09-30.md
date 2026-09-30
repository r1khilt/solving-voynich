# Neural proposals with one key shared across records

2026-09-30. Prospective algorithm derivation, written while CONFIRM-002's
unchanged prediction audit runs. No trained-source search, new candidates,
causal intervention or empirical speed measurement in this block.

## The concrete limitation

[NEURAL-002](../experiments/KEY-BANK-NEURAL-002-results.md) improves old-panel
ranking, but the exact true tuple is absent from the statistical proposal union
in four of eight cases despite positive bank support. Rescoring that union
cannot select an absent tuple. This does not prove an edit-distance floor;
another available tuple could still have fewer errors. A neural proposal search
must retain literal channel constraints and the same globally shared key.
It must not select a convenient key independently for every letter or record.

This proposal addresses reading *within an already fitted bank*. It cannot
restore a missing dictionary or solve unknown-key fitting by itself. The fresh
test's answer evaluation will determine whether bank coverage is the more
urgent bottleneck. Do not change that test in response to this proposal.

## Source review and differences

[Nuhn et al. 2013](https://aclanthology.org/P13-1154.pdf), abstract and the
existing project's qualification review, provide key-assignment beam-search
precedent for substitution decipherment. That supplied-symbol setting does
not establish recovery of our ambiguous variable-length emission family.
The prior Voynich scope review remains
[the fresh qualification review](blind-channel-fresh-qualification-review.md).

[Kambhatla et al. 2018](https://aclanthology.org/D18-1102.pdf), sections 3.1–3.3,
use neural predictions to fill unsolved positions and score partial dictionary
hypotheses. Their rest-cost construction is an estimate, not the normalized
shared-key marginal bound derived below. It motivates integrating a neural
source into search; it does not certify our candidate coverage or applicability
to Voynich.

[Huang et al. 2017](https://aclanthology.org/D17-1227.pdf), section 3, justify
stopping because continuation probabilities cannot increase a prefix's score.
Their guarantee is conditional on earlier beam pruning. Recording bounds for
every discarded branch is additionally necessary for an unrestricted search
claim here. Their section 4 also explains why changing the length objective
changes the argument. Keep our original geometric law fixed.

The following formulas are our elementary derivation under declared assumptions,
not a new theorem attributed to these papers or a novelty claim.

## Model and grouped compatibility

There are R observed records, a deterministic nonempty unit per source letter
in each distinct bank key, and frozen normalized positive weights w_k. The
normalized causal neural source resets to BOS at each record boundary.
For a complete plaintext tuple x, its unnormalized joint text/ciphertext score is

```text
S(x) = sum of causal letter log probabilities
       + total_letters(x) * log(1-rho) + R * log(rho)
       + log(sum_{k compatible with every record} w_k).
```

No true plaintext length, unit boundaries, new fitted weights or length reward
is supplied. The objective sums key mass for one text tuple; it does not sum
distinct text tuples. It is not exact evidence or a calibrated historical
posterior. Bank construction remains a data-dependent approximation.

A search hypothesis retains completed record strings, the exact current
record prefix, its full recurrent state, and `(observed_offset, key_bitmask)`
groups. Precompute literal matching key masks for each record/offset/letter/end
offset. Extension intersects the old masks with the matching masks, advances
offsets and combines equal end offsets. Dead keys never return. This is the
same consistency mechanism as the existing finite-state mixture decoder,
without merging different neural histories by suffix or observed offset.

When some keys reach a record's end while others have not, create **both**
legal branches: a boundary keeps only the finished keys, while a letter
extension keeps only keys that can emit another matching unit. A boundary
resets the neural source and starts the next record with those surviving keys.
The final boundary scores the sum of compatible key weights exactly once.
Empty observed records need the same explicit boundary rule.

One neural transition for an identical current record prefix can serve many
keys and many completed histories. Cache sharing is not permission to add
their probabilities or merge their text tuples. Floating computations can
also vary with batch shape; empirical paths need independent score replay.

## Two valid prefix upper bounds

Let L(h) contain all emitted-letter log probabilities and continuation factors
so far, with no stopping factors. Let A(h) be the still compatible key set.
Because future causal probabilities and continuation factors are at most one,

```text
U0(h) = L(h) + R*log(rho) + log(sum_{k in A(h)} w_k)
```

bounds every complete descendant. The unused stopping factors are included
upfront, once for every registered record, even when a boundary has not yet
been taken. Thus neither a boundary nor a terminal score double-counts them.

A stronger channel-only bound is possible. For each surviving key k, calculate
the minimum number m_k(h) of additional source letters that can emit the current
observed suffix plus all future records, independently of source probabilities.
A backward minimum-cost recurrence uses only legal literal units, with cost one
per letter. An impossible suffix gives infinity and removes that key exactly.
Then

```text
U1(h) = L(h) + R*log(rho)
        + logsumexp_{k in A(h)} [log(w_k) + m_k(h)*log(1-rho)].
```

For each descendant and each key compatible with that descendant, its remaining
length is at least m_k and its future source probability is at most one.
Bound each key contribution separately and sum them, giving U1. Therefore
U1 <= U0. This bound includes no gold lengths or source heuristic. It may still
be much too loose to certify realistic searches. Computing it over large key
banks may itself be expensive; measure that overhead before promising speed.

With nonempty units, total emitted letters are bounded by total observed glyphs.
A finite search can proceed by letter depth, with zero-letter boundary closure
performed before pruning. Record the largest valid upper bound of every
discarded branch, including resource/candidate pruning. If a returned complete
score exceeds every surviving and discarded bound by a declared numerical
margin, the branch argument establishes optimality for the model in exact
arithmetic. Ordinary floats are not an interval-arithmetic certificate. A beam
width alone, repeatable output or a better reading does not establish optimality.

## An exact dominance rule at reset boundaries

Suppose hypotheses A and B have the same record index and **identical full
current record prefix**. Source resetting makes every future letter probability
the same for an identical continuation. Suppose their compatible-key offsets
are identical on B's keys, A contains all of B's keys, and L(A) >= L(B).
For every continuation, B's final compatible-key set is a subset of A's and
its source-prefix factor is no larger. Therefore its final tuple score is no
larger. For finding one maximum-scoring tuple, B can be removed in favor of A.

Equality needs a declared deterministic tie policy. If key sets differ,
score superiority alone is insufficient. A higher-scoring prefix with fewer
keys can lose later. If BOS does not reset, identical current-record strings
do not imply identical source histories and this rule fails. If the task is
evidence, posterior samples or multiple distinct texts, discarding B loses
probability mass and this MAP dominance rule is insufficient. Preserve a
representative hypothesis and its bounds if later beam pruning removes it.

## Qualification before any empirical run

The implementation should first pass exhaustive tiny-text and key enumeration:
overlapping units, duplicate unit rows, shared keys across records, mixed-key
impossible tuples, empty/impossible records, tiny positive weights, ties,
deliberate greedy failures and all issued discarded-branch bounds. Construct
counterexamples to unsafe source-state merging and score-only dominance.
Compare cached neural transitions with an independent whole-sequence source;
verify BOS resets and the length/stopping law independently.

Only then register a finite artificial systems benchmark for state-sharing,
peak driver memory, host memory, batch-shape error and complete work counters.
No trained source or old gold enters this benchmark. A later empirical
development experiment must freeze both neural seeds, all original cases,
bank weights, beams and limits before one run, publish predictions before
accuracy evaluation, retain missing cases, and measure proposal coverage
separately from ranking. No best-seed selection or fresh-test retuning.

Mechanistic analysis is a distinct next branch: use matched context/state
interventions to determine why a source prefers a wrong *available* reading.
Search gains or a low-dimensional next-letter readout do not themselves reveal
a causal workspace or deciphering algorithm. No new empirical intervention
is claimed by this memo.
