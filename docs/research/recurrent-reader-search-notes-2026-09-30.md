# Next reader: distinguish search failure from a wrong source preference

2026-09-30. Algorithm development with artificial inputs only. No fresh cipher
panel, reserved-author plaintext, historical reading or source-model selection
is performed by this work. Neural source fitting continues unchanged under
[LATIN-SOURCE-MODEL-001](../experiments/LATIN-SOURCE-MODEL-001.md).

## Reviewed sources and limits

[Huang, Zhao and Ma2017](https://aclanthology.org/D17-1227.pdf), section3,
uses nonincreasing prefix probabilities to justify stopping after finding a
sufficiently good complete candidate. Its guarantee is explicitly conditional
on beam pruning. Section4 explains why length normalization/rewards alter that
argument. Our observed-glyph offset, deterministic variable-length emission
constraint and fixed geometric stop law differ from neural translation.
[Kambhatla et al.2018](https://aclanthology.org/D18-1102.pdf) is prior neural
cipher-source/key-search precedent; this implementation instead assumes a
supplied key and tests reading/search competence. It cannot discover a key or
support a Voynich language claim by itself.

## Derived bound for the supplied-key problem

Let u(c) be one nonempty glyph string per plaintext letter. Several letters may
share a unit, and a unit may be a concatenation of others. For a candidate
plaintext prefix h ending at observed offset i, use

S(h) = sum over letters t of [log p(h_t | h_<t) + log(1-rho)].

A complete plaintext ending at the final observed offset has score
S(h)+log(rho). No source-length oracle or length normalization is supplied.
Every additional letter contributes at most log(1-rho), since its conditional
probability is at most1. A channel-only backward calculation supplies m(i),
the minimum number of letters needed to emit the remaining observed suffix.
An impossible suffix is removed exactly. For every viable partial candidate,

U(h) = S(h) + m(i)*log(1-rho) + log(rho)

is an upper bound on any completed descendant's score. This derivation assumes
normalized causal probabilities, deterministic nonempty units and the fixed
geometric stop law. No semantic or historical claim follows from it.

The prototype retains at most B prefixes at each observed offset, preserving
their distinct neural hidden/cell states. It records the maximum U(h) among
**all** discarded prefixes. Every missed complete path descends from such a
prefix. Thus max(returned score, recorded discarded upper bound) bounds the
true best score. If the returned score meets/exceeds that bound, it is a global
MAP solution under the specified exact-arithmetic model. If the inequality
fails, report no certificate; a bigger beam or unchanged answers do not prove
optimality. This bound may be very loose in realistic long passages. Do not
promise that it will certify useful manuscript-scale searches.

This is a standard monotone branch-bound argument, not a novelty claim.
Float32 device computations can vary slightly with batching; the prototype's
boolean indicates the mathematical bound on its computed scores, not an
interval-arithmetic numerical proof. Empirical use must replay returned paths,
record tolerances/margins and avoid calling a near-tie rigorously certified.
No exact marginal likelihood or posterior calibration is computed by the beam.

## Efficient full-history handling

The neural adapter batches only surviving prefixes. Each stores the state before
its pending final letter; that letter is consumed once to obtain next-letter
probabilities and its own new state. An initial BOS establishes the empty
history. Scores include emitted letters once and stop once. Completed paths
need no unnecessary final recurrent update. Hypotheses are never merged because
they share an observed position, recent suffix, or plausible translation.
Expansion caps fail explicitly; unsupported inputs have a distinct empty-support
result, rather than an invented plaintext.

The compact statistical source gets a separate lazy adapter with integer state
IDs partitioned by history depth. A leading'a'(IDzero) does not make'a', 'aa'
and the empty history collide. Probability rows and transitions are bounded
caches; probabilities use the unchanged recursive smoothing rule. The existing
exact finite-state decoder runs against this adapter without changing its
inference recurrence or silently relaxing its state cap.

## Artificial checks and next empirical gate

Fifteen neural-beam checks include complete enumeration for every binary glyph
string through length5 under three beam widths and four emission dictionaries,
a full-history toy source, deliberate greedy-search failure, empty/impossible
inputs, normalized probability rejection and hard expansion caps. Every issued
bound claim agrees with exhaustive enumeration in these cases. A separate
random double-precision LSTM checks full-forward candidate scores against
incremental opaque-state search. No trained source checkpoint is used.

Fourteen lazy-statistical checks compare all short-history probabilities and
transitions with literal string-context models at five orders/two cutoffs,
including tiny cache evictions and leadingzeroIDs. Three dictionary families
compare exact marginal/MAP scores, paths and node counts with the old decoder.
These tests validate implementation on artificial cases, not real reading
competence. Before empirical use, freeze selected sources, a fresh known-key
panel, beam widths, resource limits, scoring tolerances, error thresholds and
all output predictions before evaluating their accuracy. Report every source
and seed, not whichever yields the most attractive text.
