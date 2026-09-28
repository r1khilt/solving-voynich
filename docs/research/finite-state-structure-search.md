# Bounded search over unknown channel units and mappings

2026-09-27. Implementation development for the
[blind-channel design](blind-channel-recovery-design.md), using the independently
checked normalized engine and [fixed-support fitting/code](finite-state-fitting.md).
No corpus ciphertext, benchmark generator, target plaintext, historical answer,
or final recovery score was used to implement or check this module.

`src/voynich/finite_state_channel_search.py` implements a concrete seeded local
search. Its inputs are ciphertext records, a fixed `SourceModel`, the shared
`CodingContext`, a source index, and finite search settings. It performs no
filesystem access. The source alphabet/model, glyph inventory, source-candidate
list, geometric stop law, and coding bounds remain declared assistance. The
default interface has no target, oracle mapping, known unit inventory, or
known state trajectory input. An oracle with supplied support can use the
separate `fit_em` function and must be reported as an assisted arm.

The primary-source review before implementation is the table in the linked
blind-channel design, especially Chiang et al. (2010) on normalized finite-state
estimation and supplied inventories; Ravi and Knight (2011) on ciphertext-only
source/channel training and its constraints; Ristad and Yianilos (1997) on
path summation and proper termination; and Grünwald and Roos (2019) on explicit
two-part codes. The new proposal mechanism is ordinary finite randomized local
search, not their Bayesian inference algorithms, a learned proposal model,
Metropolis–Hastings, or a new identifiability theorem. The existing review also
covers constrained Voynich applications and their limitations. This engineering
step withdraws supplied emission strings and mappings inside a small declared
family; it does not withdraw all structural assumptions.

## Search support and initialization

The unit pool contains **every singleton glyph in the shared declared alphabet**,
including glyphs not present in fitting records, followed by observed multi-glyph
substrings up to the coding length bound. Multi-glyph substrings are counted with
overlap, never across record boundaries. They are ranked by descending count,
shorter length, then lexicographic declared glyph indices; a fixed `max_units`
caps the retained pool. All declared singletons must fit under that cap. No
unobserved multi-glyph strings are proposed. Transfer may introduce an absent
singleton; the method cannot assume the correct unseen multi-glyph codeword.
The pool is a search restriction, **not an uncharged model dictionary**: literal
symbols and lengths are still included in each channel's actual binary code.

Each initialization has a requested fixed state count. States start with
quantized uniform initial weights. At every state, distribute all declared
singletons across the source-letter rows with self-loop destinations, and fill
otherwise empty rows with a singleton. Row weights are normalized and quantized.
Consequently every source row exists and every glyph is available at every state.
For strictly positive source conditionals this supplies a positive path for any
record over the declared glyph alphabet. The initializer requires

```text
|declared glyph alphabet| <= |source alphabet| * min(denominator, max_alternatives).
```

This is an **initializer capacity restriction**, not a proof that channels outside
it cannot explain the records. Sources with zero transitions are permitted, but
the coverage argument does not apply. Every initialized channel must therefore
pass exact observed-record support checks before entering search. A finite number
of unsupported attempts is recorded; no supported initialization yields an
explicit result with no channel/score, not an invented fallback or a successful
recovery. Empty records, including a collection of only empty records, are valid
because the declared alphabet supplies initialization and the stop law assigns
their observed probability. They supply no mapping evidence.

When `frequency_initialization=True`, the first attempt of the first **one-state**
restart ranks source letters against fitting-ciphertext glyph frequencies, using
declared-order ties. Source order zero uses its supplied letter probabilities;
order one uses the first 256 source-position marginals, weighted by geometric
survival and normalized. The finite truncation is only an initialization heuristic
and is never substituted for the exact source model in scoring. Equal alphabet
sizes give an initial bijection. Extra source rows are filled by the seeded
singleton procedure. Other attempts/restarts shuffle assignments. This ordinary
frequency heuristic may help a simple substitution and may mislead a verbose or
stateful cipher; it does not supply a true mapping. Its settings were fixed before
any corpus-based structural recovery outcome.

## Finite structural proposals and selection

Choose uniformly among eight move types. Record an explicit rejection when a
selected move is inapplicable or produces the unchanged channel:

1. Swap two source-letter rows within a chosen state.
2. Swap two source-letter assignments across every state.
3. Replace one emission's literal string with a different pool unit.
4. Add a new destination/unit pair, within the row/denominator bounds.
5. Remove an emission, retaining at least one alternative in its row.
6. Retarget an emission to a different state.
7. Transfer positive integer grid counts between alternatives in a row.
8. Transfer integer grid counts between initial-state weights.

New alternatives receive positive mass and the previous row weights are scaled
to maintain normalization; removal renormalizes the remaining row. Quantization
merges any duplicate destination/unit pair and enforces the shared grid. There
is no implicit escape channel, no epsilon emission, and no per-record replacement
rule. Moves can destroy observed parse support, in which case the proposal is
rejected after exact scoring detects zero probability. The whole proposal is
rejected, rather than discarding the troublesome record or path.

Score a raw quantized proposal first. Optionally run at most `em_iterations`
ordinary fixed-support EM updates, quantize again, and **rescore those actual
weights**. Keep the better of raw and refined complete scores; quantization and
the model-code change need not favor the EM update. Skip EM when every row has
one alternative because it then has no trainable row weights. The initial law,
source, state count, stop law, and strings stay fixed during EM. All model changes
outside EM are explicit proposals above.

The objective is exactly

```text
len(encode_channel(candidate, shared_context, source_index))
  - sum_records log2 forward_probability(record | source, candidate).
```

The forward probability sums all compatible plaintext/state/segmentation paths,
including normalized nonmatching alternatives and proper stopping through their
probability costs. Joint Viterbi output is never used for selection. A candidate
replaces its restart incumbent only if its total-bit improvement exceeds the
fixed numeric tolerance. Across restarts, retain the smallest score among **all
finished candidates**, including improvements below that local tolerance.
Restart rounds interleave the requested state counts; by default these are one
and two. State birth/death is not a within-restart move, and budget exhaustion
may prevent a later requested restart from starting.

This is a greedy heuristic with random restarts. A better found score does not
prove global optimality, a posterior sample, a Bayes factor, or correct plaintext.
It can get stuck behind temporarily unsupported or higher-cost intermediate
structures. It cannot discover units outside its finite pool, deletions, nulls,
copying, transposition, more states than requested, or a new source language.

## Budgets, deadline behavior, and audit trace

`SearchConfig` explicitly specifies the seed, state counts, restart count,
proposal count per restart, initialization attempts, EM count/tolerance,
unit-pool limit, frequency initialization, local acceptance tolerance, and global
wall-time budget. Proposal attempts include inapplicable moves. Exact fixed-count
replay is deterministic when the wall limit does not bind; wall-limited runs
can stop at different candidates on different machines.

Deadline checks occur between exact per-record scores. An interrupted record
sequence contributes no candidate score. A raw proposal scored completely before
an interrupted refinement remains eligible, as does the best previous restart.
A call that finishes the last record is a completed score even if that atomic
call runs past the deadline. No completed supported candidate yields `channel=None`
and `score=None`. The returned `stop_reason` distinguishes wall expiry, count-budget
completion, and exhaustion of unsupported initializations.

The wall budget is cooperative, **not a hard process limit**. Substring counting,
source-frequency initialization, quantization/serialization, one exact record
evaluation, and `fit_em`'s initial whole-corpus likelihood pass are not preempted.
Each may overrun a deadline; callers needing a hard system limit require an
external worker/watchdog and must preserve prior saved checkpoints. The retained
unit pool is capped after counting observed substrings, so that cap is not a
bound on temporary counter memory. No asynchronous/background work survives the
call. The search returns its best complete model; it does not itself write a
checkpoint to disk.

Every initialization and proposal has a trace event, including inapplicable
proposals, zero-support rejections, and interrupted evaluations. It records the
restart seed, state count, move details, raw/refined serialized candidate, complete
score or failure status, number of records scored, EM trace when invoked,
acceptance/rejection reason, and running best score. Completed stages retain
actual model/data/total bits and log likelihood. Partial stages have no score;
no JSON infinities are needed. `SearchResult.to_dict()` includes the chosen
channel, score, configuration, unit pool, proposal/evaluation counts, elapsed
time, and complete trace. Independent audit can reconstruct every scored
candidate from these records. Trace storage grows with the number and size of
evaluated channels; callers should save it losslessly rather than deleting
rejected candidates.

## Artificial-fixture checks and measured cost

The owned tests cover declared singleton retention, overlapping substring ties
and record boundaries, an analytically known geometric-stop score, an actual
unit-changing improvement, normalization/grid/code invariants for every completed
candidate, reproducible one/two-state restart budgets, unsupported zero-transition
sources, frequency initialization, skipped degenerate EM, and rollback from
partial candidate/refinement scores. The
[separate independent audit](finite-state-search-audit.md) reconstructs path sums
and literal code-field widths without calling the production scoring helpers.
These checks establish implementation behavior, not empirical cipher recovery.

A single timing check on 2026-09-27 used Python 3.12.13 on the local Mac, no paid
compute, and **hand-specified artificial strings only**. Source alphabet was the
23 ASCII letters `a` through `w`; its start row was uniform and each order-one
row assigned .75 to the cyclic successor and .25/22 to every other letter.
There were four 224-glyph records (896 glyphs total), each a periodic glyph
alphabet cycle offset by its record index, and stop probability 1/225. These
are throughput fixtures, not generated cipher tasks with hidden targets.

| Hand-specified channel | Full corpus forward score | Full corpus posterior counts |
| --- | ---: | ---: |
| One state, 23 glyphs, one emission per row | 0.00331 s | 0.00969 s |
| One state, six glyphs, one singleton per row | 0.01432 s | 0.03907 s |
| One state, six glyphs, three alternatives per row | 0.07999 s | 0.22836 s |
| Two states, six glyphs, three alternatives per row | 0.08000 s | 0.22979 s |

In the three-alternative fixtures, source-letter row index i emitted glyph i,
glyph i+1, or the length-two string (i+2,i+3), modulo six, at equal weights;
alternative index j selected destination j modulo state count. One-alternative
23-glyph rows used their correspondingly indexed glyph; six-glyph rows used
index modulo six. A 20-proposal one-state search on the 23-glyph fixture, seed29,
denominator32, length2/alternatives3/state2 coding bounds, EM disabled, completed
11 scored candidate configurations in 0.04038 s. No objective or decoded text is
reported as a recovery result.

These one-run times are not worst-case estimates. The two-state construction
does not force independent dense state uncertainty; its near-equal timing is not
evidence that states are free. Longer emitted strings, more ambiguous mappings,
EM iterations, more source contexts, and trace serialization increase work.
The main bottleneck is repeated exact marginal/posterior inference, which scales
with reachable context/state combinations and competing emissions at each glyph
offset. Actual exposed-development throughput and complete oracle/negative
controls remain necessary before any sealed run budget or recovery claim.
