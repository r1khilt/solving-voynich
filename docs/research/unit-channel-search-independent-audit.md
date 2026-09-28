# Independent audit of deterministic unit-channel search

Date: 2026-09-28. This reviews `src/voynich/unit_channel_search.py` and its
[implementation note](unit-channel-search.md). All tests use small artificial
sources and literal strings. No empirical corpus, pilot ciphertext, fitted
pilot model, transfer material or answers were opened or scored. The
[batched-kernel audit](batched-unit-channel-independent-audit.md) is separate;
its completion does not establish recovery by the search built on it.

## Independent objective and candidate checks

A key is a tuple containing one deterministic, nonempty unit per source letter.
It has one channel state; duplicate units are allowed and retain distinct source
contexts. The declared glyph alphabet, maximum unit length, source, stop law
and coding context are fixed inputs. For each record, the independent reference
uses a backward `Fraction` recurrence over matching whole units:

```
F(end, context) = rho
F(offset, context) = sum over matching u(letter):
    (1-rho) P(letter | context) F(offset+len(u(letter)), next_context)
```

Every record starts with the source start context. Multiplying record
probabilities, then taking the log of the exact numerator and denominator,
gives the reference corpus likelihood. This reference does not use the batched
kernel, its scaling, or the original finite-state inference engine.

Let `w(K) = ceil(log2 K)`, with `w(1)=0`, and let `A` be the source alphabet
size. The independent model-code length for units `u` is

```
w(source_count) + w(max_states)
  + A * (w(max_alternatives) + w(max_emission_length))
  + w(glyph_alphabet_size) * sum(len(unit) for unit in u).
```

Initial-state weights, destination states and row-weight compositions each
have one possible value and contribute zero bits. The other field widths are
fixed under the supplied context, even for unused source rows. This confirms
the production shortcut of an actually encoded all-singleton baseline plus
the literal-glyph length difference. Length headers have fixed width; the
literal strings still cost bits. An independent test constructs and compares
every bit of all 144 two-row encodings over three glyphs with length bound two,
including Unicode and trailing U+0000, and non-power-of-two coding bounds.

The pool contains every declared-alphabet string of length one through the
bound, including strings and glyphs absent from observations. The cap rejects
an oversized pool instead of silently omitting candidates. With `U` distinct
units, a key's neighborhood has

```
A*(U-1) + number of unequal unordered row pairs
```

distinct keys: all single-row replacements plus all unequal-row swaps. A swap
changes two rows, a replacement changes one, and distinct unequal swaps change
different row pairs. Tests independently enumerate the set for all 36 two-row
keys over the six-unit pool `(x,y,xx,xy,yx,yy)`. Duplicate pool entries passed to
the public helper do not repeat neighbors.

The search-trace checks independently replay every scored candidate for both
source orders. Additional tests inject all 36 artificial initial keys to expose
the whole tiny family to the evaluator, then replay each supported start's
complete neighborhood. This injection is solely a test fixture, not an
empirical or production warm start. Supported trace scores and returned scores
agree with the rational likelihood and independent code count within
`6e-13` absolute tolerance, with zero relative tolerance. Unsupported candidates
have `score=null` rather than nonfinite JSON values. The returned model has the
lowest completed score encountered, including candidates rejected as moves.

## Initialization, budgets and certificates

Every ordinary initialization retains each declared singleton, including absent
glyphs. The first restart ranks source rows by fixed **start** probability and
glyphs by observed counts; it is not a stationary-frequency estimate. Later
restarts use recorded seeds, singleton coverage, and the complete pool for extra
rows. Repeated count-bounded runs with the same seed reproduce units, scores
and trace. A binding wall-time budget can change the number of completed
batches across runs or machines; a seed alone does not fix that outcome.

Singleton coverage is an initializer restriction, not a restriction on later
proposals. It requires `A >= |G|`; rejection when this fails does not show the
larger deterministic-unit family is impossible. Source zeros can defeat every
singleton start even when a supported variable-length key exists. For example,
with `P(a)=1`, `P(b)=0`, glyphs `(x,y)` and record `xy`, every covering singleton
assignment is unsupported, but units `(xy,x)` give probability `1/4` at
`rho=1/2`. The implementation correctly reports
`no_supported_initialization`, without a family-wide impossibility claim.

The timeout tests use a scripted clock and real artificial scoring, without
sleeping. They verify three boundaries:

- A generated batch interrupted before scoring is absent from completed
  neighbor counts and cannot displace the initial key.
- Every candidate in a finished batch remains eligible after the deadline,
  including a genuine improvement in a partially evaluated neighborhood.
- A final finished batch can complete the whole neighborhood even if it crosses
  the deadline. A complete no-gain sweep can certify its parent; a partial sweep
  cannot.

Timed returns still execute the original-engine score and literal-code replay
on their selected channel. Batch/kernel calls, preprocessing, trace construction
and the final replay remain cooperative atomic work and can overrun the nominal
time limit. The kernel's bounded rolling buffers do not bound total search
memory: the complete retained trace grows with evaluated keys, their unit tuples
and score fields.

Acceptance and retention have different tolerances. The parent moves only for
an improvement exceeding the configured threshold, while any strictly better
completed candidate can become the global best encountered. A completed
no-sufficient-gain sweep certifies its parent within that threshold. Its
certificate does not transfer to a different returned key. The independent
large-tolerance fixture verifies that the retained better neighbor has
`best_is_certified_local_optimum=false` even though `local_optima=1`. These
certificates concern the declared neighborhood and floating-point tolerance,
not symbolic exact arithmetic.

## A complete local certificate is not a global result

Use source starts `(a,b)=(3/4,1/4)`, transitions after `a` equal to `(1/8,7/8)`
and after `b` equal to `(5/8,3/8)`, `rho=1/4`, glyphs `(x,y)`, maximum unit length
two, and observation `xxxx`. Units `(xx,xx)` give marginal `9/64`. This key is
strictly better in total code-plus-data bits than every distinct one-row
replacement and unequal-row swap.

Changing both rows to `(x,x)` gives marginal `81/1024`. It loses
`log2(16/9) = 0.830075...` data bits but saves two literal-model bits, so the
total score improves by `1.169925...` bits. The two-row coordinated change lies
outside the checked neighborhood. The independent test verifies each local
comparison and this better witness using rational probabilities. A local
certificate therefore cannot establish a globally best dictionary, correct
plaintext, identifiability, or failure of the whole model family.

## Validation and conclusion

`tests/test_unit_channel_search_independent.py` has **15 passing tests**,
covering the objective, literal bits, candidate completeness, normalization,
source orders and zeros, absent glyphs, deterministic initialization budgets,
unsupported values, trace accounting, deadline boundaries, certificate
applicability and the non-global counterexample. The implementation note's
mathematical and scope claims agree with this review. No production change was
required and no unresolved correctness blocker was found in this scope.

```
PYTHONPATH=.:src .venv/bin/python -m pytest -q tests/test_unit_channel_search.py tests/test_unit_channel_search_independent.py
```

Result: **38 passed** (23 author tests and 15 independent tests). Ruff passes
for the independent file. The tests qualify this restricted implementation;
they do not provide empirical speed, recovery, Bayesian evidence or
decipherment results. Source mismatch, local optima, initialization support,
time coverage, floating-point error and retained-trace size remain explicit
limits for subsequent bounded experiments.
