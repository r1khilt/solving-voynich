# Independent audit of the batched unit-channel likelihood

Date: 2026-09-27. Scope: `src/voynich/batched_unit_channel.py`, using artificial
fixtures only. No empirical corpus, pilot input, fitted model, answer or transfer
record was opened or scored for this audit. This is an implementation audit, not
a recovery experiment or evidence of decipherment.

## Model and independent calculation

A candidate assigns one nonempty literal unit to each source letter. The channel
has one deterministic state; different letters may emit the same unit and must
retain their different source contexts. Source probabilities and the geometric
stop probability `rho` are fixed. A record boundary resets the source context.
Each source step contributes `(1-rho) P(letter | context)`; exactly one final
factor `rho` is charged. Thus an empty record has probability `rho`, while an
empty collection of records has likelihood one.

The independent tests use a backward suffix recurrence with Python `Fraction`:

```
F(end, context) = rho
F(offset, context) = sum over matching whole units u(letter):
    (1-rho) P(letter | context) F(offset+len(u(letter)), next_context)
```

This calculation does not call the matrix kernel, its helpers, or the existing
finite-state inference engine. It enumerates every admissible segmentation
through the recurrence. Corpus likelihood is the product of independent record
probabilities. The fixtures use exactly representable rational probabilities.
Expected logs are computed from the resulting numerator and denominator.

## Scaling and underflow review

The production recurrence keeps mass for all pending glyph offsets in a ring.
It uses one common scale per candidate across **all** pending offsets and source
contexts. Copying and clearing the consumed slot before adding future mass
prevents stale mass from surviving ring reuse. Different candidates have
separate scales, and each record starts from fresh buffers.

Common scaling preserves ratios between paths that have consumed different
numbers of source characters. For an equiprobable two-letter source, `rho=1/2`
and observation `xxxx`, units `(x,xxx)` give probability `33/512`, while units
`(x,xx)` give `29/512`. The independent test checks both explicitly. Dividing
separate offsets by separate totals would not represent this model.

The risk guard examines positive current-context mass before multiplication by
the source and continuation probabilities. If a product could approach the
float64 underflow range, the entire candidate/record is recomputed using the
separate rolling log-domain recurrence. Checking only whether the final answer
is zero would be insufficient: a dominant context can survive while another
context needed by a later suffix disappears. The guard is conservative because
it uses the smallest positive source probability, including possibly irrelevant
transitions.

The long adversarial fixture starts equally in `a` and `b`. Letter `a` repeats
forever, `b` repeats with probability `1/2` or moves to `c` with probability
`1/2`, and `c` repeats forever. Units for `a` and `b` can both be `x`, but only
`c` emits `y`. For `x` repeated 1800 times followed by `y`, the only complete
source path is `b^1800 c`, despite the much larger competing `a` prefix. The
returned log likelihood matches the closed-form path expression and is finite.
Variants with two-glyph `b` emissions and with an incompatible `a` emission are
checked together and separately. A positive smallest-subnormal source
probability also yields the expected finite log likelihood through fallback.

The review found no path-pruning or state-merging discrepancy. "Exact" here
means the full finite path sum for the restricted model, evaluated numerically;
it does not mean exact real arithmetic. Both the fast calculation and fallback
use float64, and this audit is not a universal floating-point error proof.

## Checks and result

`tests/test_batched_unit_channel_independent.py` contains 10 passing tests:

- All 49 two-letter candidate maps from seven literal units against every binary
  string of length zero through five, for source orders zero and one: 6174
  candidate/record comparisons with exact rational expectations.
- Different pending lengths with the hand probabilities above; duplicate units
  whose distinct source contexts have different future support; exact source
  zeros and impossible observations.
- Candidate order, batch partition and record order independence, mixed record
  lengths, repeated empty records, and input immutability.
- Rare-context and positive-subnormal fallback controls, including a candidate
  that remains safe alongside candidates requiring fallback.
- Unicode units, embedded/trailing U+0000, whole-emission matching, and strict
  record boundaries. Explicit length masking correctly distinguishes NumPy
  Unicode padding from literal trailing U+0000.

Command:

```
PYTHONPATH=.:src .venv/bin/python -m pytest -q tests/test_batched_unit_channel.py tests/test_batched_unit_channel_independent.py
```

Result: **35 passed** (25 author tests and 10 independent tests). Ruff passes
for the independent test file. No production change was required by this audit.

## Limits

This kernel covers single-state deterministic nonempty units with order-zero or
order-one fixed sources. It does not fit parameters, search units, identify a
key, implement arbitrary finite-state stochastic channels, or establish search
optimality. Its rolling probability buffer is bounded by candidate count,
source alphabet size and maximum actual unit length; the exposed buffer-byte
diagnostic is not total process memory. The synchronous kernel has no internal
deadline: callers must bound batches and records and enforce any hard process
limit externally. Underflow fallback can materially change runtime on extreme
inputs. No empirical speed or recovery claim follows from these fixtures.
