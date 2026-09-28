# Fixed-support finite-state emission fitting: independent audit

2026-09-27. This is an implementation audit on hand-specified tiny models, not
a recovery experiment. It extends the [exact-channel mathematical checks](finite-state-channel-math-audit.md)
and the methods review in the [blind-channel design](blind-channel-recovery-design.md).
No corpus, reference plaintext, supplied segmentation, benchmark generation,
or sealed answers are used.

## What one EM update estimates

Hold the source probabilities, source order, initial-state distribution,
stopping probability, states, emission strings and destinations fixed. For
independent observed records `Y`, let `N_(s,a,j)` be the sum over records of
the posterior expected number of uses of row alternative `(s,a,j)`. These
expectations integrate source characters, channel paths and segmentation.
The terms of the expected complete-data log likelihood that depend on the
emission probabilities are

```text
Q(q' | q) = constant + sum_(s,a,j) N_(s,a,j) log q'_(s,a,j).
```

Maximizing separately over each normalized probability row gives

```text
q'_(s,a,j) = N_(s,a,j) / sum_k N_(s,a,k).
```

If the denominator is zero, the row has no effect on `Q`; retaining its old
weights is a valid maximizer. Positive-occupancy rows set zero-count
alternatives to zero. An alternative initialized at zero has zero posterior
responsibility and cannot be revived by this update. Empty records contribute
`log(rho)` each and no emission counts. Duplicate emission alternatives retain
their separate latent identities and responsibilities.

With exact posterior expectations and this unregularized M step, the
observed-data likelihood cannot decrease in exact arithmetic. Numerical
implementation checks require explicit tolerance. This statement concerns the
same fixed structure, fixed source, and fixed stop law. Smoothing, probability
quantization, deletion of alternatives, and structural moves are separate
operations and do not inherit this monotonicity claim. EM may stop at a local
optimum or a boundary fixed point; it does not establish global optimality,
channel identifiability, recovered plaintext, or decipherment.

## Hand-calculated updates

Take a one-letter source, one channel state, and `rho=1/2`. Let the channel
emit `x` or `xx`, initially each with probability `1/2`. For the observed
record `xx`, the single-event path has probability `1/8`; the two-event path
has probability `1/32`. Thus `p(xx)=5/32`, posterior event counts are
`N_x=2/5` and `N_xx=4/5`, and the updated row is `(1/3,2/3)`. Direct path
summation at the new row gives `p(xx)=13/72`, an increase. Choosing a single
best segmentation would give a different update.

For two independent records `x` and `xx`, the first adds counts `(1,0)`.
Aggregate counts are therefore `(7/5,4/5)`, giving row `(7/11,4/11)` and
joint record probability `959/42592`, up from `5/256`. Averaging normalized
per-record rows would incorrectly give `(2/3,1/3)`; counts must be summed
before normalizing. Adding empty records changes likelihood by their stop
factors but leaves this emission update unchanged.

For a second example, let source probabilities be `p(a)=3/4,p(b)=1/4`, with
single-glyph rows `q(x,y|a)=(1/4,3/4)` and `q(x,y|b)=(3/4,1/4)`. An `x`
record assigns posterior source probabilities `(1/2,1/2)`; a `y` record
assigns `(9/10,1/10)`. For records `(x,x,y)`, separate row normalization
must yield `q'(x,y|a)=(10/19,9/19)` and
`q'(x,y|b)=(10/11,1/11)`. The fixed source probabilities are not refitted.

## Independent implementation checks

`tests/test_finite_state_channel_fit_independent.py` checks these rational
updates and input preservation. Its separate recursive enumerator sums every
compatible complete path using `Fraction` arithmetic; it calls neither the
production forward/backward routines nor any decoder. It verifies observed
likelihood before and after six successive EM steps on both a one-state
source-order-zero fixture and a two-state source-order-one fixture. Six
single-step fits are compared with one six-step fit.

Additional cases check unreachable states, zero-probability source rows,
zero-weight alternatives, duplicate latent alternatives, empty records and
rejection of any impossible record instead of silently excluding it. These
checks validate bounded implementation behavior and the stated formulas.
They neither discover units nor test recovery on unknown channels.

Hand-written binary descriptions also check a 12-bit one-state model and
5-bit two-state models, including positive emission-weight compositions and
weak initial-state compositions that permit zero probabilities. Incompatible
shared contexts and insufficient declared support bounds are rejected. A
concrete rounding counterexample takes the `(1/3,2/3)` EM row to `(1/2,1/2)`
on denominator two, reducing `p(xx)` from `13/72` to `5/32`; quantization must
therefore be scored separately. A scripted clock checks interruption both
during candidate scoring and during the next E step, confirming return of
the last fully scored model without real sleeps.

Validation: `.venv/bin/pytest -q tests/test_finite_state_channel_fit_independent.py
tests/test_finite_state_channel_fit.py` passed **31 tests in 0.05 seconds**,
including **22 independent checks** in the new audit file. Its Ruff check also
passed. No production-code discrepancy was found, and no production file was
edited by this audit.
