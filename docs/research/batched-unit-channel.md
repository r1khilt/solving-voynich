# Batched deterministic unit-channel marginal

This is an isolated likelihood kernel for a one-state deterministic channel,
with one nonempty literal unit per source letter. Units may have different
lengths and may coincide across letters. The fixed source is order zero or one.
Every compatible plaintext and segmentation contributes to the marginal;
there is no beam, Viterbi substitution, unit discovery, or probability fitting.
“Exact” describes the path sum, with ordinary floating-point arithmetic.

`batch_log_likelihood(source, candidate_units, records, stop_probability,
max_emission_length=2, diagnostics=None)` lives in
`src/voynich/batched_unit_channel.py`. The diagnostics argument is keyword-only.
Candidate tuples follow `source.alphabet` order. The returned float64 array has
one corpus log likelihood per candidate. Records reset the source start context
and independently pay the stop probability. Empty records have probability
rho; an empty record collection has log likelihood zero; impossible records
produce negative infinity. Input values are copied and validated, including
Unicode unit lengths. There is no file or target-corpus access.

## Recurrence and scaling

Let A be the source alphabet size, L the maximum actual unit length in a batch,
and B its candidate count. Use source-context indices 0 through A−1 and an extra
start-only index A. The fixed matrix P has shape (A+1,A). Its last row is the
source start distribution; order-zero sources repeat that row everywhere.
Candidate b starts with mass one in the start context at glyph offset zero.

At observed offset t, multiply its context row by P and by (1−rho). For each
source letter a, add that mass to context a at t+|u[b,a]| only when u[b,a]
matches the observed substring. A rolling array with L+1 slots holds all
reachable pending offsets. Since units are nonempty, the current slot can be
cleared before depositing its outgoing mass. This keeps duplicate emission
units as distinct source-context paths. Empty current slots are harmless when
longer units have placed mass at future offsets.

After advancing an offset, divide **all pending offsets and contexts for each
candidate by the same total** and add its logarithm to that candidate's scale.
Using different scales for different future offsets would alter the relative
weights of different segmentations. Endpoint context mass plus accumulated
log scale and log(rho) gives the record likelihood. Candidate scales and
messages never mix across the batch.

## Numerical guard and resource behavior

Shared scaling prevents a long record's total probability from underflowing,
but does not alone protect a rare context while another context dominates. The
rare context can subsequently be the only one matching a suffix. Before a
matrix product, the implementation checks whether the smallest positive live
context times the smallest positive source factor times (1−rho) approaches
the smallest normal float64 value (with a factor-16 margin). A flagged
candidate/record pair is recomputed from its start by a rolling log-space
recurrence with log-add-exp. This applies even when the fast result might have
remained finite. Zero source factors stay impossible and are excluded from the
positive-factor minimum. The conservative guard can cause extra fallback work;
it does not discard small paths or wait for an observed negative infinity.

The fast rolling buffer uses (L+1)B(A+1) float64 values. Including source,
candidate strings, lengths, and temporary masks, auxiliary space is
O(B(L+1)(A+1)+A²+BAL), independent of record length. The scalar fallback also
uses a rolling buffer, O((L+1)(A+1)+A²), rather than the existing engine's full
lattice. Inputs and returned values are excluded from these bounds. The
diagnostic `rolling_buffer_bytes` is the theoretical fast ring size, not peak
resident memory, and may be reported even when no record needs allocation.

Diagnostics also report candidate/record counts, source order/alphabet size,
maximum actual unit length, and `log_domain_fallbacks` (candidate/record pairs).
The call has no internal deadline. Callers must bound batch and record sizes;
one batch or a numerically difficult fallback may overrun a cooperative search
deadline. There is no resource-triggered approximate result. Hard process
limits can terminate a call without a result.

## Artificial validation and timing, 2026-09-27

Own tests compare exhaustive rational path sums for every two-letter mapping
from four units and every binary string through length five, for source orders
zero and one, against both the kernel and original engine. Additional tests
cover long/empty/impossible records, duplicate units, skipped offsets,
independent records/candidates, zeros, Unicode and trailing U+0000, subnormal
source probabilities, and a rare-context path that alone matches a late
suffix. Independent audit tests use a different rational suffix recursion:
[audit](batched-unit-channel-independent-audit.md). All 35 combined tests pass.

Two bounded, non-empirical timing processes used Python 3.12.13, NumPy 2.5.3,
one BLAS/OpenMP thread, and `resource.RLIMIT_CPU=(30,30)` before NumPy import.
The hand-specified 23-letter source has uniform starts and transition .75 to
the cyclic next letter, .25/22 to every other letter. Glyphs are `012345`.
Each candidate has all six singleton units and 17 independent random two-glyph
units, shuffled across source rows (duplicates allowed). Records are direct
periodic glyph cycles, shifted by record index. They are not generated from
hidden plaintext or fitted to a real corpus. No empirical input or answer was
opened for these measurements.

| Synthetic call | Batch time | Original engine check | Max log-likelihood difference | Fallback pairs |
| --- | ---: | ---: | ---: | ---: |
| Seed 20260927; B=256; 2×128 glyphs; rho=1/129 | 0.01542 s | All 256: 0.33577 s | 2.17e-12 | 0 |
| Seed 20260928; B=1200; 4×400 glyphs; rho=1/225 | 0.38993 s | First 16: 0.13733 s | 3.96e-11 | 0 |

The larger call's ring is 691,200 bytes; process CPU was 0.5960 s and peak RSS
31,375,360 bytes. The smaller call's ring is 147,456 bytes; process CPU was
0.4165 s and peak RSS 28,098,560 bytes. Extrapolating the larger call's 16-key
reference to 1200 gives 10.30 s, **not a measured full-reference runtime**.
These are single artificial timings and provide neither empirical search
throughput nor a guaranteed speedup. Trace storage, neighborhood generation,
final original-engine checks, and fallback frequency are separate costs.

This kernel specializes the existing finite-state forward path sum. It makes
no new statistical identifiability or global optimization claim and changes
neither the original normalized model nor its model-description code.
