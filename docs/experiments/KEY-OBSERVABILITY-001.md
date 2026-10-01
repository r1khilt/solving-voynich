# KEY-OBSERVABILITY-001: a source-supplied key ambiguity oracle

2026-10-01, exploratory calibration. Question: when the original plaintext is
supplied but unit boundaries and the dictionary are withheld, are the used key
rows uniquely determined in the current inverse's 64 development episodes?
This isolates one necessary observability issue before attributing poor key
recovery to architecture, optimization or a mechanistic circuit.

This is **not** ciphertext-only decipherment, a neural experiment, a new
qualification panel or a Voynich reading. A unique source-supplied solution does
not imply that an unknown source can be recovered. A poor neural fit need not
be caused by ambiguity even if some cases are ambiguous.

## Sources and mathematical rationale

The existing PRIOR_WORK.md review retains Voynich substitution/anagram
applications and flexible-language artifacts; no proposed Voynich crib is used.
[Manea and Schmid2019v2](https://arxiv.org/html/1906.06965v2), §§1–3 definitions
and hardness discussion, treats uniform substitutions of repeated variables by
terminal strings, including the non-erasing case. General hardness does not
predict the cost of this fixed23-row/length1-or2 task.
[Day et al.2019](https://arxiv.org/abs/1906.11718), abstract reviewed, introduces
bounded word equations with linear length constraints via SAT. We do not
implement or replicate Woorpje; a direct bounded enumerator is sufficient here.
[Aldarrab and May2021](https://aclanthology.org/2021.acl-long.561/), abstract and
previous project review, concerns learned simple-substitution decoding; its
unknown-plaintext evaluation differs from this source-supplied oracle and our
variable-unit, duplicate-allowing teacher. The existing proper-loss and
key-inverse-next-objectives memos preserve uncertainty and objective caveats.

## Exact inputs and solver

Only the already exposed64 Pliny episodes from JOINT-KEY-TRAIN-001 preparation
are admitted. Bind the published preparation manifest, source and episode
archive hashes; reproduce their source windows, canonical ciphertext and gold
dictionary for provenance. The solver receives only source integer strings and
unsegmented ciphertext strings. It has no gold key, gold emitted-unit lengths,
raw-label order, dictionary presence mask or plaintext-boundary offsets. Gold
key comparisons are performed after enumerating solutions.

There are23source rows and6declared glyphs. Each row emits any of42singleton or
digram units, including duplicate assignments. Two original records share one
dictionary and reset their cipher/source offsets; no A,B duplication is used
as extra evidence. All source labels and source lengths are supplied to this
oracle and are explicitly unavailable to the trained ciphertext-only encoder.

Scan each known source left to right. An assigned row must match its existing
unit literally. At the first occurrence of an unassigned row, its only possible
values are the next cipher substring of length1 or2. Branch on both legal
lengths, propagate that row across all later occurrences and records, and
accept only complete exact record matches. Suffix occurrence counts yield safe
remaining-length bounds, with assigned lengths fixed and unknown lengths1..2.
Induction on first occurrences gives complete enumeration if no cap is hit:
every compatible used dictionary has exactly one sequence of length choices.
No injectivity, count constraint, neural score or language-model score is added.

Per case caps:1,000,000search nodes and4096stored used-dictionary solutions.
A cap hit is explicitly incomplete, even at exactly4096solutions; report its
observed count only as a lower bound, and never compute exhaustive posterior
statistics from a partial bank. Empty exhaustive support in a generated positive
is an engineering failure; a complete bank missing the true used dictionary is
also a failure. Retain incomplete cases and the whole64-case denominator.

## Metrics and claims

Record exact used support, determined used rows, unused row counts and search
work. Under the **iid uniform validation dictionary prior**, each compatible
used dictionary has the same probability. If S used assignments survive and U
rows never occur, there are S×42^U compatible complete dictionaries. The
source-supplied posterior has used-key MAP mass1/S, complete-key MAP mass
1/(S×42^U), and entropy log(S)+Ulog(42). Row-wise Bayes expected matches are
the sum of modal row frequencies divided by S. Store exact integer support
sizes and rational probabilities, not rounded probability claims.

Canonical first-occurrence naming does not change that equal-weight statement:
for a fixed canonical observation using m glyphs, each compatible canonical
dictionary has the same6!/(6−m)! raw relabelings. Remaining raw symbols are
ordered by the declared alphabet and unused rows stay unrestricted. An
independent tiny raw-dictionary enumeration checks this including unseen glyphs.
The training teacher's finite excluded-key prior is different; do not label
these numbers the posterior of the trained neural model.

These are conditional oracle posterior probabilities, not hard limits on the
number of lucky correct guesses in a finite observed sample. They do not
estimate ciphertext-only conditional entropy. No binary recovery-success
threshold is retrofitted to this diagnostic: report all cases, both unique and
ambiguous support, caps and exact arithmetic. Nothing changes the full-row
training loss, existing failures, main model selection or historic unit layer.

## Finite execution and validation

Publish source, tests and this protocol; verify the exact remote commit before
one invocation. Python3.12.13/NumPy2.5.3; pure Python search, BLASthreads1,
CPU-only, no model weights loaded or GPU inference, no training/paid/new holdout
use. One600stage-wall/500absoluteCPU-second run with2GiBsampled hostRSS, estimated
under2minutes but capped at10minutes. No retry or cap extension. Exclusive start
and ignored per-case solution trace; compact metadata/results tracked in Git.

One separate600wall/500absoluteCPU/2GiB artifact auditor replays all64bounded
searches and exact summaries, independently re-encodes every returned solution,
and verifies source/result/trace hashes. This uses the same counting algorithm
and author; it is not independent mathematical or paleographic confirmation.
Exhaustive tiny tests compare against full-dictionary enumeration over units,
cover repeated/duplicate/unused rows, ambiguous segmentation, shared records,
unsupported inputs, cap accounting and actual runner/audit serialization.
Full original training-ledger completion audit remains pending.

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=.:src \
.venv/bin/python -u scripts/run_key_observability001.py --freeze <source-commit>
```
