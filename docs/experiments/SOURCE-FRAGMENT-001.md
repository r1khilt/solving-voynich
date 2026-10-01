# SOURCE-FRAGMENT-001

Registered exploratory development experiment, 2026-10-01. Actual Voynich
decipherment remains unresolved. Question: can exact source-fragment bindings
propose useful coherent key changes where blind local blocks failed?

Read [method rationale and primary-source review](../research/fragment-proposals-2026-10-01.md)
and [research protocol](../research/PROTOCOL.md). This is a new proposal engine,
not a change to any live/frozen model, language prior, channel or original test.

## Inputs and immutable preparation

`scripts/run_source_fragment001.py:PATHS` binds all parent dependencies, the
source/compiler code, tests, this protocol/research memo, and preparation
manifest. The existing `results/LATIN-SOURCE-COMPACT-001/large.json` pins the
19,566,187-byte count archive with SHA256
`9769415ac460c56778a461af7646635abf5906fa726484ba84d98d4f401815f6`.
Original native full-text likelihood build and dense probabilities/gotos are
unchanged and checked. Use only the four **already exposed** SOURCE-GUIDE
fixtures, with SOURCE-REVISE iid-guided seed75511's selected key/reader as
baseline. Two source lengths64 and two224, two original records/key, 1/2-glyph
units, 23 source rows, 6 glyphs, duplicate units allowed. No new independent
keys/author/folio holdout and no unseen language or mechanism qualification.

Compile a separate small matcher before freezing; save its compiler command,
version, C++/library hashes and ABI1. Tests use fresh temporary libraries and
independent tiny length-assignment enumeration, not exposed empirical cases.
Publish and verify the exact registration commit remotely before **one** run.
Then execute **one** full audit. No retry, extension, retuning or replacement
of a failed cell under this ID. Compilation/tests are engineering preparation,
not an empirical sample or recovery result.

## Fixed workloads and decision

Four cases × fragment/random-length-matched arms = eight cells. Source12gram
library20202; sixteen glyph anchors/record with valid ±1 neighbors, at most96
offsets/case; every offset exhaustively scans all templates/bindings.
Top16 per each of two fixed rankings, union at most32 candidates/window;
50,000,000 nodes/call, cap hit is a failure. No beam pruning in the matcher.
All scans computed once/case and reused by both arms/three rounds.
Random control77111+case; masks and lengths exactly matched, equality with
the original target or baseline is allowed; no rejection to force changes.

Three rounds, at most8192 distinct full keys including baseline per cell.
Fixed round-start key; retain all new scores/events and earliest maximal key;
normal termination at key or round budget, without interpreting it as
convergence. Whole-key score sums two exact native record marginals, rho1/225,
uniform42^23 prior. Each record2M nodes/8M edges hard failure bounds.
Warm/final same exact fixed-key Viterbi reader,500k nodes/2M edges/50k expanded;
cross-check its marginal and scalar source score against the native likelihood.
No MAP-over-readings or posterior coverage claim.

Close fit/prediction archives before gold diagnostics; close both arms before
true-fragment coverage. Primary exploratory signal SUPPORTED only if all eight
finish, the fragment arm reduces baseline total edits by at least10%, and its
final edits strictly beat random in at least3of4 cases. Report all errors,
exact records, used-row matches, unsupported candidates, scores, caps, time,
and library/ranking coverage. These four repeatedly studied keys permit no
population significance statement or fresh recovery qualification. A failure
is retained; exploratory thresholds are not retroactively changed.

## Resource budget, commands, validation

One CPU process/BLAS1,0GPU/new training/paid calls/downloads/holdout access.
Run2400wall/2200absolute CPU seconds; one audit same caps. Each2GiB sampled
host RSS/128MiB bulk cap. Up to65,536 full-key evaluations/run,131,072 native
record scores plus bounded warm/final decoding. Matcher worst-case19.2B visited
nodes across384 windows, although actual calls may fail sooner. Expected5–20
minutes/run is an estimate, not a measurement; maximum40minutes each stage.
Original finite MPS training continues separately with unchanged inputs.

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python scripts/run_source_fragment001.py prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -u scripts/run_source_fragment001.py run --freeze FROZEN_COMMIT
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -u scripts/audit_source_fragment001.py
```

Audit full template scans, literal emissions, all scored keys, random targets,
proposal order/round bases/cache/best selection, predictions, diagnostics,
coverage and aggregate clauses/hashes/source identities. At every window check
four fixed individual templates by alternate Python word-equation enumeration
over all legal consumed lengths; tiny tests exhaustively enumerate independent
length assignments and check top-rank/tie union, duplicates and caps. Full-record
Python DP cross-check every distinct baseline/final key within1e−7. Full replay
shares the native matcher/fit code and author; do not call it independent agent
review or an independently implemented global solver. Full tests and scoped
lint/diff checks before registration; validate/report artifacts and commit/push
the outcome checkpoint with notebook and project memory updates.
