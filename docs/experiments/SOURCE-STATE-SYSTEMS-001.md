# SOURCE-STATE-SYSTEMS-001

Exploratory engineering/coverage study registered2026-10-01. Question: does
exact Markov-state merging make compatible partial-key search tractable, and
does a32× wider guided frontier reach better complete hypotheses? Linked
negative evidence: SOURCE-FRAGMENT-001 and SOURCE-COUPLING-001. Read
[method/source review](../research/source-state-lattice-2026-10-01.md) and
[protocol](../research/PROTOCOL.md). No fresh recovery qualification or claimed
historical language/encoding; no success threshold is adapted after the run.

## Inputs and preparation

Use the SAME four exposed SOURCE-GUIDE fixtures, fixed seeds75501–75504,
two original records of64/64/224/224 source letters, iid23-row42-unit keys,
duplicate units allowed. Source letters/keys/true lengths are diagnostics only;
search takes observed glyphs and the fixed source arrays. Forced generator
lengths differ from search's original geometric prior rho1/225; do not conceal
that distinction. No new corpora, neural weights, paid API or holdout access.

Fixtures archive45672bytes SHA256
6372436be35865f6dde6f2fe2bd94b8918936970266ea5ca56523bba8fc0fd9e;
source count archive19566187bytes SHA256
9769415ac460c56778a461af7646635abf5906fa726484ba84d98d4f401815f6.
Original source/old native full-key kernel/reader stay unchanged. Source
order12/tau64,23 letters,399571824bytes dense float64/uint32 arrays. Parent
admission checks source arrays, audited parent fixture hashes and old build.

Compile a NEW C++17/O3/fPIC/ffp-contractoff ABI1 library in a fresh namespace;
pin source/library hashes and compiler/version/command in native-build.json.
No actual fixture search before registration. New tests compile temporary
libraries and independently enumerate tiny ciphers, including null support.
First preparation compiled successfully but manifesting failed on str/Path
conversion. The existing library was preserved and hash-pinned without a
second compilation; repair record is frozen. Original captured compile logs
were lost on that error; command reconstructed from the unchanged helper,
compiler version observed immediately afterward, missing logs labeled null.
This pre-empirical failure was not a cipher run or a replacement fit.
`scripts/run_source_state_systems001.py:PATHS` freezes all parent dependencies,
new code/tests/protocol/method memo/build manifest. Commit/push and verify the
exact remote registration before ONE run and ONE full audit. No retry, resume,
extension, replacement or retuning after any outcome.

## Fixed cells and observations

Four cases × ordered arms unmerged/merged/guided/wide_guided =16 cells.
All use balanced observed-glyph progress, shared once-binding iid prior and
original source edge weights. No source-history library restriction.

| Control | Width | Merge | Ordering | Pre-guide width | Call wall cap |
| --- | ---: | --- | --- | ---: | ---: |
| unmerged |128|No|geometric bound|unused512|120s|
| merged |128|Yes|geometric bound|unused512|120s|
| guided |128|Yes|iid future surrogate|512|120s|
| wide_guided |4096|Yes|iid future surrogate|16384|180s|

Narrow calls: expanded500000/generated20000000/pending100000/guide1000000
table-builds/cache1024. Wide: expanded4000000/generated120000000/pending500000/
guide20000000/cache4096. Max returned terminal keys512 for every cell.
Expand/preflight/time caps are normal accounted incomplete outputs; pending
allocation/guidework failures stop the single run and remain recorded. Time
checked before each whole-layer expansion, after guide/pruning, not a hard
preemptive foreign-call interrupt. No guarantee the wide arm finishes.

Primary systems metrics: expansions/generated edges/merged arrivals/pruned
states/peak pending/unpruned layer/guide builds/time/RSS/stop reasons, terminal
counts and found/unresolved evidence sums. Compare merging and guidance at
equal widths, report actual work/time; neither equal width nor the wide arm
has equal CPU. All found masses remain finite-beam lower sums; no posterior
coverage, full-key posterior, optimum or float interval certificate.

After closing search output, select highest returned partial-history mass,
lexical key tie. Fill unknown rows once using NumPy default_rng79101+case,
23 iid42 choices; same fill vector across arms. Score only that full key using
unchanged exact all-path marginal and read it with unchanged fixed-key Viterbi.
Selection mass is not the selected full key's score. No terminal means no
reading; report null edit errors/read-cell denominator, not zero-error success.
Close prediction before gold used-row/key-support/reading edit diagnostics.
Diagnostics cover only returned terminal keys, not all internal candidates.

Engineering execution reports all16 accounted outputs or a retained failure.
Only the complete audit can qualify execution correctness. Gold coverage and
edits are exploratory descriptive observations on repeatedly exposed keys,
not a fresh synthetic or historical recovery gate.

## Resources, audit, commands

One CPU process/BLAS1,0GPU/new training/paid dollars. Run3600wall/3200absolute
CPU seconds, sampled2GiB host RSS/256MiB ignored bulk. Sum call wall targets
≤2160seconds before source loading/readout/serialization; expected2–35minutes
is an unmeasured estimate. One audit same3600/3200/2GiB/256MiB caps. Original
finite MPS training continues separately; no inputs or queued fits changed.

Full audit verifies every cell/fixture/source/hash/config/summary/terminal/trace,
replays all state calls and all selected fills/scores/readings/diagnostics,
recomputes every aggregate/resource denominator. For a time-stopped call,
replay the identical prefix with max_expanded fixed to the already observed
whole-layer expansion count; audit per-call600s cap, overall3600s unchanged.
It cannot expand beyond that frozen prefix. Zero-expanded time stop fails full
prefix replay rather than silently declaring audit success. This is validation,
not a replacement fit. Alternate Python full-key DP checks all selected-key
record marginals within1e−7; tiny full-key/string rational checks supply an
independent correctness implementation. Full-scale search replay is same
author/native implementation, not an independent agent/global solver audit.

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python scripts/run_source_state_systems001.py prepare
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -u scripts/run_source_state_systems001.py run --freeze FROZEN_COMMIT
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -u scripts/audit_source_state_systems001.py
```

Run appropriate full-tree tests/scoped lint/diff checks before freeze; preserve
old unrelated lint findings. Publish results/failures, notebook and current
project memory with a verified routine commit/push. Actual Voynich unsolved.
