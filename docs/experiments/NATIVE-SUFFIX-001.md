# NATIVE-SUFFIX-001: exact marginal-only scoring, Python and native controls

Prospective computational test, before empirical use of the new scorer. The
previous key-bank reader improves development recovery but all eight null banks
remain unavailable from CPU limits. Wider key coverage needs affordable exact
scores, not a wider reading list for already-separated finite-bank objectives.

## Method and prior-work scope

The model, fitting objective and observation lattice are unchanged. The
[shared-key derivation](../research/shared-key-mixture-decoding-2026-09-30.md)
reviews weighted automata/decipherment; [the fixed-key diagnosis](KEY-SOURCE-DIAG-001.md)
records Voynich relevance and supplied-alphabet/language limitations. This is a
forward log-sum implementation of that same acyclic dynamic program, not a new
statistical model, novel algorithm, or historical-cipher recovery result.

At `(observed offset, source state)`, emit every source letter whose fixed unit
matches, multiply by its unchanged continuation/source probability, and sum all
paths arriving at the same next state. Add the geometric stop only at the final
observed offset. No probability pruning, beam, source quantization or key repair.
With nonempty emissions all edges point forward, so processed offset maps can
be released when no output paths are needed. Fitting weights need this marginal
likelihood; computing Viterbi text/backpointers at every candidate is unnecessary.

Compare three implementations: existing full Python decoder using the validated
dense source, Python marginal-only recursion, and C++ marginal-only recursion
using the very same immutable float64 probability/uint32 transition arrays.
The second arm isolates removal of Viterbi work from compiled execution. C++
uses log-domain arithmetic and compensated terminal summation; traversal order
can change floating roundoff. No fast-math or probability threshold. No extra
log-probability table or source model is trained.

Build explicitly in a new ignored output directory with the installed compiler;
record source/binary hashes, compiler version, flags and command. No compiler
download or global installation. ctypes verifies layouts, ABI and artifact
identities. Native errors/caps do not return usable partial likelihoods.

Twenty-six artificial tests cover exhaustive short observations at orders0/1/3/12,
independently enumerated text probabilities, repeated/overlapping units,
Unicode/NUL, a long underflow-prone input, unsupported/empty inputs, resource
caps, source/library/layout guards, pinned source buffers/metadata and refusal to overwrite build directories.

A standalone address/undefined-behavior sanitizer harness also exercises1,014
valid/error/cap calls. Its result and source are frozen with the benchmark. The
first direct CommandLineTools compiler invocation could not locate standard
headers; the installed `/usr/bin/clang++` wrapper used by the working test builds
compiled and ran it successfully. No sanitizer findings.

## Frozen workload and outcome

Use only original four fitting records and unchanged parent keys for all16old
CONFIRM001cases: eight positives/eight glyph-shuffles. For each full one-move
neighborhood take exactly8equally spaced indices, including first/last. All32
workloads/case run under each backend;512distinct workloads/1,536scorings total.
Rotate backend order by case index. Keep full likelihood/support/node/edge
records and timings. Load no transfer, gold, or new source corpus.

Require identical graph/support counts and absolute likelihood delta≤1e-7nats.
Report full initialization/compiler time separately, and timing aggregates for
positives/nulls as well as per case. Engineering gate: all16cases complete with
numerical agreement; total native time at least3×faster than full Python and
2×faster than marginal-only Python. This is a fixed hardware/workload criterion,
not a universal speed claim. Never remove a slow or failed null from the totals.
No fitting/recovery claim follows even if the computational gate passes.

OneCPU worker/thread, zero GPU/paid use,900wall/CPU seconds,960external timeout,
500kstate/2Medge per-record caps, sampled2GiB RSS. Expected1–10minutes. Native
build subprocess has60second timeout. No retries, budget extension or smaller
workloads; preserve partial results and process status. Benchmark source and
protocol are pushed/remoteverified before its sole invocation. Production
adoption or renewed empirical fits need a separately frozen protocol/namespace.

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
  .venv/bin/python scripts/benchmark_native_suffix001.py --freeze COMMIT
```
