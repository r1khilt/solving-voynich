# SHARED-PREFIX-SYSTEMS-001

2026-09-30. Artificial engineering development; no trained source, empirical
cipher panel, transfer text, answer file, historical transcription or training.
The frozen GLOBAL-KEY-READ-001 comparison continues unchanged.

Question: can the qualified tiny shared-prefix search run with the actual
7.405M-parameter architecture and longer two-record inputs without prohibitive
Metal graph/state-cache allocation? Prior mathematical and decipherment/Voynich
scope review: [implemented engine](../research/recurrent-shared-prefix-engine-2026-09-30.md).
Previous SEQUENCE-MEMORY-001 found shape-sensitive GPU allocation for whole
sequence rescoring; that result does not establish transition-search memory cost.

Compare two execution arms in fresh sequential processes: original variable
step batch/device state cache; fixed8 single-step batches/independent CPU state
cache. Pad unused rows with BOS/zero states; ignore their scores/states entirely.
The CPU cache never merges different histories. Tiny independent full-sequence
and branch/state-storage tests precede publication. This is an execution change,
not a new language architecture or a causal interpretation of its neurons.

Generate all inputs deterministically from seed71821 without reading any prior
cipher corpus: uniformly random letters, six glyphs, duplicate-allowing23-row
units independently drawn from all42legal one/two-glyph units. Include the
generating key plus random whole keys, fixed uniform weights, no injectivity or
unit-count constraint. Two records per workload: (32source letters,1key,beam32),
(128,64,128), (224,1024,128). Independent key/input generation for each workload;
all shared across arms. Seeds71831/71839 create two untrained full-size models;
same initializer hashes required across matched arms. This measures resources
and numerical correctness, not learned-language recovery. The generated plaintext
is available solely to establish literal input provenance, not constrain search.

Fixed geometricrho1/225; expansion/cache100,000 and channel-cell10million caps.
Retain every returned tuple/counter/bound. Independently reencode every bank key,
check globally compatible key set and whole returned tuple's CPU float64 full
history score within.002nats. No history truncation. All six workloads must finish
and obey resource/numeric gates for an arm engineering PASS. Cross-arm tuple and
timing differences are reported, never selected to replace the primary reader.
No global optimality/evidence/interval certificate claim follows.

TwoCPU numerical threads plus oneGPU process alongside the four one-thread
CPU reader workers. Per arm600wall/500CPU seconds, outer615, sampled2GiBdriver/
4GiBhost guard after every transition; sequential maximum20.5minutes. Memory
guard is sampled, not a hard whole-machine cap. Resource or graph/cache failure
remains failure, no retry/extension; retain completed workloads/trace/missing
seeds. No paid API or training, expected paid cost$0. All-arm alternate accounting
audit120wall/100CPU; it verifies hashes/literal/scoring arithmetic/work/resources,
not another numeric model inference or independent agent approval.

Freeze/push/verify source before one invocation; use exclusive start/results.
Archive artificial inputs and reading/trace outputs under ignored outputs with
checksums, compact result/process/audit records in Git. CPU-only meaningful tests
precede measurement. Do not adopt the provider for trained-panel use until this
engineering qualification is complete; register that next comparison separately.

```text
PYTHONPATH=.:src .venv/bin/python scripts/benchmark_shared_prefix_systems001.py campaign --freeze SOURCE_COMMIT
PYTHONPATH=.:src .venv/bin/python scripts/audit_shared_prefix_systems001.py
```
