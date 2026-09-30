# SEQUENCE-MEMORY-001: bounded artificial recurrent-scoring memory benchmark

Registered 2026-09-30, after KEY-BANK-NEURAL-001's resource failure and before any
full-size run of this benchmark. Engineering diagnosis only; no cipher, corpus,
trained checkpoint, or answer is read. No linguistic or decipherment claim.

## Question and rationale

Does smaller batching alone, or smaller batching with fixed padded shape, permit
whole-history scoring under the same 2 GiB driver / 4 GiB host guards? The failed
run used 64-row variable-length batches. Fixed shapes may reduce shape-dependent
workspace/cache growth; this is a hypothesis, not an established cause.

The underlying source probability and shared-key objective are unchanged. The
neural decipherment rationale remains the review in KEY-BANK-NEURAL-001, including
[Kambhatla et al. 2018](https://aclanthology.org/D18-1102.pdf); the existing
Voynich/unknown-unit reviews still delimit applicability. This benchmark is
about executing that score, not a new architectural or historical inference.

PyTorch 2.14 documents that
[driver allocation](https://docs.pytorch.org/docs/2.14/generated/torch.mps.driver_allocated_memory.html)
includes allocator caches and MPS framework allocations. Its
[tensor allocation](https://docs.pytorch.org/docs/2.14/generated/torch.mps.current_allocated_memory.html)
excludes unused cached allocations.
[Emptying the cache](https://docs.pytorch.org/docs/2.14/generated/torch.mps.empty_cache.html)
releases unoccupied allocator cache, not all live framework memory. Therefore
record both domains, use fresh processes per arm, and avoid claiming host RSS
alone measures GPU demand. Documentation consulted 2026-09-30.

## Fixed workloads and arms

All three arms receive the same 4,096 artificial strings: NumPy generator seed
71001, independent uniform letters from the fixed 23-letter alphabet, record i
length `180 + (37*i mod 91)`. Save a SHA-256 identity. Two newly initialized,
untrained 7.405M-parameter 96-embedding / 2-layer / 768-width LSTMs, seeds 71101
and 71109, run sequentially in each arm. Save initial-weight hashes to verify
matching models across completed arms. No optimizer or checkpoint loading.

1. `variable64`: original scorer, batch64, each batch padded to its longest row.
2. `variable8`: same original scorer, batch8, variable padded length.
3. `fixed8`: new scorer, exactly 8 rows ×270 positions on every call, including
   the partial final batch. Extra rows and right-padding positions earn no score.

All complete histories begin at BOS; no mid-record reset, truncation, sequence
packing, reduced precision, probability interpolation, or recurrence change.
MPS forward uses float32; CPU log-softmax and accumulation use float64. Scores
here exclude the external length law, identically in all arms. Input validation
rejects oversized/out-of-alphabet records before any forward call.

## Resources and criteria

Three fresh processes run sequentially in the listed order, one MPS process and
two CPU numerical threads at a time. Each has 180 wall /150 CPU seconds, a
195-second outer deadline, sampled 2 GiB driver and 4 GiB peak host RSS guards.
Sample/flush a trace after every synchronized batch, before raising on excess;
record initial/model-cleared allocations too. Clear unused cache between models,
not between batches. Timeouts and failures are retained with no retry or extension.
Full validation allowance600CPU seconds; component ceiling1,050CPU seconds under
18CPU-minutes, expected2–5wallminutes, zero paid spend. Fresh CONFIRM-002 CPU
workers may overlap; retain wall measurements rather than treating load as fixed.

For each completed model, compare 16 evenly spaced records with the original
full-forward CPU float64 scorer, maximum delta≤1e-3nats/record. Require both
fixed8 models and all8,192 records complete under guards with these numerical
checks for engineering PASS. Tiny tests separately compare independent one-step
recurrence, >512-character histories, dummy rows, errors, and exact call shapes.
Compare saved per-record scores across any completed arms; report differences
and timing without promoting an incomplete control to a pass. Numerical samples
are not a proof about every trained-model candidate ranking.

Report exact sampled driver/tensor peaks, host RSS, throughput, failures, initial
weight identities, data checksum, and prospective time for 37,032,100 candidate
letters. Throughput is a planning estimate, not an accuracy gate or empirical
repair. No automatic neural rerun: a successful engineering result may support a
separately frozen KEY-BANK-NEURAL-002 registration with all original cases and
both trained seeds retained. NEURAL-001 stays failed. CONFIRM-002 stays frozen.

```text
.venv/bin/python -u scripts/benchmark_sequence_memory001.py campaign --freeze SOURCE_COMMIT
```
