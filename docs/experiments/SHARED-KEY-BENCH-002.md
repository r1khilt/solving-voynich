# SHARED-KEY-BENCH-002: bounded-list alternative after scale failure

Prospective method/resource development using the exact retained artificial
inputs and1,197-key bank from BENCH001. No new random draws, key recovery,
empirical Latin transfer or manuscript access. Original500k-state failures
remain. See the [derivation and source review](../research/bounded-key-list-mixture-2026-09-30.md).

Use the same audited large compact source, same two remaining records for each
source-generated/shuffled workload and uniform key weights. Freeze k8before
either run. For each key, enumerate the8best whole-record tuples and the next
tuple score through exact per-record lattices and prefix A*. Re-score every
proposed tuple against all1,197keys with one globally consistent key. Compute
the weighted next-score bound for unseen tuples and exact mixture evidence.
Keep a result whose bound stays open; do not relabel it exact or increase k.
Report resource success separately from floating bound separation. Neither
means historical accuracy. All cases are adaptively chosen artificial development.

Validation in-run: every listed tuple re-encodes and its source probability
is replayed by a separate literal path loop; every32ndkey plus the last has
both record evidences recomputed by the original forward decoder. Tolerance
1e-7nats. Preserve all per-key readings in a hashed ignored archive, progress
counts and partial failures; use null only for mathematically negative-infinite
logs, and reject NaN/positive infinity.

Two sequential workloads;300wall seconds each,650totalCPU seconds,660outer
process seconds;500kstates/2Medges/50kprefixexpansions per record. Sampled4GiB
RSS cap, oneCPU process/thread,0paid. Expected1–10minutes. This may run alongside
the two bounded fitting workers, for a combined12GiB planning ceiling; the
two bank workers remain unchanged. Graph/prefix limits raise failure, no beam
or bank pruning. One attempt after source/protocol commit and remote verification.

```
PYTHONPATH=.:src .venv/bin/python scripts/benchmark_shared_key002.py --freeze COMMIT
```
