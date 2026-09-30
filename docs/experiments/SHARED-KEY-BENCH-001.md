# SHARED-KEY-BENCH-001: full-size feasibility before empirical use

Prospective resource probe, not recovery qualification. The mathematical model,
review of shared-key Bayesian decipherment and weighted determinization, and
artificial correctness checks are in the
[shared-key design](../research/shared-key-mixture-decoding-2026-09-30.md).
No Voynich, original confirmation or reserved-author passages are opened.

Use the audited large compact source, its frozen Pliny-selected tau64, and seed
552003. Generate six 224-letter records by sampling that source from a reset
state. The fixed length sets a hardware workload, not decoder assistance;
decoding retains rho1/225. Independently shuffle each encrypted record for a
second workload. A shuffled source-generated string is not a calibrated
historical non-language model.

Generate a key with all six singleton units and seventeen distinct pairs from
ABCDEF, then permute its 23 rows. Build the entire one-move bank: parent, every
row swap, every row replacement from all42units, deterministic deduplication.
There are1,197keys. This bank includes the generating key by construction and
cannot demonstrate unknown-key recovery.

For each of the two workloads, time exact per-key decoding on four fitting
records for32bank indices evenly spanning0..1196. Report the linear full-bank
projection, recognizing variation among candidates and real learned keys.
Separately decode the two other records with the entire uniformly weighted
bank. The sample's scores do not select or weight the bank. No accuracy metric
or parameter tuning is permitted. Record complete/failure, time, nodes/edges
where available, and peak RSS. A cap failure is retained, with no smaller-bank
retry in this namespace.

Limits:120wall seconds per operation,500,000states per exact decoder call,
600process CPU seconds,4GiB post-operation RSS check, one CPU process with
one numerical thread, zero paid cost. Four capped operations plus preparation
give an expected2–8minutes and a10-minute outer process timeout. The RSS check
is a sampled bound, not an operating-system hard memory limit. Use an external
process timeout as well as internal alarms; changing per-operation alarms
replaces the original whole-run alarm. Future empirical bank work must declare
its own budget from these observations; a favorable probe is not proof that
the empirical workload fits.

Freeze and publish this protocol, generator, bank constructor and decoder before
one invocation. Preserve all partial results and errors. No reruns or empirical
model claim. Command (thread counts all1):

```
PYTHONPATH=.:src .venv/bin/python scripts/benchmark_shared_key001.py --freeze COMMIT
```
