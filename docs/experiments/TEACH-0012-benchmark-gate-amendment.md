# TEACH-0012 benchmark-gate amendment

**Frozen before any TEACH-0012 training update or checkpoint.** The first source-matched
benchmark at launch commit `e22978d` passed every numerical and resource gate: all five arms
had exact repeated-forward agreement and finite gradients; peak sampled MPS allocation was
690,716,672 bytes; the measured campaign projection was 11,135.58 seconds and the frozen 1.5x
projection was 16,703.37 seconds, below the 21,600-second ceiling.

The subsequent scientific command stopped before constructing a training model. Its benchmark
guard compared the JSON-loaded config, whose seed tuples necessarily become JSON arrays/Python
lists, against `dataclasses.asdict(Config())`, which retains Python tuples. The values were
identical but the container types were not, so the guard falsely raised `benchmark did not
qualify or source/config changed`. A background attempt and the attached diagnostic attempt
both hit this same pre-training guard. No `status.json`, scientific `report.json`, loss,
checkpoint or prediction artifact was produced.

The repair changes only the guard's comparison: the in-memory frozen config is passed through
the same JSON encode/decode normalization before equality testing. Seeds, generator,
architectures, optimizer, schedule, evaluation suite, thresholds, resource caps and benchmark
measurements do not change. The original benchmark is preserved as
`invalid-config-roundtrip-benchmark.json`. Because the source hash changes, a new benchmark
must run from the repaired committed source; the old passing measurement cannot authorize the
scientific campaign.
