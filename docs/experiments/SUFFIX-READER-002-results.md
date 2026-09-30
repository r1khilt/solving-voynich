# Calibrated context weights help, but the reader still fails

2026-09-30. **Registered reader gate FAIL.** On 7,168 true letters, learning
context weights reduces errors from 254 to 229 (3.5435% to 3.1948%). The 9.8425%
reduction misses the required 25%; overall error exceeds 2%; three keys exceed
5%. The matched short-context ablation makes 258 errors. Longer histories help
in aggregate under this intervention, but do not establish dependable reading.

All keys were supplied. This is synthetic encrypted Latin with ambiguous
letter boundaries, not blind key recovery, language identification or a reading
of any historical manuscript. These are different passages and keys from
SUFFIX-READER-001; its lower absolute error is not a same-data comparison.

## Frozen procedure

The [registration](SUFFIX-READER-002.md), implementation and tests were pushed
and remotely verified at `313c68ea68354c1118a2a16593814978be94396a` before
preparation. Source counts used 50,000 Caesar letters. Four prescribed Adam
trajectories fitted twelve interpolation masses on the first 40,000 Virgil
letters. Selection used the remaining 10,000 letters, with contexts reset at
body and selection boundaries. All 2,004 iterates were retained; 24 specified
checkpoints and the fixed baseline were eligible. No cipher accuracy selected
these parameters.

The winning checkpoint starts at mass 1,024 and stops at step 100. Source
selection loss is 3.4239906 bits/letter versus 3.4413488 for fixed mass 256.
Later steps fit the calibration segment better but do not win selection.
Refit counts on Caesar and Virgil equal the previous 688,558-context archive
exactly; only interpolation weights change. Calibration/selection use reused
development source material, not an untouched validation author.

Calibration and all sixteen new cases were pushed and remotely verified at
`a56afd70e5a9bdc9fc6fae467d162ea70cb29518` before prediction. Two 224-letter
Cicero windows per key use the registered offsets within the second body,
without overlap with previous cipher windows. Cicero was already acquired and
processed. This is a new prospective development allocation, not independent
author replication.

All 128 predictions and sixteen per-case manifests were pushed and remotely
verified at `10a258710f53c8366e303e6392336dc3b4063cf5` before the single
accuracy evaluation. No reruns, dropped cases or adjusted thresholds.

## Complete comparison

Every key has 448 true letters. All arms receive the same complete true key;
they infer letter boundaries and plaintext length. Error means aligned
character edit distance, including insertions and deletions.

| Key | Fixed 3 | Fixed 12 | Calibrated 12 | Calibrated 3 |
| --- | ---: | ---: | ---: | ---: |
| 1 |14|7|8|7|
| 2 |17|20|20|17|
| 3 |25|19|17|27|
| 4 |18|18|15|20|
| 5 |24|24|20|22|
| 6 |12|12|6|12|
| 7 |4|4|4|4|
| 8 |28|25|27|25|
| 9 |27|27|27|23|
| 10 |29|27|19|23|
| 11 |24|16|8|18|
| 12 |3|3|5|3|
| 13 |9|6|9|6|
| 14 |8|8|8|8|
| 15 |28|28|28|28|
| 16 |12|10|8|15|
| Total |282|254|229|258|
| Error rate |3.9342%|3.5435%|3.1948%|3.5993%|
| Exactly correct records /32 |1|1|2|1|

Calibrated 12 improves seven keys, ties five and worsens four versus fixed 12.
Keys 8, 9 and 15 remain above the per-key error limit. Compared with its own
three-letter ablation, seven keys improve, three tie and six worsen. Aggregate
long-context benefit is the predeclared ablation result; it is not a significance
test or an alternative route to qualification. Fixed 3 is descriptive and does
not replace the stronger primary baseline.

Decoded totals are 7,085, 7,096, 7,112 and 7,091 letters respectively. The
calibrated long model still tends to produce shorter text, though a net length
deficit does not explain all 229 edits. Correct source length was not supplied.

## Validation and resources

Full pre-execution regression: **1,896 tests plus 23 subtests passed**, eight
skipped, 129.97 seconds. Fifteen new tests cover gradients against automatic
differentiation and finite differences, uniform-mass regression, full tiny
plaintext enumeration, record boundaries and bounded optimization. Changed
code is lint-clean; five previously recorded unrelated full-tree findings remain.

All 128 readings have separately written reverse-inference, MAP and direct
path-score checks, maximum discrepancy 1.0801e-12. Every reading re-encodes to
its ciphertext. All per-case archives equal the complete archive; input/source
bindings and all 128 independently replayed edit distances pass. Both new
implementations are root-authored; this is algorithmic checking, not replication
by another researcher. Previous experiment sources remain unchanged.

Measured stage CPU: preparation 7.870701 seconds, prediction 10.953467,
evaluation 0.452215; total **19.276383 seconds**, excluding tests and auxiliary
checks. Wall times 7.883870, 10.995553 and 0.486148 seconds. Peak reported RSS
1,155,629,056 bytes. One CPU thread, zero paid services. Efficient known-key
inference does not establish affordable blind search over dictionaries.

Compact evidence: [calibration](../../results/SUFFIX-READER-002/calibration.json),
[predictions manifest](../../results/SUFFIX-READER-002/prediction_manifest.json),
[evaluation](../../results/SUFFIX-READER-002/evaluation.json).
Predictions SHA256 `3efbaad0dd3c890f2af258cc5ec9c360fb45c1c5c8652ea364f4693262a91fe6`;
source counts SHA256 `5fcea148591aa23446b9cff3bd7ebf88eea55799bd70b585dca307d05ee2ba4d`.
Raw/bulk artifacts remain ignored. Exact commands and ordered freezes are in
NB-240 through NB-243.

## Decision

The source improvement is real on this fixed comparison but insufficient.
Do not spend another new panel on nearby smoothing choices or scale blind
search while this known-key failure is unexplained. Next examine the unchanged
model's posterior uncertainty and an explicitly answer-informed exact-length
intervention. These can distinguish model confidence and length ambiguity;
neither can establish an information-theoretic limit for natural Latin or
qualify the reader. Stronger source families and more varied source text remain
candidate changes after this diagnosis. No Voynich mapping or meaning is known.
