# Shape-only qualification: saved panels closed; original run failed reporting

2026-10-10 PDT. All sixteen finite panels completed their mathematical checks,
then the producer failed during counter aggregation. **The original scientific
run remains FAIL.** Its start/failure receipts and all 257 inputs are unchanged;
no original result, audit, retry or re-enumeration was created.

Two reporting defects are documented in the separately frozen
[receipt closure](CONSTRAINT-SHAPE-THEORY-001-CLOSE.md): four `00` panels omit
two conditional counters that are necessarily zero, and the registered
length-three dictionary-count formula overcounts by 168. The correct total is
24,000 assignments, using `72R²−60R` for the two mixed-width partitions.

| Preserved panel accounting | Total |
| --- | ---: |
| Panels | 16 |
| Shape states / hard states | 460 / 318 |
| Visited-key assignments / summed dictionary terms | 24,000 / 96,000 |
| Local / exchange flux checks | 23,154 / 66,780 |
| Stationarity equations | 3,680 |
| Restored zero fields in copied receipt data | 8 |

Original checker freeze `ebd45a9d9ad34bbdfc9000cd552de1133c91ede4`, session 4710,
terminal 1: 7.281947 wall seconds, 7.267492 CPU seconds, 225,230,848 peak RSS
bytes, $0 paid use. Failure hash
`dde6f5dac474699d30c3d35cfa5fa91edb0df454b4fc45e9d6c0a4a5f88aa3ac`.

Closure freeze `dd86839f62688176f6efb9981a136aeaa737395e` was pushed and its
exact origin ref verified before ONE execution, session 93842, terminal 0.
The separate [closure receipt](../../results/CONSTRAINT-SHAPE-THEORY-001-CLOSE/closure.json)
passes all 262 input bindings, ordered panel/counter schemas and corrected
arithmetic, with `original_status=FAIL`. Measured 5.817222 wall seconds,
1.090425 CPU seconds, 229,605,376 peak RSS bytes, $0. Its interval excludes
initial frozen-byte checks; the complete command took longer. No enumeration,
RNG draws, source scoring or training occurred in closure.

Thirteen focused receipt fixtures pass, including omitted/extra counters,
incorrect dimensions, altered allocations/hashes/resources and exclusive
output creation. A separate agent reviewed the mathematical implementation
and saved-panel arithmetic without editing inputs or rerunning the producer.
This is not external expert validation or a new proof of every saved check.

The implemented marginalization removes sampled dictionary noise. These tiny
finite checks do not establish useful warm-to-hard transport, mixing, original
source cost, literal text recovery or decipherment. Under the user's three-day
priority, another sampler-admission ladder is deferred while manuscript
mechanism predictions and historical decoding calibration proceed.
