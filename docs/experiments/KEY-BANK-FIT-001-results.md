# KEY-BANK-FIT-001: all positive banks complete; all null banks unavailable

Source/protocol `60cfa9069212e19fd6eb09f3d0eb81789e92d1d6` was pushed and
remotely verified before the single campaign invocation. Terminal process report:
8 positive cases exit0, all8 glyph-shuffle cases terminate with SIGXCPU(-24)
at the registered600CPU-second limit. No restart, partial normalization or
replacement case. Campaign status **incomplete_failures_retained**.

This stage fits ciphertext only. Objective gains are not plaintext recovery.
Transfer predictions and answers have not been opened in this campaign.

| Positive case | Complete candidates | Finite fitting mass | Best objective gain vs parent, bits | Maximum conditional bank weight |
|---|---:|---:|---:|---:|
|1|1193|1030|5.254246|0.548710|
|2|1194|1073|0.230128|0.035504|
|3|1194|1073|0.005972|0.030797|
|4|1189|946|182.354070|1.000000|
|5|1194|952|0.005535|0.030847|
|6|1194|952|1.174972|0.062375|
|7|1193|1152|20.414725|0.992720|
|8|1191|1109|77.971154|1.000000|

Weights rounded for display; all finite log weights remain in the archives,
including extremely small positive masses. The data-dependent neighborhood is
not the full key family or a calibrated posterior over that family.

All9,542 positive candidates are inventoried,8,287with finite fitting mass.
Eight audits pass complete neighborhood/move enumeration, per-record likelihood
sums, unchanged code penalties, normalized weights, progress and archive hashes.
Independent reverse inference replays parent/best/middle/last keys on four fit
records:128 checks, maximum difference below1.60e-12nats. This samples numerical
inference; it does not independently recompute all9,542candidates.

Terminal accounting rechecks74frozen source/input bindings and retains hashes
for all process logs and compressed progress files, including killed workers.
Unfinished gzip streams are not treated as complete archives. No complete null
bank exists; no all-case success or null-discrimination claim is available.

Campaign elapsed2,666.926435seconds(44.45minutes), two single-thread CPU workers.
Completed positive workers total376.259256wall/376.008268CPU seconds. Killed
workers have no final CPU/RSS reports; their600second limits are recorded as
limits, not fabricated precise measurements. No GPU or paid service; $0.

Next: separately frozen KEY-BANK-READ-001 tests original key, fitting-selected
key, joint key/text choice and summed-key reading on the old transfer passages.
The cases were historically exposed and remain development examples. The
validated dense lookup adapter changes source execution cost, not probabilities.
