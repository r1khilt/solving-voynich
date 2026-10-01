# Whole-key revision improves errors only modestly

2026-10-01. Exploratory result on FOUR previously exposed synthetic keys,
not a recovery qualification or manuscript decipherment. Registration
891dd4e971dffd7a5739c59626996ecfa19e09ed was pushed and its exact remote
main ref verified before the single run. No retry or tuning after launch.

The search could revise every symbol mapping, but its total edit errors fell
only from4527 to4295 per4608true source letters: **5.1248% fewer errors**.
It improved12of16cells and worsened4; all32 selected passages remained wrong.
The prespecified requirement was at least25%aggregate improvement and12better
cells, so the exploratory signal is **NOT_SUPPORTED**. Every cell completed
and reached its8192-key cap; no graph/decoder/numerical failure or approximate
scoring occurred. The search was finite, not exhaustive or stationary.

## All outcomes under the same reading rule

Before and after revision both use exact fixed-key Viterbi. Warm keys are the
highest original literal particle leaf with unseen rows iid-completed, rather
than the parent's modal reading. Consequently these before/after values are
not the parent's original error baseline. Both arms use the same filled-key
scoring/decoding procedure, seed schedules and unique-key cap.

| Warm-start bank | Before errors | After errors | Gold-used rows matched before → after | Score gain, nats | Generating-key score above selected, nats |
| --- | ---: | ---: | ---: | ---: | ---: |
| case0-75511-none | 172 | 158 | 0 → 0 | 91.93 | 166.15 |
| case0-75511-iid | 109 | 93 | 2 → 5 | 78.43 | 147.69 |
| case0-75513-none | 157 | 155 | 1 → 1 | 47.77 | 170.38 |
| case0-75513-iid | 123 | 127 | 0 → 0 | 36.82 | 182.30 |
| case1-75511-none | 143 | 144 | 1 → 0 | 95.80 | 128.92 |
| case1-75511-iid | 106 | 109 | 3 → 2 | 30.50 | 113.71 |
| case1-75513-none | 153 | 143 | 0 → 2 | 40.05 | 141.72 |
| case1-75513-iid | 80 | 72 | 6 → 7 | 35.81 | 115.77 |
| case2-75511-none | 531 | 493 | 0 → 0 | 251.06 | 533.00 |
| case2-75511-iid | 346 | 302 | 4 → 5 | 174.63 | 445.06 |
| case2-75513-none | 557 | 503 | 3 → 1 | 191.50 | 511.59 |
| case2-75513-iid | 341 | 340 | 6 → 6 | 110.24 | 426.22 |
| case3-75511-none | 483 | 457 | 1 → 2 | 363.24 | 475.68 |
| case3-75511-iid | 355 | 351 | 5 → 5 | 208.32 | 419.25 |
| case3-75513-none | 496 | 445 | 1 → 2 | 113.39 | 457.75 |
| case3-75513-iid | 375 | 403 | 2 → 1 | 207.62 | 450.88 |

Constant-guide warm banks:2692→2498errors/2304letters, used rows7→8/156.
Future-guided warm banks:1835→1797errors/2304letters, used rows28→31/156.
Combined used-row matches35→39/312. The312counts repeat four actual keys
across seeds/guidance; they are not312 independent discoveries. An edit ratio
can exceed100% when wrong readings are longer than the true text.

## What the supplied-key diagnostic establishes

After immutable predictions, we supplied the generating key to the SAME exact
reader. It produced64errors/4608letters (1.3889%) and16exact records/32
repeated evaluations. Deduplicated over four fixtures:16errors/1152letters
and4exact records/8. Per fixture, the two-record errors are0,1,14,1.

Thus this reader/source combination can recover nearly all letters when its
mapping is known, although duplicate units still leave ambiguity and the
forced source-length fixtures differ from the decoder's geometric-length law.
This diagnostic does not identify the exact global MAP text or quantify all
posterior uncertainty. It is not a new unassisted recovery result.

In EVERY cell, the generating full key scores above the best visited key by
113.7074..532.9996nats under the actual frozen objective. None of the16banks
contains that full key OR its gold-used mapping. This is a concrete witness
that the search missed a much better available explanation; it does not prove
the generating key globally optimal or prove which moves cause the failure.
All selected scores improve30.5026..363.2376nats, including the four cells
whose true-text errors worsen. Better objective values are not recovery.

## Work, validation and remaining limits

One run70520 terminal0 completed16cells/131072unique-key evaluations summed
across cells/132350proposals. Exact native scoring sums every compatible
source path with the uniform42^23full-key prior; it uses no old MDL penalty.
Run270.512723wall/270.170657CPU seconds,905,691,136peakRSSbytes;
14,373,932ignored bulk bytes. Within2400wall/2200CPU/4GiB/512MiB bounds.
Zero GPU/new training/paid API/download/holdout use; actual fixed statistical
source prediction and exact reading computation occur.

Before launch:2596tests+23subtestsPASS/13skips/152.91s,7focused tests/scoped
Ruff/diffPASS. Fulltree retains exactlyfive previously recorded unrelated
unused-variable/import findings. Original training and guide freezes remain
unchanged; no shared kernel/scorer/source or original charter edited.

One completion audit39876 terminal0: PASS_full_score_rng_trace_and_reading_replay.
It checked all131072key scores and132350proposal events, full proposal/exchange
RNG and acceptance/accounting, exact predictions/gold diagnostics/parent/source
hashes and aggregate signal clauses. Alternate scalar traversal of warm/final
full records agreed with native scoring within5.68434e-13 (bound1e-7).
Audit261.414937wall/261.088131CPU seconds,985,710,592peakRSSbytes; all bounds
held. It is authored here and reuses the native scorer; no independent-agent
or independent global-search claim. Result5906bytes SHA256
a97d1d1970bad8fca44a011d0a667bb80e3dee85f8ada5852abeb69a80226069.

The next useful question is how to propose coordinated mapping changes using
the observed constraints. These results alone do not show that additional
random proposals or a larger neural network will find the right assignments.
A diagnostic of conditional move gains, joint-row interactions and support
barriers would distinguish local improvement from coordinated correction;
gold-assisted tests must be labeled as such and kept outside a recovery solver.
No such new diagnostic or revised search is launched by this report.

The final95M neural fit remains in its original bounded queue. Neither this
search nor the unfinished training campaign identifies Voynich's language,
glyph units, encoding rules or meaning. Voynich remains unsolved.
