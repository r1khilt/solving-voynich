# LATIN-SOURCE-COMPACT-001 results

2026-09-30. Source revision7233373a3713674d3796cedac84c47e6ebe948d2
was pushed and its remote main ref verified before one small fit, one large fit
and one audit each. All four processes exited0. The original large statistical
arm's1.2million-context failure remains; this separately named adaptive storage
revision supplies a supplementary comparison, not a retroactive PASS.

| Same statistical rule | Small text | Large text |
| --- | ---: | ---: |
| Eligible training letters |97,850|6,497,939|
| Retained contexts |21,553|1,447,724|
| Selected interpolation mass |64|64|
| Pliny bits/letter |3.03741466899|2.44209291570|
| Stored count-array bytes |1,728,752|115,093,408|
| Fit CPU seconds |0.213791|5.009964|
| Fit wall seconds |0.216304|5.018253|
| Fit peak RSS bytes |299,679,744|1,522,024,448|

Both score the same359,276Plinyletters with identical512chunk resets. More text
reduces selection loss by0.59532175329bits/letter, or19.60%. This is a source
prediction result after source-only smoothing selection, **not** a19.60%reading
accuracy improvement. One author and one data-family comparison do not establish
performance on medieval Latin, other languages or the Voynich manuscript.

Every small-data count equals the frozen original dictionary archive, and all
four tau scores are identical. Source support cutoff/order/root/interpolation
were unchanged. The large corpus required1,447,724contexts, explaining why the
older1.2million-row cap was insufficient without implying physicalRAMexhaustion.
The new compact storage remained far below its explicit8million-context/2GiBarray
limits. No paid services were used.

Separate audits checked every archive/input hash and structural closure,
all root counts and49root/context checks per dataset via overlapping substring
search. Scalar bisect plus recursive literal-history scoring reproduced each
selected model's full359,276letter selection score with zero observed difference.
Small/large audits used4.637154/7.667421CPUseconds,4.644257/7.675364wallseconds,
333,217,792/524,255,232peakRSSbytes. Full source revision regression had1,989tests
and23subtests pass,9skips. These are separate computations by the same agent,
not independent researcher replication.

Bulk count archives remain ignored: smallSHA26438ec1,304,162bytes;
largeSHA9769415a,19,566,187bytes. Complete hashes, all four mass scores,
per-depth inventories, commands and resource reports are in
`results/LATIN-SOURCE-COMPACT-001/`. No reserved-author archive or new cipher
panel was opened. The paired neural fits are still running; their final
comparisons and audits must be reported separately.
