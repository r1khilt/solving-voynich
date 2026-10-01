# SOURCE-FRAGMENT-001: fragment proposals did not improve recovery

2026-10-01. Exploratory signal **NOT_SUPPORTED**. All eight cells completed,
but every selected key remained its baseline: zero score gain and zero edit
reduction in both arms. This is a negative solver result, not decipherment.

Registration `6e6eda61f79790af1fdbd51dcb038d7b908dfd21` was pushed and exactly
verified on origin/main before one run42259 and one full audit89257, both
terminal0. No retry, extension, retuning, new training or holdout access.
The [protocol](SOURCE-FRAGMENT-001.md) and [method review](../research/fragment-proposals-2026-10-01.md)
remain frozen. Original source, channel, native likelihood and reader unchanged.

| Exposed case | True letters in two records | Baseline edits | Fragment edits | Random edits | Correct used rows / used rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 128 | 93 | 93 | 93 | 5 / 19 |
| 1 | 128 | 109 | 109 | 109 | 2 / 18 |
| 2 | 448 | 302 | 302 | 302 | 5 / 21 |
| 3 | 448 | 351 | 351 | 351 | 5 / 20 |
| Total per arm | 1,152 | 855 | 855 | 855 | 17 / 78 |

Both arms0of8 exact records; no scored bank contained its true used mapping.
All cells stopped at three rounds, below the8192-key cap. The unchanged
round-start keys made rounds2/3 cached repetitions, with no new compositions.
Fragment11,600 unique full keys, random11,737; total23,337 keys/70,398 events.
Supported keys4,024/3,899 respectively (including baselines); unsupported
7,576/7,838. Every supported candidate scored no higher than its baseline.
This finite bank is not a local/global optimality or posterior-coverage result.
The earlier coupling diagnostic found improving one-row moves at these
baselines; failure of these fragment proposals does not make them local maxima.

## Where this attempt failed

All368 selected glyph windows exhaustively scanned the20,202 templates and
their bindings: **1,963,632,635 search nodes and110,693,848 compatible
template/binding matches**. The two rankings retained11,733 matches total.
These enormous match counts are accidental compatibility opportunities, not
evidence of language or decipherment. The matching itself was inexpensive
after compilation; exhaustive full-text scoring of110million proposals was
not attempted or justified.

The registered post-prediction coverage showed205 true unit-boundary windows
and **zero in-library true twelve-letter fragments at selected offsets**.
Not every such boundary had twelve source letters remaining. Explicitly
exploratory after-outcome bookkeeping found192 eligible true twelve-letter
starts among those offsets. None of the11,733 retained partial bindings was
entirely correct on its assigned rows, even allowing a different source
fragment to induce a correct binding. This does not prove that all110million
matches were wrong or that a sequence of overwrites could never reach truth.

The separate [post-outcome record](../../results/SOURCE-FRAGMENT-001/post-outcome-coverage.json)
checks **every** true source window in the four exposed fixtures:

| Source fragment length | In retained library / all true source windows |
| --- | ---: |
| 4 | 1,122 / 1,128 |
| 5 | 1,066 / 1,120 |
| 6 | 859 / 1,112 |
| 8 | 244 / 1,096 |
| 10 | 40 / 1,080 |
| 12 | 4 / 1,064 |

Only four true12grams exist anywhere in the retained library; none was at a
selected offset. These generated passages come from a **smoothed** source
model, which has positive support beyond retained literal contexts. Requiring
a common literal12gram was an unsuitable proposal restriction for these cases.
Adding iterations to the same fixed rankings cannot create missing templates;
changing offsets, ranks, lengths or the library requires a new registration.

The source mixture coefficient from depths≥12 averages0.0003857404, whereas
depths≥4 average0.8747958026 on these histories. This is a decomposition of
the source's smoothing weights, not a neural circuit, effective semantic
memory, or the target-specific probability supplied by those contexts.
It explains why “order12” does not mean samples reproduce stored12grams.
It does not establish that Latin is wrong or that the language model lacks
all useful information. Original neural key training uses actual corpus
windows, a different source distribution from these generated fixtures.

## Validation and resources

Run37.308278wall/37.005476CPU seconds/861,454,336 peak RSS bytes;
audit38.089821wall/38.046284CPU/868,368,384RSS. All2400wall/2200CPU/2GiB/128MiB
bounds held;1,331,638 ignored bulk bytes,0paid/GPU/new training/downloads.
The5–20minute estimate was conservative; actual run/audit each took≈38seconds.

[Full audit](../../results/SOURCE-FRAGMENT-001/audit.json) replays all template
scans,23,337 key scores,70,398 events, random draws, cache/round decisions,
predictions, diagnostics, coverage and hashes. Alternate1,472 individual
template checks use Python word-equation enumeration at every window.
Baseline/final full-record Python/native marginal discrepancies≤4.548e−13
against1e−7. Same author/backend/fit code; not independent-agent review or an
independently implemented global solver. Original arrays remain unchanged.

Final preregistration full2613tests+23subtests passed/13skipped;9new focused
tests passed including4096 depth12 length assignments. Scoped lint and diff
checks passed; five old full-tree lint findings retained. Post-outcome coverage
utility reproduces its closed record without overwriting or scoring; numeric
depth12 membership agrees with an alternate decoded-tuple set. No success
criterion was retroactively changed. Results/audit/derived bindings and exact
remote publication are recorded in the notebook.

Next direction: retain consistent partial key/source states without requiring
literal long fragments or an immediate whole-key score improvement. See the
[state-merging design](../research/after-fragment-source-state-2026-10-01.md).
No new policy or neural training is queued; the original fourth fit and final
all-four training audit remain pending. Actual Voynich decipherment unresolved.
