# All-pairs diagnosis: wrong inventory and reading structure dominate

All485 states and122705 pair transformations completed under exact published freeze
`c507e40bfb0adbc896a1f27c9a90b54cab595d8f`, original invocation12372/terminal0.
Every pair panel was sealed before known-answer metrics. The single registered independent
audit95572/terminal0 passed every transformed literal reading, exact source target/sign/
fingerprint/log/probability, selected neighbor and structural/metric accounting. All160
inputs stayed unchanged. No retry, panel extension, training, sampling or chain occurred.

**Pure letter reassignment cannot recover any of the320 tested positive reading states.**
Every state lies outside the generating reading's permutation orbit. The existing encoding
inventory also prevents complete used-dictionary recovery in every case, regardless of
how its labels are rearranged. This is a verified obstruction for these fixed synthetic
states and this particular operation, not an impossibility theorem for decipherment.

| Phase | Maximum correct used bindings under any label permutation /1246 | Length-only edit lower bound | Reachable correct readings /64 |
| --- | ---: | ---: | ---: |
| Initial | 188 (15.09%) | 16146 | 0 |
| 92821 root-only | 273 (21.91%) | 14525 | 0 |
| 92821 regrowth | 426 (34.19%) | 14509 | 0 |
| 93821 root-only | 275 (22.07%) | 14450 | 0 |
| 93821 regrowth | 425 (34.11%) | 14556 | 0 |

The dictionary ceiling is the exact maximum for dictionary matches alone across ALL
global permutations, including unused rows; it need not coexist with the correct text.
The length lower bound is invariant under label moves. Gold contains18772 letters; the
wrong readings are mostly too long. Cross-record repeated-letter equality patterns disagree
in all320 positive states. Only0/2/1/2/1 cases respectively share Gold's per-record source
lengths (and those cases also share unit boundaries); none has the required equality pattern.

## Best single move by target score

The first lexicographic strictly best target-increasing neighbor was selected without
Gold, or the base was retained if no pair improved. This is a one-hop diagnostic, not a
solver run or posterior draw. Every phase still has0 exact records/128 and0 complete
used dictionaries/64 after this selection.

| Phase | Edits before→after | Correct used bindings before→after | No uphill label pair /64 | Positive best-target gain sum, nats |
| --- | ---: | ---: | ---: | ---: |
| Initial | 26012→26014 | 28→28 | 41 | 528.638318 |
| 92821 root-only | 24835→24828 | 41→42 | 39 | 579.755766 |
| 92821 regrowth | 24706→24704 | 49→50 | 52 | 88.171097 |
| 93821 root-only | 24840→24824 | 34→35 | 39 | 523.655860 |
| 93821 regrowth | 24814→24814 | 40→40 | 45 | 87.984682 |

Regrowth increased available true codes relative to the initial inventory, yet its best
label neighbors barely improve known-answer accuracy. A larger model score is not a
translation. The no-uphill count certifies only a transposition local maximum; larger
permutations could improve the score, but no permutation can remove these structural
obstructions. The study does not enumerate every permutation's target score or prove a
global optimum. The exact dictionary ceiling and orbit criterion are different claims.

## Full nulls and resource accounting

All33 nulls in each phase were retained:16 within-record shuffles,16 IID glyph records and
one unrecoverable ambiguity. Across initial/root92821/regrowth92821/root93821/regrowth93821,
null uphill-pair totals are23/14/18/14/12. Their best-target gain sums are277.961922/
143.649974/34.893747/143.649974/23.284654nats. Nonsense also permits target increases;
there is no calibrated semantic-rejection gate or invented null plaintext here.

Actual producer526.928234wall/524.243731CPU seconds,1052393472bytes peakRSS,
9309544ignored private archive bytes/$0. Audit423.230044wall/419.380909CPU seconds,
1070825472bytesRSS/$0. Both held7200wall/6900CPU/2GiB bounds, producer private bytes
below256MiB. The20–60minute planning range was conservative; the fixed panel and budgets
were not expanded after the measured shorter run.

The auditor independently rescored dense coefficients in per-record source order,
reconstructed transpositions and target-selected neighbors, used canonical equality patterns
and multiset intersection for orbit/ceiling checks, and checked errors with separate integer
dynamic programming. Shared literal/hash helpers and the same researcher remain limitations;
this is not external expert replication. These64 cases are exposed development data.

[Registration](READING-LABEL-LANDSCAPE-001.md),
[method and preparation controls](../research/reading-label-landscape-2026-10-02.md),
[producer result](../../results/READING-LABEL-LANDSCAPE-001/result.json),
[sealed panels](../../results/READING-LABEL-LANDSCAPE-001/sealed-cells.json),
[independent audit](../../results/READING-LABEL-LANDSCAPE-001/audit.json).

The next design must revise inventory, segmentation and repeated-letter structure coherently
across all records. An [all-occurrence merge/split candidate](../research/global-pair-rewrite-proposal-2026-10-02.md)
has a proposed reversible triplet construction and initial primary-method review, but no
implementation or qualification yet. Pure label chains are deprioritized as the main
repair for this stage; labels may still complement a genuinely structural kernel. The
source model's adequacy and global search remain unresolved. No historical decipherment,
neural circuit or new recovered message is claimed.
