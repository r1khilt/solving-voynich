# Global pair rewrite: finite correctness passes, recovery remains untested

2026-10-08. Exact scientific freeze `e4577c49165cfe97d956f663a792e15c4c056815` was
remotely verified before one checker and one receipt closure. Neither was retried.

The twelve registered binary panels contain3,392 complete visited states and71,904
ordered-triplet cases. Every global merge/split preserves the encoded records, inverts
the complete visited state, reconstructs the correct action schedule and agrees with
full reference replay. Both the uniform-all and uniform-valid corrected selectors pass
exact rational detailed balance, row normalization and3,392 stationarity equations each.

| Check | Witnesses |
| --- | ---: |
| Merge / inverse split edges | 2,712 / 2,712 |
| Identity triplets | 66,480 |
| Repeated replacements / changes across records | 1,224 / 936 |
| Changed visited-row count | 2,448 |
| Unsafe split with old pair fails inversion | 1,464 |
| Omitted visited prior breaks balance | 2,448 |
| Omitted geometric continuation breaks balance | 5,424 |
| Wrong action-policy q correction breaks balance | 5,424 |
| Omitted legal-list size correction breaks balance | 4,248 |
| Corrected legal-selector structural edges pass | All 5,424 |

The fixed-rank selector chooses many identity moves on these small panels. The efficient
selector's legal list agrees with independently testing every triplet; its acceptance needs
the old-list/new-list size ratio. These finite checks establish correctness under the stated
artificial model, not speed, mixing, irreducibility, useful acceptance or recovery.

Checker2.929316wall/2.916853CPU seconds,208,420,864peakRSS bytes/$0. Receipt closure
0.000633wall/CPU seconds,203,128,832peakRSS bytes/$0; all14 input hashes and counter/resource
identities pass. It does not repeat kernel enumeration or constitute independent expert review.
Source environment: NumPy2.5.3/Torch2.14.0/Python3.12.13. No original corpus, learned weights,
Gold/manuscript archive or neural evaluation was used. Full test suite3,061 tests plus23
subtests passed,7 skipped;14 focused tests passed. New scoped lint clean, five old full-tree
findings preserved.

The next separately registered original-source admission tests literal correctness, full
source/target ratios and cost on previously failed archived readings and all retained nulls.
No decipherment or competence claim has passed.
