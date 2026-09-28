# Blind unit recovery: a search failure with a concrete diagnostic

2026-09-27. The [registered exposed pilot](BLIND-CHANNEL-DEV-001.md) did **not**
recover any of its four messages within the10%character-error diagnostic.
Independent evaluation replay passed. No Voynich text was processed.

| Case | Transfer edits /448letters | Edit rate | Correct-channel oracle edits | Decoded letters | Diagnostic flag |
| --- | ---: | ---: | ---: | ---: | --- |
| A key1, single glyph | 71 | 15.85% | 0 | 439 | yes |
| A key2, single glyph | 188 | 41.96% | 0 | 443 | no |
| B key1, hidden variable units | 477 | 106.47% | 4 | 672 | yes |
| B key2, hidden variable units | 430 | 95.98% | 21 | 594 | yes |

Character edit rate can exceed100%because insertion errors count. No learned
positive transfer record was exactly recovered. All four paired within-record
glyph shuffles were unflagged. However, three badly recovered positives passed
the preregistered fit-compression/transfer-likelihood flag. Thus the flag detects
some structure in this small panel; it cannot certify a reading. The nulls are
only four exact-unigram shuffles, not an extensive nonlanguage benchmark.

## What actually failed

The correct A encodings cost216.866 and302.241fewer fit bits than the selected
models under the **same** source and model code. They are in the available
candidate pool. These runs therefore missed concrete better candidates.
Correct-key A decoding itself is automatic and does not prove source quality.

For B, the literal gold channels cost778.729 and332.917fewer bits, but each has
one unobserved two-glyph unit absent from the proposal pool. An initial draft
of the diagnostic's interpretation incorrectly said all gold units were in the
pool; the printed counts contradicted it, and it was corrected before publication.
The corpus grammar permits those units, but the restricted proposal pool did not.

We then constructed an explicitly **gold-assisted, post-evaluation diagnostic**:
replace every gold emission absent from the pool with its lexicographically
first singleton, `A`, retaining all other rows. Since the removed unit never
occurs anywhere in fit ciphertext, no previously compatible fit path is lost.
The replacement can add paths and is one glyph shorter, so it cannot worsen
fit likelihood or this code length. Both resulting channels satisfy the same
grammar and available unit pool. They cost781.732 and337.630fewer fit bits than
the found models, and retain the gold oracle's4/21transfer errors. Separate
production and independent scalar inference agree within4.55e-13. The saved
[oracle diagnostics](../../results/BLIND-CHANNEL-DEV-001/oracle_diagnostics.json)
record the exact replacement rule, scores and metrics so the construction can
be reproduced from the pinned answer/channel and unit-pool artifacts.

This establishes a substantial optimization/initialization gap on these cases.
It does not prove that gold is globally optimal, that a sequence of improving
single moves reaches it, or that the cipher is uniquely identifiable. The B
oracle still makes errors, so source/decoding ambiguity also remains. B's much
too-long found readings further motivate separate length and segmentation
interventions. Increasing compute without changing the search is not yet a
justified solution to those barriers.

## Resources, validation and preserved artifacts

Source/extraction/protocol7bc3f16, selectedsource/data5a5f155 and selectedmodels
4b84ab0 were each pushed and remotely verified before their dependent stages.
Eight searches used two workers,400.881seconds campaign wall time and755.448
recorded processCPU seconds before final writes; zero paid cost. A completed
all4,000proposals/four starts per case. B reached its120second limit after
1,166–1,688proposals and two of four requested starts. Neither is exhaustive
search or a global certificate. The two largest measured process RSS peaks sum
470,007,808bytes; this is not host-wide RAM measurement.

The independent trace validator checks all19,887finished raw/refined score
stages for selection/count/accounting consistency. The independent evaluator
replays72record evaluations, every selected-model code/score, baseline code
and fitted probabilities, literal strings/edit metrics, flags, and all artifact
hashes. Maximum difference1.3642420526593924e-12, [audit pass](../../results/BLIND-CHANNEL-DEV-001/audit.json).
It does not re-evaluate every search proposal numerically. The source-only
auditor separately verifies all7validation scores and552final source probabilities.

Full source regression:1,353passed+23subtests,8skipped. Subsequently added
independent pilot/source/trace checks passed22+3+7targeted tests. All changed
Ruff/compile/diff checks passed; five prior full-tree Ruff findings remain.
Bulk literary text, answers, predictions and traces stay ignored and hashed.
Read the [compact evaluation](../../results/BLIND-CHANNEL-DEV-001/evaluation.json)
and [search coverage](../../results/BLIND-CHANNEL-DEV-001/search_summary.json).

## Next discriminating step

The [results-blind inference review](../research/blind-channel-next-inference-review.md)
derives a sufficient-statistics substitution objective and full253-swap
neighborhood for the23-letter A subfamily. Its implementation and independent
tests used only artificial fixtures. A separately frozen exposed follow-up
will test whether this narrower, systematic optimizer closes A's gap. B still
needs dictionary/segmentation interventions, coordinated support changes and
source-capacity checks. Final authors, stateful family C and Voynich remain
outside this pilot; no decipherment claim follows.
