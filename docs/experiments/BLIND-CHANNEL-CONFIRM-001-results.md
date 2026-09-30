# BLIND-CHANNEL-CONFIRM-001: fresh recovery qualification failed

2026-09-30. Both the registered recovery gate and the separate shuffled-control
screen **FAIL**. All 16 jobs completed normally; no failure was dropped or
retried. The learned dictionaries make 395 edits over 3,584 transfer letters
(11.0212%). One unsupported 224-letter record incurs the registered full-deletion
penalty. The other 15 records still contribute 171 edits; omitting the failure
would not turn this into the required near-exact recovery result.

This is a known-Latin synthetic variable-unit control, **not Voynich text**.
The failed test is useful evidence against treating the earlier two-key
development success as a general solution.

## Frozen sequence and measured results

The method was frozen at `0d3f88568d4820c752c99a6fa8e13aac6ce1dfe9`
before final-text acquisition, with the disclosed HTTPS transport amendment
and original failed acquisition preserved. The prepared corpus was frozen at
`88c75654957c7d19fd6a41d99241dfa3bbba3e3d`; the complete eight-positive/eight-null
panel at `19188c2e2ba8eefe9b9bf348b2b123b6183ab002`. All attempts, process records,
both learned stages and baselines were committed, pushed and remotely verified
at `3ee1a7f5936e47e471e236c48269f77ee755aaac` before the one answer evaluation.
The [registration](BLIND-CHANNEL-CONFIRM-001.md) and its thresholds are unchanged.

Each row below has 448 true transfer letters. All three decoding arms use the
same frozen order3 source. “True key” supplies the generating dictionary; it
does not supply the correct segmentation or force the true reading.

| Fresh key | Stage1 edits | Final edits | Final CER | True-key edits | Correct literal rows /23 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 24 | 24 | 5.3571% | 24 | 20 |
| 2 | 26 | 18 | 4.0179% | 18 | 20 |
| 3 | 21 | 18 | 4.0179% | 18 | 20 |
| 4 | 309 | 21 | 4.6875% | 10 | 18 |
| 5 | 247 | 8 | 1.7857% | 8 | 20 |
| 6 | 320 | 232 | 51.7857% | 8 | 20 |
| 7 | 236 | 29 | 6.4732% | 19 | 19 |
| 8 | 240 | 45 | 10.0446% | 37 | 18 |
| Total /3,584 letters | 1,423 | 395 | 11.0212% | 142 | 155 /184 |

The final dictionaries improve on the frequency initialization for all eight
keys: 395 versus 4,548 edits, a 91.3149% relative reduction. That initializer
uses singleton units for a mostly-digram generator and is structurally weak;
its uncapped CER exceeds 100%. It is not evidence of adequate final recovery.
No final learned transfer record is exact (0/16); the true-key decoder has1/16.

| Prospective recovery condition | Result |
| --- | --- |
| All 16 jobs completed | Pass |
| Every key at most5% transfer CER | Fail; four keys exceed it |
| Eight-key average at most2% | Fail;11.0212% |
| Every key within2percentage points of its true-key decoder | Fail; keys4,6,7 |
| Every key improves on its frequency initializer | Pass |

The separate screen flags seven positives and zero of eight matched shuffles.
Key6 is unflagged because one transfer record has zero support, making its
aggregate transfer likelihood unavailable. The predeclared requirement was
eight positives and zero nulls, so screening also fails. All shuffled fitting
margins are negative (approximately−106 to−136bits), but these are easy
within-record shuffles; semantic or structured-gibberish rejection is not
qualified.

## The failure has several causes

**Reading-model/decision errors remain with the true key.** The true-key arm
makes142edits (3.9621%), already above the2% average target. The generator's
singletons and digrams admit ambiguous segmentations; the fixed Caesar/Virgil
source and MAP decision do not always choose Tacitus's actual text. This is
not a proof of irrecoverable information loss. A stronger source, a different
decision rule or additional evidence must be tested separately. The learned
readings for keys1,2,3,5 exactly match their true-key decoder on all six records,
while still containing plaintext errors.

**Some better keys were missed within the fixed budget.** The true key's fitting
objective is better by90.461bits for key4 and26.393bits for key8. Both final
refinements stopped at the time limit, not a complete local-optimum certificate.
Their learned dictionaries have transfer edit floors12 and9: some true text
cannot be expressed by those dictionaries regardless of language scoring.
This proves a known better candidate exists, not global optimality or that
merely extending time would find it.

**A wrong dictionary can score better.** Key7's learned key beats the generating
key's fitting objective by11.496bits despite29 versus19 transfer edits and an
8-edit transfer dictionary floor. It misassigns x, seen only once in fitting
and four times in transfer, as well as rare k/y/z rows. This selected result is
locally optimal for the tested moves. Better optimization of the same objective
alone is not a dependable cure for that preference.

**One unseen rare letter exposes catastrophic support loss.** Key6 gets20/23
literal rows correct; its wrong k/y/z rows have zero fitting occurrences.
The generating y emits singleton F; the learned y emits C, and no other learned
row emits singleton F. In transfer, y appears once. A source-independent parse
reachability check stops at glyph offset131 of the first391-glyph record,
before an FF prefix. That entire224-letter record has no parse, consistent
with both independent inference and edit-floor checks. The other record has
eight errors, exactly the true-key decoder's reading. The wrong selected key
has a6.977bit better fitting objective than the generating key; finite-sample
compression does not protect unseen assignments or future support.

The last diagnosis motivates a separately declared, answer-informed
[single-row intervention](BLIND-CHANNEL-CONFIRM-001-DIAG-A.md). It does not repair
or reclassify this fresh test. No further fitting or corpus/source tuning has
been performed on this panel.

## Audit, resources and next decision

The [evaluation](../../results/BLIND-CHANNEL-CONFIRM-001/evaluation.json) includes
all cases/arms, uncapped errors, lengths, support, exact-record counts, model/data
scores, dictionary floors and screening flags. Independently authored reference
algorithms replay336record predictions and96floor calculations, with maximum
score difference1.37e-12. Root-authored orchestration is disclosed. The
[descriptive supplement](../../results/BLIND-CHANNEL-CONFIRM-001/descriptive_supplement.json)
checks exact rational gate arithmetic and reports exposure/conditional
uncertainty; fixed-key posterior confidence is not correctness confidence.
Prediction archive SHA256
`0602b5e8ba5d4d9da6b7e32e42cae351d3aa6a2b487ada1cfc2b1507ea242bc5`.

Fitting took3,585.80wallseconds and6,821.40summed childCPU seconds. Evaluation
took18.64wall/17.97CPU seconds. Largest fitting worker RSS was1.046GB; this is
not whole-host simultaneous memory use. All32search archives (266.31MB) and
30frozen source files were checked before evaluation. No paid compute/API use.
Full suite1,848tests plus23subtests passes,8skips; changed Ruff passes and the
same five unrelated pre-existing lint findings remain.

Do not advance this decoder as a qualified manuscript reader. The next method
needs to preserve uncertainty about unobserved key rows, evaluate source quality
with known-key controls, and improve search separately from objective quality.
A future qualification needs a declared unused allocation and controls; these
eight keys and their passages are now exposed. Unknown language, stateful
channels, transcription uncertainty and structured nulls remain unqualified.
No Borg or Voynich plaintext has been recovered by this experiment.
