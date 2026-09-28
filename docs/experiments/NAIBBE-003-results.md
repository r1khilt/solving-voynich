# NAIBBE-003 results: a source-only repair passes the final assisted control

Evaluated 2026-09-27 under the [frozen registration](NAIBBE-003.md).
**Registered primary decision: PASS.** This is a restricted cipher-recovery
control, with known table/role structure. No Voynich decipherment or source
language identification is established. This closes the current publication's
rare-letter tuning sequence; no additional repair will be selected on it.

## Main recovery result

Every row below uses the same final, previously unscored Pliny block: published
tokens[26624,34764),8,140ciphertext chunks and12,296plaintext characters.

| Frozen decoder | Correct key entries | Character edits | Character error rate | Exact chunks |
| --- | ---: | ---: | ---: | ---: |
| Source-selected Latin prior | 136/138 | 7 | 0.05693% | 8,136/8,140 |
| Frozen NAIBBE-002 Latin key/old prior | 130/138 | 12 | 0.09759% | 8,131/8,140 |
| Source-selected English prior | 132/138 | 28 | 0.22772% | 8,123/8,140 |
| Correct key, selected Latin prior | 138/138 | 6 | 0.04880% | 8,137/8,140 |
| Correct key, old Latin prior | 138/138 | 6 | 0.04880% | 8,137/8,140 |
| Correct key, selected English prior | 138/138 | 22 | 0.17892% | 8,129/8,140 |

The repair improves six key assignments and removes five edits relative to the
old frozen decoder on exactly the same text. Both Latin priors make the same
number of errors with the correct key, so the observed net improvement concerns
key recovery rather than a lower final-block oracle error count. It is a modest
absolute improvement on an already nearly solved, assisted control, not a new
ability to discover the encoding family.

All four registered criteria pass: CER<=2%; within0.5percentage points of the
selected prior's correct-key oracle; macro key accuracy>=95% (observed98.5507%);
and uniquely assigned-class-weighted key accuracy>=99% (observed99.99169%). The
last denominator is12,036source positions:260positions with unresolved class
identity are excluded because the publication lacks a table-draw trace.
The previous experiment's registered FAIL remains unchanged.

Two remaining key errors are one y/z swap in one table. Neither class has
uniquely assigned fit evidence, and the selected and true keys have exactly the
same fit Viterbi objective. In transfer one of these classes has one uniquely
assigned occurrence, and the learned decoder incurs one edit beyond its oracle.
This is a fit-objective tie, not proof of absolute unidentifiability: alternative
candidate paths and a properly marginalized channel model may contain other
information. We do not retune or resolve that tie using the now-exposed answers.

Latin selects197/200ambiguous transfer chunks correctly; English189/200.
On fit, selected Latin has4edits/12,483characters, matching its correct-key
oracle; old Latin has9 versus oracle6. English has15 versus oracle12. English
still recovers most of the key, so wrong-language performance does not establish
that this setup can identify an unknown source language. Source-search
objectives across priors are not marginal evidence for a language or channel.

## How the prior was chosen

The diagnosed002 flaw was a uniform fallback after unseen rare histories.
Before any new cipher fitting, compare the unchanged legacy prior with six
recursive Dirichlet concentrations and four interpolated absolute discounts,
using only the separately partitioned Caesar reference corpora. Both languages
select recursive Dirichlet concentration16 by development bits/character.
Refit that selected configuration on each entire original317,326-character
reference corpus before cipher fitting. No hyperparameter was selected by new
cipher key accuracy or transfer recovery.

| Source-only check (49,326characters each) | Old bits/character | Selected bits/character | Improvement, block95% interval |
| --- | ---: | ---: | --- |
| Latin | 2.646607 | 2.616286 | 0.030320 [0.025443,0.035155] |
| English | 2.541903 | 2.514385 | 0.027518 [0.020875,0.033394] |

The previously used source works were repartitioned; these are not new authors
or independent-document uncertainty estimates. The later cipher block is from
the same published Pliny sample as earlier controls. The choice to investigate
backoff was motivated by exposed002errors; this is prospective evaluation of
an adaptive repair, not a wholly independent research hypothesis.

## Freeze chain, verification and resources

- Source models, protocol, data and independent source auditor: remotely
  verified`9cba3ff7a83333c850de52353a5aed2386d4a07f` before source scores.
- Source-selected configurations and independent audit: remotely
  verified`6cec2ae82c0eaa3aef8fe090a1511fc28b85e6da` before cipher fitting.
- Learned keys and independent decoder auditor: remotely
  verified`5589b0ad0cccc684fcb991877947bbfe9dc749a1` before final answers.

The key-publication action initially hit an automatic approval-review timeout;
the permitted retry succeeded, and remote verification preceded evaluation.
Both fits completed all56cycles under the1,200second per-arm cap: Latin680.938s,
English758.086s, run concurrently. Sum of arm runtimes1,439.024s is not measured
CPU consumption. Source selection0.507s; source audit11.146s; decoder audit9.080s.
No paid APIs or external compute were used.

Independent source replay validates all22candidates and279,840conditional rows,
with maximum numerical difference2.84e-14. Independent final replay reconstructs
all16,332exact local-class lattices from original tables and verifies97,992
learned/oracle emitted chunks, all scores, edit distances, key metrics, gates
and frozen hashes. Maximum objective difference1.46e-10. It does not prove
global key-search optimality or correct historical assumptions.

Compact evidence:

- [Source selection](../../results/NAIBBE-003/source_selection.json), SHA256
  `4dec9b63f39ce91e396f7c2f9bcfc078fcd4c0aaccc8f99ccb3ba47080daa676`.
- [Source audit](../../results/NAIBBE-003/source_audit.json), SHA256
  `6a37e0e713bd0c5fb00e5e65ceb07dd82b98afd26babf81a90e1956a051fe6a8`.
- [Evaluation](../../results/NAIBBE-003/evaluation.json), SHA256
  `f442c77c2e4b0a25c18f34a25360c635d24921521b16763e4b35100856c8b5c7`.
- [Independent final audit](../../results/NAIBBE-003/audit.json), SHA256
  `268bea970c7ec7f732e20f22be66445e114a16786b528b81889c0a93f0d5df5a`.

Full choices, bulk inputs and answers remain ignored. Before fitting, the full
suite passed1,235tests plus23subtests with8skips. The subsequently added
independent decoder auditor passed8targeted tests and its exhaustive toy checks.
Final integrated regression results are recorded in the notebook checkpoint.

## What changes next

The repair supports a concrete diagnosis: better optimization was not the
appropriate response to the earlier objective's preference for rare wrong
letters. A separately calibrated prior improved actual recovery. But all126
common-letter mappings were already correct, and the supplied candidate lattice,
table groups and role linkage remain major assistance unavailable for Voynich.

The next program is [blind channel recovery](../research/blind-channel-recovery-design.md):
infer units and a reusable mapping under a normalized ciphertext-generating
model, test across source authors and unseen channel families, and reject
matched nonlanguage controls. Historical Borg is a complementary authenticity
check with unresolved transcription parsing. Neither is yet a successful
decipherment result.
