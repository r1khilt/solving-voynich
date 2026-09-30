# More source context repairs readings and reveals a dictionary limit

2026-09-29. The [registered fixed-dictionary diagnostic](BLIND-CHANNEL-DEV-004.md)
reduced transfer edits from **4 to 0** in B1 and **32 to 22** in B2, each over
448 true letters, by changing only the source model. The source was chosen
using Caesar/Virgil, without cipher answers. Independent replay passes.
This is exposed synthetic development under restricted assumptions, not a
Voynich reading, a fresh-key test, or new dictionary recovery.

| Fixed dictionary | Old source transfer edits | Selected order3 source edits | Exact transfer records | Minimum possible edits under dictionary |
| --- | ---: | ---: | ---: | ---: |
| B1 learned | 4 | **0** | 2/2 | 0 |
| B1 generating oracle | 4 | **0** | 2/2 | 0 |
| B2 learned | 32 | **22** | 0/2 | **12** |
| B2 generating oracle | 21 | **8** | 1/2 | 0 |

The minimum is an evaluator-only exact lower bound across every reading that
encodes to the observed ciphertext. It is not a reading produced without the
answer. B2's learned dictionary cannot give exact truth even with an ideal
decision rule: at least 12 transfer edits are unavoidable. Source quality also
matters, since the true dictionary improves from 21 to 8 errors with more
context. Both dictionary support and source ranking remain relevant.

## Source choice and sensitivity

All 19 configurations fit 50,000 Caesar letters and were scored on 50,000 Virgil
letters, preserving original body boundaries. Per-order source-only winners:

| Maximum preceding letters | Selected tau | Validation bits/letter |
| --- | ---: | ---: |
| 0 | 1 (unused) | 3.985297053 |
| 1 | 64 | 3.616861829 |
| 2 | 256 | 3.491420368 |
| 3, primary | 256 | **3.452566317** |

Each selected configuration was refit on both source authors. The order1/tau64
table exactly reproduces the previous source. Virgil has already served as
development data; these validation scores are not fresh generalization claims.

| Fixed dictionary | Order0 transfer edits | Order1 | Order2 | Order3 | Order3 fit edits /896 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 learned | 16 | 4 | 0 | 0 | 13 |
| B1 oracle | 16 | 4 | 0 | 0 | 13 |
| B2 learned | 83 | 32 | 26 | 22 | 70 |
| B2 oracle | 73 | 21 | 14 | 8 | 34 |

Order3 was selected from source-only validation, not this accuracy table.
B2 order2 has fewer fit errors than order3 (68 versus 70), illustrating why
neither source context nor likelihood should be equated with perfect reading.
The learned B2 fit support floor is 20 edits; all other positive/oracle fit
floors are zero. B1's exact transfer does not mean its entire fit text or key
is correct.

## The objective now supports repairing B2

Under the old source, the learned B2 dictionary beats the generating one by
30.779 fit bits despite a worse reading. Under order3, the generating one
instead wins by **8.724528233 bits**:

| Source | Learned B2 fit bits | Generating B2 fit bits | Better dictionary among these two |
| --- | ---: | ---: | --- |
| Old/order1 | 3247.839272 | 3278.618275 | Learned, by 30.779004 |
| Order2 | 3113.463163 | 3120.926764 | Learned, by 7.463601 |
| Order3 | 3071.025033 | 3062.300504 | Generating, by 8.724528 |

These totals include the unchanged one-bit channel/baseline family selector;
DEV003's search objective omitted that shared bit. Pairwise differences are
identical. They are conditional on the declared source, not language evidence
or an uncharged target-based choice of source model.

This is a concrete improved-objective direction, not proof that gold is the
global optimum or that local refinement can find it. B1 still prefers its
shorter learned dictionary by 6.609863459 bits under order3, while both give
exact transfer. Literal generating-key recovery is therefore not the same as
correct observed decoding.

## The alternate decision rule did not help at its registered budget

The old-source sampler drew 32 candidate readings and 256 independent
reference readings for each of 36 records/arms. MAP was included as a candidate.
The minimum sampled raw-edit-risk decision selected MAP **on all 36 records**,
including shuffled controls. Thus its positive accuracy is unchanged: 4/32
learned transfer edits and 4/21 oracle transfer edits.

This is a negative result for this finite candidate/reference budget. It does
not prove MAP globally minimizes expected edit loss, that larger candidate
sets cannot help, or that a different decision rule under order3 would fail.
No seed was rerolled or budget increased after seeing the result. Nulls have
no plaintext accuracy or semantic acceptance threshold.

## Verification, provenance and resources

Source/protocol checkpoint `de589d792bbdf9c86a8b230a18e5343bcaafda3d`, source
selection `44513f2f80d75751aee0a80bba674df84c3f3eff`, and predictions
`41708414cd1af85e25381b9127e3817adf0ee13e` were each pushed and remotely verified
before their dependent stages. Exact commands are NB-212–215. All input hashes,
source-only choices, full prediction/bank hashes and resource measurements are
in [source selection](../../results/BLIND-CHANNEL-DEV-004/source_selection.json),
[prediction manifest](../../results/BLIND-CHANNEL-DEV-004/prediction_manifest.json),
[evaluation](../../results/BLIND-CHANNEL-DEV-004/evaluation.json), and
[independent audit](../../results/BLIND-CHANNEL-DEV-004/independent_audit.json).

The independent implementation replays all19 source-selection scores, every
selected probability, 180 marginal/MAP predictions, 36 decisions, 10,368 saved
draws' support, 275,968 weighted candidate/reference edit pairs (243,622 distinct
pairs calculated), all literal model costs/metrics, and 24 dictionary floors.
Maximum numerical difference is **3.524291969370097e-12**. It obtains the same
literal MAP on175/180 records; the other five are different equal-scoring
representatives, with each stored reading's score and reencoding checked.
Posterior banks were not regenerated numerically; their hashes, support, seeds
and downstream risks were checked, with sampling probabilities qualified on
independent exhaustive fixtures.

| Stage | Wall seconds | CPU seconds | Peak process RSS bytes |
| --- | ---: | ---: | ---: |
| Source selection | 3.078 | 2.738 | 119,308,288 |
| Predictions | 48.691 | 48.180 | 116,719,616 |
| Answer evaluation | 0.855 | 0.468 | 49,938,432 |
| Independent replay | 80.123 | 79.851 | 142,508,032 |

Stages were sequential, single numerical thread, no paid compute/API use.
Total measured CPU is131.237 seconds, well inside the45minute bound. The source
archive is2,576,245bytes and prediction archive639,938bytes, ignored and hashed.
Before these stages, the full suite passed1,703tests+23subtests with8skips;
all163 new focused tests passed. Five unrelated pre-existing Ruff findings
remain; all changed files pass. Preparation interruptions and corrected test
arithmetic/import invocation errors are preserved inNB-212.

Next: test whether a bounded ciphertext-only refinement under the already
selected source repairs the dictionary. Every source-letter row remains
eligible; do not hand-edit exposed wrong letters. Fresh keys/authors and the
stateful/historical problems remain separate required qualifications before a
manuscript decipherment claim.
