# Known-reading probability isolates a severe inference failure

2026-10-01. This exploratory diagnostic was registered after the particle
outcome, published at `56d746027e27249921457faede19ee874ef7236a`, and exactremote
verified before one invocation. It does not change the failed recovery outcome.

**In all16varied-source cells, the known generating reading has much higher
probability than the selected wrong reading under the same declared model.**
This comparison integrates all compatible used dictionaries for both readings.
The current inference is missing an available better answer; the model does
not prefer its selected answer in these cases.

## Exact candidate counts and probability comparisons

All36source-supplied word-equation calls completed without caps:4generating
readings and32selected readings,5,089total recorded nodes. Each generating
reading has one compatible used dictionary; selected readings have1–77.
These are bounded oracle counts checked against small independent enumeration,
not an independent real-size completeness proof. Every returned dictionary
re-encodes both literal records and binds exactly the used source rows.

Reading mass is normalized source/geometric-length mass multiplied by S/42^M,
where S is the compatible used-key count and M is the number of source rows
used. Unobserved key values integrate out. No target source length is supplied
to the original particle decoder. Literal glyph labels use no canonical factor.

Ranges below cover the8paired schedule/population/seed cells for each fixture.
Positive differences favor the generating answer; negative differences favor
the sampled output. Units are natural logarithms.

| Fixture | Generating reading minus selected reading | Generating leaf minus best visited leaf | Generating leaf minus evidence estimate |
| --- | ---: | ---: | ---: |
| Repetitive64 | −834.49 to−763.18 | −846.70 to−774.32 | −971.57 to−954.42 |
| Repetitive224 | −2816.22 to−2727.60 | −2832.83 to−2727.60 | −3514.33 to−3499.50 |
| Markov64 | +244.16 to+339.17 | +242.89 to+339.17 | +156.03 to+224.33 |
| Markov224 | +1050.55 to+1381.43 | +1050.55 to+1381.43 | +831.06 to+1208.44 |

For the varied cases, even the smallest generating-to-selected reading ratio
exceeds10^106 under this model. This is a comparison between particular
available readings, not a claim that the generating reading is the global
posterior mode or uniquely identifiable. Unvisited alternative readings remain.

Repetitive controls show a different failure: their forced repeated-letter
source is extraordinarily unlikely under the Latin source model. That prior
prefers the selected alternatives. Their recovery failures do not demonstrate
the same inference error as the Markov cases. Artificial forcing, prior
misspecification and ambiguity must remain distinct.

## The finite joint bank has negligible target coverage

No generating reading or generating used-key leaf appears anywhere in any
of the32stored banks. For a disjoint generating leaf with mass Lg and total
unique visited leaf mass B, posterior joint bank probability is at most
`B/(B+Lg)`. Thus any distribution supported there has joint total variation
distance at least `Lg/(B+Lg)` from the declared source/used-key posterior.

The Markov64log coverage upper bounds range−338.89 to−241.71; Markov224
range−1381.43 to−1050.30. These indicate an extremely poor approximation of
the **joint** target. They do not bound reading-marginal coverage: keys not
in this bank may add probability to already visited readings. Rounded TV
values near1 are not exact equality or floating interval certificates.

Each sampled evidence estimate in the varied cases is below the mass of
one known compatible leaf, which alone lower-bounds total evidence. The
smallest underestimate is at least about10^67.76. This is a realized
estimation failure, fully compatible with the theoretical unbiased normalizer
identity across all random runs. An unbiased estimator can be almost always
tiny and rely on rare large outcomes; log estimates and finite posterior
samples are not unbiased. Two seeds do not characterize that tail.

## Validation, provenance and resources

One diagnostic session19672 terminal0:1.429318stage wall/1.416414CPU seconds,
413,007,872peak resident bytes. Bounds600wall/500absoluteCPU/2GiB/cache2048
held. Zero population reruns, retuning, GPU/neural work, training, paid APIs,
new corpus or holdout. Statistical source prediction and supplied-source
counting occur. The lazy source independently matches all32best visited
leaf scores from the dense source within1e-9.

Publication review checks all36literal count outputs, input/result/trace hashes,
every source/first-binding/count formula, all32gap calculations, bank
disjointness and known-reading absence. No repeated source scoring/counting.
Initial review failed early because a relative path was passed to an
absolute-root artifact helper; correcting that review path fixed it, with
no diagnostic or inference retry. Same author; no independent completeness
or scientific-confirmation claim.

The result has18,938bytes, SHA256
`010afe3dd2b08da91bffc75bb9eaecaaa761e6b57e6ba2ae2609755489c35106`.
Ignored36count trace3,467bytes SHA256
`4334544f58086280366bbbe3d854058a881880a92eeb68bf640d97ce4ff09902`.
Compact comparisons and publication review are in
`results/SOURCE-PARTICLE-SYSTEMS-001-DIAG-A/`.
Three new rational checks passed; the frozen solver's full2,567tests plus23
subtests and31focused tests remain unchanged/passing.

## Implication for the next method

More immediate-constraint particles are insufficient in this tested range.
Future constraints must influence early assignments, or inference must revisit
those assignments after later evidence arrives. Candidate remedies include
properly corrected future-completion guidance and source/key trajectory moves
that can replace early assumptions. This diagnostic supports investigating them;
it has not measured either remedy or established that resampling alone causes
the failure. No new guide, rejuvenation kernel or architecture is launched.

The95Mcampaign continues unchanged. These synthetic results diagnose an inference
problem in one explicit family. They establish no Voynich language, historical
channel, mechanism or decipherment.
