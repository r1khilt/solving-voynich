# PARAGRAPH-CARRY-001 — controls qualify; transported predictor fails on Voynich

2026-10-10. Carrying the fixed within-line terminal odds onto the next line
reduces prediction quality in both manuscript readings. The registered
paragraph-carry signal is **false**. Stop using this particular transported
predictor as evidence for a manuscript carry mechanism.

The [registration](PARAGRAPH-CARRY-001.md) was published at
`efdabe6b195a9711752ffca8dad224d71e15947b`; exact remote equality preceded
ONE producer and ONE full arithmetic auditor, both terminal0. All26cells
completed, with no retries or changed gates. Actual manuscript input comprises
828matched clean TRAIN lines on76physical leaves;247continuation and33paragraph
junctions across60event-bearing leaves. No validation/final-test text was parsed.

Gain is transported prediction minus reset prediction in bits per initial unit.
Intervals are conditional leaf bootstrap intervals without refitting models.

| Reading | Continuation gain (interval) | Paragraph gain (interval) | Permutation p | Classification |
|---|---|---|---:|---|
| ZL | −.08061 [−.16564,.00446] | −.35509 [−.74513,−.10638] | .25743 | line-reset compatible |
| GC | −.15016 [−.25334,−.05064] | −.62341 [−1.35357,−.03677] | .78218 | line-reset compatible |

The continuation-minus-paragraph differences are .27448/.47325bits, but their
intervals include zero and neither continuation gain is positive. The larger
negative paragraph penalty is not evidence that useful coding state carries
through ordinary lines. Matched nulls have141/136effectively mutable events
of280; sufficient mobility does not rescue the failed prediction.

All12planted carry/reset/unbroken-carry documents classified correctly. All8
copy-mutate/IID controls avoided paragraph-carry classification. All4shared
paragraph-common-cause documents **did** classify as paragraph carry, the
preregistered nonidentification witness: a shared paragraph style can create
this apparent observational pattern without a terminal-driven state update.
Thus even a positive manuscript pattern would not have identified a causal cipher.

Producer:2.452717958wall/2.354928CPU seconds,75,841,536bytes RSS,
337,949ignored archive bytes,$0. Audit:7.217448250wall/7.024126CPU seconds,
85,999,616bytes RSS,$0. All22,640events and2,600permutations were independently
recomputed with training distributions, junctions, manifests, scores and gates;
all bindings/resources PASS. Parsers, synthetic generator, folds, bootstrap and
gate helpers remain shared. This is alternate arithmetic, not independent
paleography or cryptanalysis.

[Producer receipt](../../results/PARAGRAPH-CARRY-001/result.json) and
[audit](../../results/PARAGRAPH-CARRY-001/audit.json) retain all cells and nulls.
The result rejects the frozen observable odds-transport predictor on this
development split. It does not prove a line-reset cipher, rule out natural
language or state-dependent channels, or recover any plaintext. Earlier
end-to-start dependence results concern a different conditional prediction;
they are not erased by this negative. Sparse clean paragraph support and
serial/topic exchangeability remain limits. No larger architecture is justified
solely by this failed transport assumption.
