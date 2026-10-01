# JOINT-KEY-ORDER-001: learned suffix-order dependence is not supported

2026-10-01. All384 fixed cells completed in the single CPUfloat64 run. The
registered exploratory order diagnostic is **NOT_SUPPORTED**. This is a negative
finding about the exposed first completed small model, not a decipherment or
proof that the cipher contains no order information.

In plain English: we rearranged most of the encrypted text while keeping its
symbol counts and naming unchanged. The trained model barely preferred the
original text on average and still recovered very few key entries. This gives
us little reason to believe this particular model learned strong decipherment
skills from sequence order. It does not settle what the larger model will learn.

| Checkpoint | Whole-unit shuffle loss change, nats/row | Descriptive95% interval | Used-key matches: original→shuffled |
| --- | ---: | --- | --- |
| Initial0 | +0.00006506 | [-0.00009581,+0.00021697] | 26→24/1248 |
| Selected20000 | -0.00001135 | [-0.00106224,+0.00102370] | 49→47/1248 |

Loss change is control minus original: a positive change would favor original
order. The selected effect is below the frozen.01 threshold, its interval spans
zero, and it declined by0.00007640 from initialization. All three diagnostic
clauses fail. Original selected loss independently replays as3.40778291 nats/row.

The individual-glyph shuffle gives selected mean+0.00019307 with interval
[-0.00064140,+0.00110684] and49→49 used matches. It can destroy codeword support,
so that contrast alone would not identify source-language reasoning anyway.
Complete greedy keys are0/64 in every condition at both checkpoints.

Both shuffles preserve all counts, lengths and first-occurrence names through
the same whole-unit frozen prefixes. Across the128 original records,31,454 of
33,984 glyph positions were mutable (92.56%). Whole-unit/glyph shuffles actually
changed25,085/25,103 positions (73.81%/73.87%); no record was unchanged. Maximum
frozen prefix was73 glyphs. Gold source segmentation was used to construct the
controls, never passed to the network. Counterfactual metadata explicitly
distinguishes transformed inputs from original corpus windows.

This does not mean outputs never respond: whole-unit shuffling changes the
selected greedy key in29/64 cases, and glyph shuffling in27/64. Individual
whole-unit loss contrasts have mean absolute0.00285645 and maximum0.01569640
nats/row. Nevertheless their average does not show learned preference for the
original order, and reading/key recovery remains weak. Ordered prefixes survive;
one shuffle per case, one seed, selection exposure and off-distribution controls
limit the conclusion. Counts/prior/prefix-based prediction is an explanation
consistent with the result, not an identified exclusive mechanism.

Source/protocol/test commit`6bde6725acf689761d92abe9469c52caa188afaf` was published
and exactly verified remotely before the one run22767, terminal0. Stage cost:
98.357864 wall/133.214371 CPU seconds,419,577,856 peakRSS bytes, CPUthreads2/BLAS1,
zero GPU inference, training updates, paid APIs or new holdout use. Every original
reference passes: true whole-logq max difference6.47923e-6, saved-greedy max
difference5.85388e-6, maximum old-choice CPU deficit0; CPU/MPS greedy keys agree.

One metadata auditor17423, terminal0, regenerated all256 transformed cases,
checked the full ordered384-cell grid, all metrics and bootstrap arithmetic,
input/weight hashes and the511,597-byte ignored record trace. It took.137987
stage-wall/.137876CPU seconds with268,615,680peakRSS bytes. Mechanical PASS does
not change the negative diagnostic. It did not repeat model inference and is
same-author verification, not independent scientific confirmation.

The full four-fit training ledger audit remains pending, so this first-fit study
is provisional with respect to that future integrity check. The original
training source/inputs/schedule were not altered. The95M first fit remains live.
The [next-objective derivation](../research/key-inverse-next-objectives-2026-10-01.md)
explains why a structure-learning auxiliary head or learned actions checked by
the exact evaluator are candidate improvements. Neither is implemented or
trained by this result. No causal neuron, hidden meaning or historical reading
is established.
