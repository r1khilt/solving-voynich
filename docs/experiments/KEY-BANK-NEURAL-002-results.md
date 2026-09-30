# KEY-BANK-NEURAL-002: both neural sources improve the exposed candidate ranking

Both unchanged trained sources pass the predeclared development criterion.
Errors decrease from the statistical43/3,584 baseline to38 and29; all16records
remain supported. This is an exposed candidate-ranking result, with no fresh
unknown-key or historical decipherment claim.

| Source | Edits /3,584 | Error rate | Reduction vs43 | Exact records /16 |
| --- | ---: | ---: | ---: | ---: |
| Statistical READ-002 | 43 | 1.1998% | — | 6 |
| Neural seed31103 | 38 | 1.0603% | 11.63% | 6 |
| Neural seed31109 | 29 | 0.8092% | 32.56% | 9 |

| Original key | Statistical edits | Neural31103 | Neural31109 |
| --- | ---: | ---: | ---: |
| 1 | 8 | 6 | 6 |
| 2 | 6 | 4 | 2 |
| 3 | 7 | 7 | 3 |
| 4 | 1 | 2 | 0 |
| 5 | 2 | 4 | 4 |
| 6 | 0 | 3 | 2 |
| 7 | 2 | 0 | 0 |
| 8 | 17 | 12 | 12 |

Both models support every record and no key exceeds5%CER. Each model improves
some cases and worsens others; neither is selected retrospectively as the sole
result. The smaller observed gain is5edits; this experiment supplies no population
confidence interval or proof of statistical significance over unseen manuscripts.

## What changed and what the result resolves

All104,290original complete candidate tuples and statistical fitting-key weights
stay fixed. Only the transfer-source scores change. One compatible key must
encode both complete records. Both7.405M-parameter models were trained/selected
before this branch. Thus stronger source scores can improve decisions even
without new key search or training. The source-specific unseen bound is unavailable:
these are optima over the saved candidates, with no full neural evidence/MAP.

The separate fixed8×270 implementation resolves the recorded resource obstruction:
all16case/model outputs complete under the original2GiB driver /4GiB host caps.
NEURAL-001's earlier failed attempt and missing-output penalties remain intact.
The engineering benchmark's initial variable64 pass did not explain that earlier
failure; successful real-candidate execution is the observed repair.

## Post-freeze candidate diagnosis

After prediction publication and gold evaluation, the exact true whole-text tuple
is found in the restricted union for keys4–7 only. Keys1–3and8 have true text
supported by some keys in the old bank, but that exact tuple was never included
in its statistical top8-per-key proposals. Bank support and proposal inclusion
are different properties. A neural ranker cannot select a missing proposal.

Those four proposal-missing cases contain29 of38errors for31103 and23 of29for31109.
These counts are observed errors, **not proven minimum edit floors**. No additional
neural gold scoring, gold insertion, or wider candidate search was performed.

Where the gold tuple is available,31103 prefers a wrong tuple on keys4/5/6 and
31109 on5/6. Key5's neural source advantages for the wrong reading are6.299/9.147
nats, outweighing its2.923nat key-mass disadvantage. Both neural models lose the
previously exact key6. These are objective preferences within the restricted set,
not failures to find its maximizer. The seed disagreement onkey4 also limits any
claim of a stable mechanism. Detailed existing-score decomposition is in
`gold-candidate-diagnostic.json`.

This motivates two separate next questions: neural-source proposals for the
missing readings, and causal context analysis of source preferences on matched
correct/incorrect candidates. Neither has been implemented or tested here.
The current fresh CONFIRM-002 pipeline retains its originally frozen statistical
source and criteria.

## Staging and validation

Source651588d625c923ba0cc4492d49e071160f80fd4c was pushed/remoteverified before
one prediction; completed predictionsd76e9106d42bc84181ae157852e984794f9eaf85
were pushed/remoteverified before one evaluation. No selective retry or retuning.
All182source/input bindings and output hashes pass.208,580candidate-score arithmetic
replays agree exactly,58,754selected literal key checks pass, and all19,822memory
trace rows respect the constant shape and prescribed guards. Original statistical
winners were reproduced. Sampled MPS/CPUfloat64 deltas≤8.034e-5nats/record; winning
paths separately pass the1e-7full-forward/stepwise gate. These samples do not
certify every possible floating-point ordering.

Prediction461.617558wall/114.627801hostCPU seconds,37,032,100letters,158,528distinct
case-record scores across both sources,1,259,061,248bytes sampleddriver peak and
1,034,649,600hostpeak;0paid spend. The fixed evaluation independently checks32
record edit distances and retains the full denominator. Source/tests preceded
prediction: full2,258tests+23subtestsPASS/13skips, targeted6PASS/1sandboxMPSskip,
separate actualGPUadapter1PASS. No source changed during the attempt.
