# A bounded terminal-effect transport test, with an identification obstruction

2026-10-10 PDT. Own proposed assay and mathematical interpretation, not a
manuscript result. This note supports [PARAGRAPH-CARRY-001](../experiments/PARAGRAPH-CARRY-001.md).

The project's strongest replicated textual constraint is the end-to-next-start
effect in [EXP-0039](../experiments/EXP-0039-results.md): both independently
read GC-v101 and ZL show it under train-to-exposed-validation scoring. It is an
old Voynich observation, not our discovery. [EXP-0040](../experiments/EXP-0040-results.md)
criticizes particular frozen pseudo-text generators; [EXP-0041](../experiments/EXP-0041-results.md)
calibrates four works and simple reversible channels. Neither supplies a
historical explanation. [EXP-0042](../experiments/EXP-0042-results.md) fails its
stronger cross-transcription context gate. More feature tuning on those nine
validation leaves cannot create independent evidence.

Reddy and Knight's [2011 primary report](https://aclanthology.org/W11-1511/)
documents structural phenomena without recovering plaintext. The
[IVTFF maintainer's explanation](https://www.voynich.nu/extra/sp_transcr.html)
distinguishes standard P0 lines, other text layouts, paragraph markers and hand
metadata; standardized line conventions are still partly editorial judgments.
The [IVTFF primary format specification](https://www.voynich.nu/software/ivtt/IVTFF_format.pdf)
§6.4 Table8 and §7.2 distinguish the first/unrelated `@` position from `+`
below and `*` below at the left margin, referring to the preceding numbered
locus. We fit clean P0 loci in these three layouts and predict across only
consecutive `+`/`*` successors. A pre-freeze availability check found and fixed
an initial `@`-only schema error, which had accidentally discarded continuations.
[Rozanova and Temerev2026 §§2.4–2.5/2.8](https://arxiv.org/html/2608.17096v1)
motivate clustered uncertainty and separation of paragraph starts. Their cipher
calibration omits several state-dependent operations; absence of those controls
does not prove such operations exist in Voynich. The new assay uses no reported
paper thresholds or their whole-manuscript text as fresh project confirmation.

## Why simply applying a within-line table is unsound

Let M(i) be a within-line marginal initial-unit law and E(i|l) its terminal-
conditioned law. A line-start role has a different marginal B(i). Substituting E
directly for B would confound transport with line-initial preferences. Our own
fixed assumption transports the ratio E/M onto B and normalizes the result.
That construction is a valid categorical distribution because all smoothed
probabilities are positive, but it is an assumption about stable odds, not a
theorem about manuscript production. Its normalization is essential.

In particular it does not constitute a complete normalized cipher/source
model. Interiors, token lengths and terminal units are conditioning data;
scores concern only held-out initials. [Knight et al.2006](https://aclanthology.org/P06-2065/)
is a prior explicit source/channel inference framework; the present assay is
more limited and produces no recovered language, reusable decoding key or
plaintext. One observed terminal unit acts as the bounded predictor state;
there is no discovered recurrent hidden state, world model or causal circuit.

## A common-cause witness prevents causal identification

Own counterexample suggested during independent preparation review: choose a
latent paragraph style Z. Draw each token's initial and terminal independently
given Z, with both biased toward style-specific symbols. Consecutive tokens
share Z, so observing the preceding terminal helps predict the next initial.
The relationship persists across line breaks inside that paragraph and drops
when a new paragraph gets a new Z. There is no previous-terminal-driven update.

Thus within-line dependence, continuation transfer and apparent paragraph reset
do not identify a carry cipher. The experiment contains a preregistered
`paragraph_common_cause` witness rather than claiming a couple of IID/copy
negatives exhaust nonsemantic alternatives. Its paragraph-carry classification
is permitted. Same-leaf/depth/length matched permutations preserve some layout
marginals but mix paragraph styles and cannot remove this witness. A later
causal proposal must condition/test paragraph-level causes or make a decoder
prediction these observationally similar processes disagree about.
Permutation ranks assume terminal exchangeability within the frozen strata;
serial dependence and topic/style mixtures can violate that assumption, so
the rank is not guaranteed calibrated for arbitrary serial or topic nulls.

## What the experiment can change

If the fixed odds fail to transfer in both readings after calibrated controls,
stop treating this simple paragraph-carry predictor as manuscript-facing
evidence and avoid building a larger stateful architecture around it. If they
transfer, record the precise conditional prediction as a constraint for future
channel models; require disambiguation of common causes and an executable
decoder before making historical claims. Even a positive result leaves natural
language, copy procedures, topic mixtures and explicit ciphers live.

Training leaves are previously used development material. Five leaf folds keep
direct fitting separate from each prediction but do not erase researcher
adaptation, neighboring-leaf dependence, scribe/section confounding or shared
editorial conventions. Two glyph readings of one manuscript are robustness
views, not two independent manuscripts. GC's sparse strata and larger alphabet
may have less power than the eight-symbol homogeneous synthetic controls.
Bootstrap intervals condition on frozen cross-fitted models and do not refit
their training distributions.

The surviving reserved physical leaves and external image/semantic anchors
remain separate later evidence. No final-test/old validation text is opened,
no external image is identified and no paid training is used here.
