# TEACH-0016 fresh cross-order clean-competence screen

**Status:** prospective implementation registration. The TEACH-0016 visible
suite was source-bound audited, but no trained TEACH-0016 prediction or
intervention has run. This instantiates the clean-competence requirement of
[`TEACH-0016-cross-order-registration.md`](TEACH-0016-cross-order-registration.md)
without changing its causal thresholds.

Only named arm/seed pairs with passing TEACH-0014 v3 artifact/replay,
TEACH-0015 fresh-screen audit/replay, and TEACH-0015 finite
`PORTABLE-INTERMEDIATE-STATE-SUPPORTED` result after its full no-model and
sampled checkpoint replay may be screened. Admit a named arm to cross-order
causal analysis only if **both seeds** pass this fresh screen. If no prior
portable arm exists, record `NOT ENTERED: PRIOR PORTABLE STATE` without a
checkpoint call.

For the frozen TEACH-0016 seeds86111/86121, score every66-cell group:
8,448 episodes per split, with source order, marker and distractor variants
kept together by logical group. On each arm/seed/split, report all composed
answers over6,144 cells, exact three-recipient triples over2,048 sets,
marked/free exact pairs over3,072 pairs, and first-hop/direct/copy over768
items each. The confirmation gate is composed≥90%, triples≥80%, pairs≥80%,
first-hop and direct≥95%, copy≥98% **in each seed**. Discovery is descriptive.
The source scorer independently reconstructs visible oracles and all
denominators and reports4,000-resample logical-group intervals; no examples
are dropped because a model answer is wrong. Native predictions later used in
TEACH-0016 causal assays must match this screen by render ID.

Use batch32, save every ordered render ID/prediction plus full2,064 logits
at fixed episode indices0,4224,8447 per split/arm/seed. A separate CPU
checkpoint replay must reproduce sampled logits and answer argmax within
absolute and relative tolerance0.002. Source, suite, checkpoint and prior
decision hashes are archived; no bulk predictions or checkpoints enter Git.

The screen has an independent1h wall,12GiB sampled MPS and1GiB artifact cap.
Before trained inference, run a source-matched random-weight MPS benchmark
with six warmup and24 timed complete batches. Worst-case all three arms×two
seeds×528 batches per arm/seed must project below3,600s under the frozen
`1.75×median batch seconds×528×6+300s` formula. Actual resource caps are
enforced during inference and auditing. No paid API or manuscript data is
used. A screen pass says the model can answer these synthetic programs; it
does not establish the internal mechanism or decipherment.
