# Cipher access, supplied history, and the first-binding bottleneck

Prospective design for [SOURCE-ACTION-CIPHER-001](../experiments/SOURCE-ACTION-CIPHER-001.md). Synthetic EXPOSED development; actual Voynich decipherment remains unresolved. This memo specifies a diagnostic, not a new training objective or a qualified mechanism.

## Evidence motivating the question

[The earlier binding-feature diagnosis](../experiments/SOURCE-ACTION-DIAG-001-results.md) found that 98.6615% of checkpoint0→1000 guided loss improvement belonged to reused bindings. Guided first-binding accuracy was192/1246, reuse14648/17526. Every first autonomous error introduced an incorrect binding. Erasing or rotating explicit neural binding features had limited whole-path effects while retaining history and symbolic legality. These observations suggest a history shortcut but do not establish it.

[The stochastic study](../experiments/SOURCE-ACTION-SAMPLE-001-results.md) saved97 actual draws, then failed at reporting; its sole full closure also failed. Narrower publication accounting verified zero exact positive records and six completed wrong readings, all receiving much lower original source targets than known outside answers. These failures remain failures. Source discrimination among completed readings and proposal coverage are different problems. Neither previous study established that the neural reader integrates the whole observed cipher.

## Primary method review and its limits

Zhang and Nanda, *Towards Best Practices of Activation Patching in Language Models* ([2024 version](https://arxiv.org/html/2309.16042v2), sections2.1 and6 read in this work block), show why corruption and metric choices affect patching conclusions. Our suffix shuffle preserves glyph counts, lengths and first-occurrence naming, but is not claimed to follow the cipher distribution. We report fixed-competitor log-probability differences alongside loss and autonomous behavior. A large loss effect under an artificial route cut may include distribution shift.

Heimersheim and Nanda, *How to use and interpret activation patching* ([2024 version](https://arxiv.org/html/2404.15255v1), sections2.6 and3 read in this block; prior review includes2.5), discuss backup behavior and why necessity and sufficiency are different. Removing four cross-attention writes tests an aggregate route. A small effect does not show that the entire input is irrelevant; a large effect does not identify a minimal circuit. We therefore preserve and name the legal-mask and history routes explicitly.

Chiang et al., *Deciphering Foreign Language* ([NAACL2010](https://aclanthology.org/N10-1068.pdf), cascade context in section2 reviewed here; source/channel discussion in prior joint-source review), supply precedent for combining a separate source model with a decipherment channel. Our language-only baseline is a diagnostic of what that original source plus literal action legality can achieve. Locally normalizing joint-target increments does not integrate future continuations and is not an exact posterior solver.

The user's [Anthropic global-workspace reference](https://www.anthropic.com/research/global-workspace), including its Jspace reporting and language-continuation examples reviewed here, motivates separating prediction from explicit globally available information. Frontier-language-model task dissociations do not establish the same workspace in this96M supervised action reader. This study constructs no Jspace lens, sparse direction, union cone, or neuron-level circuit. Prior program context: [latent-method review](latent-mechanisms-2026-09-21/METHODS_REVIEW.md) and [prior decipherment work](PRIOR_WORK.md).

## Five neural conditions

All five conditions preserve the actual cipher, literal environment, targets, legal masks, previous selected actions, record-selection rule and progress features. Only neural computations below change. Gold paths supply earlier correct actions in guided evaluation, never in autonomous decoding.

- **base:** unchanged inference.
- **sham:** clone glyph/binding inputs and cross-attention outputs. Every guided logit and free prediction must remain bit identical.
- **shuffle:** deterministic per-record suffix permutation at the glyph embedding lookup. Keep the prefix through the last first-occurrence glyph, every glyph count, record length, padding and original environment. The permutation is fixed for a case across all checkpoints and decoding routes.
- **cross_bias:** replace every decoder cross-attention output with its learned output-projection bias, or zero if no bias exists. This removes variable context writes, including their dependence on the decoder query; it is not a pure removal of cipher information alone.
- **cross_bias_erase:** the same cross-attention cut plus each binding row's own unbound neural embedding index. Symbolic bindings and history remain available through legality and past actions. This still is not a wholly cipher-free system.

The fast decoder implements attention manually, bypassing ordinary module forward hooks. Both implementations receive the intervention, count its execution, and are compared on every actual greedy prefix. Without this check a false null effect would be possible.

## Two source-only conditions and exact potential identity

Both use ONLY the selected record's previously chosen source letters and the original symbolic legal mask. They use the original audited dense suffix source, not a newly fitted model, gold continuation, other record's future plaintext, or cipher encoder.

For source row a and legal action with length l, **row_only** scores log Q(a|past) minus log of the number of legal unit lengths for that row. This avoids counting an unbound row twice merely because both lengths are legal. Conditional row probability is Q restricted to legal rows; lengths are uniform within the chosen row.

**source_increment** instead scores log Q(a|past)+log(1−rho), minus log42 when introducing a new binding, plus log rho when this action completes the selected record. rho=1/225. On a complete compatible path, the unnormalized increments sum to the original source point log probability minus visited_rows×log42. That identity is checked on all64 known traces. The resulting locally normalized greedy policy is a heuristic: it does not account for future legal completions or compute whole-key posterior density. Unused key rows are integrated out in the original joint target, never scored as recovered.

## Readouts and interpretation

Guided evaluation separates first binding/reuse, source-row accuracy, unit-length accuracy and exact-action accuracy. Split loss exactly into −log P(true new/reuse category) and −log P(correct action | true category). True category labels are used only to analyze outputs. The baseline's highest-scoring legal non-target action is fixed across neural interventions; report its target-minus-competitor log probability. Source policies select their own fixed competitor independently. Whole-path loss remains primary descriptive accounting; do not treat thousands of dependent tokens as independent experimental units.

Autonomous evaluation starts with an empty dictionary and follows each policy without repair. Report all64 positive attempts and33 null/ambiguous controls for every policy. A failed reading incurs full true-length edit penalty and zero used-key matches/exacts; completed readings are checked for literal re-encoding. Probability recorded along a greedy path is its density under the conditional policy, not the output law of deterministic greedy decoding, key density, or cipher evidence.

Compare checkpoint0/1000/4000 and all cases; no favorable subset or positivity gate. A language baseline matching guided reuse while neural cipher-route cuts barely alter that behavior would strengthen a shortcut explanation. A significant free-reading deterioration under cuts would show aggregate dependence on those computations, with OOD and redundant-route caveats. Neither outcome identifies a historical channel. A new first-mapping or globally revisable objective would require its own frozen calibration; the ongoing training campaign is unchanged.
