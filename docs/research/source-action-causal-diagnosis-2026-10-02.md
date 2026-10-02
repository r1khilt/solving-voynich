# Diagnose memory use before interpreting a failed inverse as a circuit

Own exploratory checkpoint diagnosis, not a recovery gate. SOURCE-ACTION-TRAIN-002 remains unchanged. Its first1000-update checkpoint reduced guided-path NLL776→171 but reduced greedy completion16→4/64 and recovered no exact records. We must distinguish early irreversible binding mistakes from inadequate use of already correct memory.

## Prior work and method choices

The existing [interpretability review](deep-review-2026-09-21/INTERPRETABILITY.md) and [raw mechanism program](latent-mechanisms-2026-09-21/METHODS_REVIEW.md) distinguish readable information, predictive state, causal intervention and historical identification. Earlier project probes/steering and source-state experiments already exist; changing predictions alone did not qualify a manuscript mechanism. The [reading-action review](source-action-inverse-2026-10-02.md) records selected decipherment-method precedents, including synthetic recurrent plaintext generation, and their channel/data/spacing limitations. This diagnostic uses that existing fixed toy channel; it does not infer Latin or glyph units for Voynich.

[Zhang and Nanda2024, v2 §§2.1 and6](https://arxiv.org/html/2309.16042v2), selected sections read, show that corruption methods and metrics affect localization. We use a sham plus two distinct structured feature changes rather than Gaussian noise. Their in-distribution symmetric replacements do NOT make our inconsistent-memory packets in-distribution. [Heimersheim and Nanda2024, v1 §§2.6 and3](https://arxiv.org/html/2404.15255v1), selected sections read, explain the dependence of patching claims on what changes and distinguish necessity from sufficiency. We measure input dependence, not a minimal circuit or semantic variable.

Existing Voynich boundaries remain in [prior work](PRIOR_WORK.md) and the [joint-reading review](joint-reading-independence-2026-10-02.md): ambiguous transcription, source-language/channel uncertainty and language-like output are distinct from independently verified reading. No established Voynich application of this exact memory intervention was found by this focused review. The larger12-layer96M architecture does not remove those limits.

## Fixed comparison

Use ONLY the published binding92403 checkpoints0and1000 and the original64 EXPOSED Pliny development episodes, with all source/role/window/key/metadata hashes unchanged. Reconstruct each complete teaching trace from its audited source windows. Teacher forcing supplies correct preceding choices; it does not measure autonomous search. Every query's memory contains only past bindings, with future teacher states causally masked. No new keys, sampling, source windows, hidden-author data or manuscript panel.

Each of128 checkpoint/episode pairs receives four CPUfloat32 forward passes:

1. Base: original packet, original legal masks and whole-path probability.
2. Sham: clone binding indices, change nothing.
3. Erase: replace each occupied row/unit feature with that row's unbound feature. This retains row identities, baseline unbound embeddings and the entire symbolic task.
4. Rotate: at EVERY active query, cyclically move bound unit identities among its occupied source rows. Keep occupancy, unit multiset, row identities and unbound slots fixed. Zero/one occupied row is unchanged; duplicate-unit equality may also yield no change.

All other packet tensors are the SAME objects: observations, offsets, selected record, preceding actions, legal masks, active queries and targets. Changed neural memory can be inconsistent with the real dictionary; it is explicitly a feature intervention, never a new literal environment state. It can induce distribution shift. Erase also removes occupancy information from this neural channel; rotate retains occupancy/unit multiset but changes row/unit relationships. Neither alteration is guaranteed to preserve embedding norm or represent a naturally occurring donor state. Do not equate differing effect sizes with isolated causal factors.

Measure whole-path NLL, plus action counts, correct argmax counts and summed NLL separately for FIRST-binding versus REUSE target actions under the original truth trace. Also retain forced-single-legal-action counts and correctness among NONFORCED queries: the literal grammar can supply the answer without neural discrimination. These categories and denominators do not adapt after seeing intervention effects. Keep per-episode scores and every per-action target log probability privately hash-bound. Report descriptive paired changes; tokens are dependent and not independent experimental replicates. A positive sensitivity is not a recovery/circuit PASS. Two independently trained seeds and matched free-running behavior remain required for any stronger mechanism claim.

Even zero sensitivity would not prove the entire model ignores the dictionary. The legal mask still encodes compatibility with its past bindings, and the causal decoder's earlier actions plus observed ciphertext can provide redundant binding information. This experiment isolates the EXTRA explicit neural memory contribution. It does not delete the symbolic dictionary or all alternate computational pathways. A large effect can also reflect distribution shift, so neither sign justifies a named latent circuit by itself.

## Where free reading first goes wrong

Replay saved greedy actions against the fixed truth action trace and find the first divergence. Before and at that divergence the preceding history is identical, so full causal teacher logits can verify the saved greedy choice without resampling. Classify the truth action as first-binding/reuse, the chosen action likewise, wrong source row versus wrong unit length, and whether the chosen action permanently introduced a unit incompatible with its row in the generating dictionary. Gold is used ONLY for diagnosis, never inputs or repairs. Wrong source labels with duplicate emission units may leave the dictionary correct and must not be mislabeled inventory failure.

The classifier does not analyze later off-trajectory errors or show that revision would solve the case. It tests the restricted claim that irreversible dictionary mistakes occur at first divergence. Correct-prefix truncation cannot be a true dead end because the next truth action remains legal; reject such artifacts. This finite structural reasoning and independently replayed initial/1000 outputs precede the diagnostic.

## Engineering and scope gates

Eight tiny CPU fixtures verify sham identity, fixed non-memory tensors/legal masks, row/occupancy/unit-multiset and padding handling, exact CPUdouble insensitivity of the matched binding-input-off network, invalid modes/row identities, duplicate-unit versus wrong-inventory classification, trace grouping and forced-action accounting. Only newly added diagnostic source/tests are edited; all73 live-training dependencies remain byte-identical to its original freeze.

For each original full-size base path require CPUfloat32 whole-path NLL within.01 of saved MPS score. Shared-prefix greedy chosen-logit deficit≤.0002; every legal score finite, illegal mask exact, sham logits EXACT. These are numerical admission gates, not source-law calibration. Save partial checkpoint receipts/failure if the registered single run cannot complete. No extrapolation from short numeric fixtures to arbitrary future model checkpoints.

[SOURCE-ACTION-DIAG-001](../experiments/SOURCE-ACTION-DIAG-001.md) freezes all128 pairs and512 forwards. Estimate several minutes to tens of minutes for CPU workload, bounded by1hwall/5000absoluteCPU seconds/3GiBhost/32MiBprivateoutputs/zero paid cost/CPUthreads2. This estimate is planning, not a measured full-size CPU benchmark. GPU training runs independently under its existing bounds; no diagnostic GPU calls or training changes. Record actual forward timings and per-phase resource checks. The inference model is96,039,982parameters at both checkpoints, no training/optimizer or new generation. Stronger causal path/neuron/J-space analysis waits for results and its own prospective design.
