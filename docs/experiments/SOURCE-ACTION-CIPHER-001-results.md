# SOURCE-ACTION-CIPHER-001 results

The trained96M reader is almost insensitive to this permutation of the cipher supplied to its encoder. It chooses exactly the same autonomous paths on all97 cases, despite substantial changes to the encoder input. Its guided gains mostly concern reusing a supplied correct dictionary. A separate original-source policy scores better on that guided task, but none of the seven policies recovers an exact record. This strengthens a shortcut explanation; it does not identify a minimal mechanism or decipher Voynich.

[Registration](SOURCE-ACTION-CIPHER-001.md), [method and primary-source review](../research/source-action-cipher-history-2026-10-02.md), [complete compact result](../../results/SOURCE-ACTION-CIPHER-001/result.json), [model-free closure](../../results/SOURCE-ACTION-CIPHER-001/audit.json), [post-outcome accounting](../../results/SOURCE-ACTION-CIPHER-001/publication-summary.json). Exact origin/main freeze **4fce9f32ac646ff93a016f3d03d21c61d805ba20** preceded ONE actual CPU call/session96797, terminal0; ONE registered audit/session68634, terminal0.114 frozen tracked dependencies, including all73 live-training and102 prior stochastic-study dependencies, unchanged. No retry, sampling, optimizer, GPU diagnostic, paid API, reserved author or manuscript.

## What was actually changed

Same64 EXPOSED Pliny two-record cases/33 controls, binding92403 checkpoints0/1000/4000, original96,039,982-parameter architecture and audited original source.960 guided neural forwards;485 cached neural greedy paths with485 full causal prefix references;194 source-only greedy paths. Every autonomous policy starts empty and receives no gold history. Guided evaluation supplies correct past actions and hence correct symbolic bindings.

The encoder shuffle retains each record's prefix through its last first-occurrence glyph, counts, length, padding and naming. On positives, protected2912 of34918 glyph positions; eligible suffix32006, actual changed positions25600 (73.31% of the whole input). Crucially, the literal environment, current observed glyphs used by legal masks, past actions and symbolic bindings remain original. This tests a specific route for global order information, not removal of every cipher signal. The feature counts and literal-path comparisons are post-outcome saved-file accounting, not another model run.

Cross_bias replaces all four decoder cross-attention writes with their learned projection biases; cross_bias_erase additionally removes explicit neural binding-unit features. These cuts remove query-dependent context computation and may be OOD. History and legality still transmit dictionary information. Neither cut is an isolated proof about cipher content or a minimal circuit.

## Guided behavior

Mean whole known-path negative log probability, nats; lower is better. This is conditional on correct supplied history, not cipher evidence or autonomous recovery.

| Policy | Checkpoint0 | Checkpoint1000 | Checkpoint4000 |
| --- | ---: | ---: | ---: |
| Neural base | 776.414682 | 171.249600 | 162.795802 |
| Sham | 776.414682 | 171.249600 | 162.795802 |
| Encoder suffix shuffle | 776.417746 | 171.249507 | 162.795664 |
| Cross-attention bias only | 776.095453 | 188.150896 | 166.059262 |
| Bias only plus binding erase | 783.494710 | 192.846227 | 165.553406 |
| Original source, row_only | 141.638598 | same source | same source |
| Original source, source_increment | 107.185404 | same source | same source |

At1000/4000, shuffle changes **zero** of18772 guided action argmaxes per checkpoint. Mean absolute paired whole-path loss change is0.000389/0.000522 nats; signed change−0.000093/−0.000138. Untrained shuffle changes82 argmaxes and has0.186947 mean absolute effect. This loss of sensitivity after training is consistent with reduced use of suffix order; it does not distinguish constant encoder memory, unordered summaries, a preserved-prefix dependency or redundant routes.

Cross cut at4000 raises mean loss3.263460 nats (mean absolute3.719284); combined binding cut raises2.757604 (absolute4.202022). Some individual cases improve; aggregate cuts are not interchangeable necessity measures. At1000, increases16.901296/21.596627. Context writes have a measured contribution, especially first-binding loss, even though suffix shuffling scarcely matters.

First-binding denominator1246; reuse17526. No forced single-legal-action query anywhere.

| Policy at4000 or fixed source | First-binding letter correct | First-binding unit length correct | First-binding full action correct | Reuse full action correct |
| --- | ---: | ---: | ---: | ---: |
| Neural base | 236/1246 (18.94%) | 954/1246 (76.57%) | 196/1246 (15.73%) | 14880/17526 (84.90%) |
| Encoder shuffle | same | same | same | same |
| Cross cut | 201/1246 | 939/1246 | 160/1246 | 14877/17526 |
| Cross cut plus binding erase | 198/1246 | 939/1246 | 164/1246 | 14830/17526 |
| Source row_only | 563/1246 (45.18%) | 233/1246 (18.70%) | 87/1246 (6.98%) | 15568/17526 (88.83%) |
| Source_increment | 468/1246 (37.56%) | 288/1246 (23.11%) | 73/1246 (5.86%) | 16005/17526 (91.32%) |

The source baseline selects letters more successfully on supplied histories; the neural reader selects unit length more successfully. A low whole-path loss hides this tradeoff. These are guided conditional accuracies, not independent mapping recoveries.

The registered exact loss decomposition is −log P(true new/reuse category) plus −log P(correct action | true category). Post-outcome arithmetic of checkpoint0→4000 shows98.6772% of total loss gain in reuse, and91.7062% of that reuse gain in category allocation. The model mostly learns to put probability on already established rows under correct past bindings. This is descriptive accounting, not an identified category neuron or proof that all gains are a shortcut.

## Autonomous outcomes

All64 positive attempts per policy;128 record attempts,18772 true letters,1246 used key rows. Every policy has **zero exact records and zero complete used keys**. Failures receive full true-length edit penalty and zero matches; unused rows are never scored. Edit totals can exceed true length when completed wrong readings are longer than truth.

| Policy | Completed readings /64 | Used-row matches /1246 | Full-penalty edits |
| --- | ---: | ---: | ---: |
| Base | 4 | 5 | 18537 |
| Sham | 4 | 5 | 18537 |
| Encoder shuffle | 4 | 5 | 18537 |
| Cross cut | 5 | 7 | 18503 |
| Cross cut plus binding erase | 4 | 9 | 18545 |
| Source row_only | 64 | 29 | 26122 |
| Source_increment | 64 | 28 | 26012 |

Base and shuffled **actions, literal dictionary, texts and ending are identical on all97 positive/control cases**. Conditional path densities can change slightly; identical greedy paths do not imply identical probability models. Shams are bit identical including density.

Both source policies select only one-glyph actions in all97 cases (51742 actions each), producing long, wrong readings. They also complete every shuffled/IID null and the ambiguity control. All five neural policies fail all16 shuffled and16 IID controls and complete the single ambiguous case. Therefore completion is not a semantic/language discrimination result. Source_increment's unnormalized complete-path potential is mathematically correct, but local normalization and greedy selection still ignore future consequences; it is not exact posterior inference.

## Engineering and audit scope

All192 guided base CPU32–saved MPS whole-path comparisons pass, maximum1.804488e−5 nats≤.01. All485 cached/full intervened neural prefixes pass: maximum legal-logit difference4.959106e−5≤.002, path-density difference1.816520e−5≤.01, reference argmax deficit0≤.0002. Every expected ordinary/manual intervention hook executes; every mask/legal score check passes; all shams exact.

Actual diagnostic1154.874850 wall/1209.405310 CPU seconds, peakRSS2102050816 bytes, private archived payload33485560 bytes plus trace, $0. Limits1h/5000absoluteCPU/4GiB/64MiB held. PyTorch2.14.0, CPU2 threads, MHAfastpath disabled, float32 neural/float64 source/scoring; environment saved in receipts.

ONE closure68.569649 wall/66.450196 CPU seconds, peakRSS1285029888 bytes/$0, under600wall500absoluteCPU3GiB. Checks all114 freezes/manifests/hashes; all saved neural teacher group/category/margin arithmetic and192 CPU-MPS values; all679 literal paths, chosen-action densities,485 actual/reference vector comparisons and route counts; source-policy teacher/path reconstruction through original arrays; every positive metric with independent integer DP and all controls. No neural model, optimizer or sampling calls. Neural teacher probabilities are saved-arithmetic checked, not model-replayed. Same researcher, not independent expert replication. Post-outcome publication accounting11.473258wall/3.126235CPU/.229GB additionally verifies entire1708-row trace order, private ignores, feature changes and base/shuffle path identity; no new scientific outcome/gate.

The ongoing training remains unchanged/live beyond12181updates. Its saved10000 checkpoint reports NLL158.154122, four completed positives, zero exact records/used keys. A separate checkpoint publication check verified64 literal positive metrics/independent DP and file hashes, then **FAILED** comparing saved control lists with reconstructed tuples. Failure and a narrow normalized-metadata explanation are preserved; control replay/finite-tensor checks were not completed and no full-check retry occurred. An initial direct-script import failure happened before its start receipt and was corrected in the invocation environment. This ancillary failure does not turn the completed CIPHER001 run or closure into a failure, and it does not qualify the10000 checkpoint's full audit. Original campaign completion audit remains pending.

## Implication and next question

The neural reader's first dictionary decisions remain poor, its global encoder order contribution is tiny under this corruption, and better guided prediction has not delivered recovery. A justified redesign should explicitly target dictionary hypotheses from the full cipher, use the separately audited source rather than require the neural reader to relearn it, and permit coherent revisions of early mappings. That is a prospective direction, not an implemented improvement. It must pass fresh-key exact recovery and null controls before use as a manuscript solver.

This is one exposed seed, three correlated checkpoints, a conditional Latin/source/channel test and artificial feature interventions. The original cipher still enters local legality, and preserved prefixes/counts can carry useful information. No stable circuit, Jspace, language identification, historical mapping, translation or Voynich decipherment was established.
