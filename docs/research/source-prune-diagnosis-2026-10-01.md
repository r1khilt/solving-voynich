# Known-answer pruning diagnosis before scaling search

2026-10-01. Own prospective method and interpretation; no new empirical outcomes in this document.

The last compiled source/key search recovered one of four exposed artificial cases. Its other three missed the true used dictionary in the returned groups. That observation does not say whether the literal true path died early, a compatible dictionary arrived through another reading, or a compatible terminal group was merely omitted from the 512 returned outputs. These are distinct failure modes.

## Primary-source review and applicability

Read Wiseman and Rush (2016), introduction, related work and selected section 4 beam-search optimization passages. Their supervised method penalizes loss of the gold sequence during training and can resume from gold after a search error. This motivates observing first loss; our experiment neither trains a discriminator nor reinstates gold. Their nonprobabilistic scores must not be substituted for our source probabilities without a new model/protocol. [Paper](https://aclanthology.org/D16-1137.pdf).

Revisited Nuhn, Schamper and Ney (2013), section 4.5 pruning and selected section 6 beam/order results. Histogram pruning keeps a fixed number of ranked hypotheses; their experiments show that higher-order scores need sufficient search capacity. Their letter/word substitution and homophonic tasks differ from our shared variable-length emission dictionary and two interleaved records. Larger widths and published hardware timings do not guarantee our recovery. [Paper](https://aclanthology.org/P13-1154.pdf).

Revisited Hauer and Kondrak (2016), selected prior-work passages and section 5 Voynich data/experiments. Their language identification and anagram/abjad decipherment operate under different transcription and vocabulary assumptions. Those Voynich-specific experiments are precedent to scrutinize, not a validated manuscript solution or justification for imposing Latin here. [Paper](https://aclanthology.org/Q16-1006.pdf).

The previous [state-lattice derivation](source-state-lattice-2026-10-01.md) reviews Mohri's semiring/DAG framework. Exact merging is justified only for our supplied Markov state and shared partial key; neural hidden states cannot be merged merely because their cosine similarity is high.

## Own observational construction

Keep the frozen original C++ source byte-for-byte. Generate a separately hashed copy by unique, hash-guarded insertions. Preserve original State/Value types, candidate hashes, ranking comparator, source weights, pruning, guide/cache and terminal selection. A separate thread-local observer receives a deterministic literal source/key trajectory. A separate guide cache evaluates this trajectory; it never touches the ordinary guide's cache/work counters. A known-answer observation is supervised diagnosis, not a blind search input or causal intervention.

Track literal-path survival monotonically. With unmerged search, retain the actual child serial ID; another state with equal offsets, contexts and partial key cannot revive the literal path. With merged search, the true history remains included if its sufficient-state aggregate is retained and its correct edge expanded. Track structural aliases separately after loss.

Before geometric pruning, record actual aggregate priority/rank and the number of structural aliases. After preliminary pruning/guide scoring, record presence and guided rank; after final pruning, record presence and correct-edge expansion. Also record raw literal-prefix log mass plus geometric bound and iid surrogate priority, and how many actual competitors have strictly greater/equal priorities. The ghost uses literal mass, while a surviving merged state uses aggregate mass; their ranks need not coincide. These counts are conditional on the actual already-pruned frontier. Ghost rank after loss is not the rank in an alternative world where gold had been retained.

At completion examine ALL terminal key groups before output truncation. Record both the exact gold partial-key group and the best group agreeing with every used gold row while possibly assigning extra unused rows. Neither compatible key support nor another terminal reading restores literal true-path survival. Absence from all visited groups still does not prove global impossibility.

Original sixteen complete frontiers are replayed with unchanged finite work/width/allocation limits. The per-call time allowance is 600 seconds solely for observer overhead; every normal returned field, trace, count, mass and terminal must exactly equal its frozen old JSON values. Any mismatch fails, with no retuning. This is full-value equality, not raw binary identity or a CPU-time-matched recovery improvement.

Alternate Python arithmetic independently schedules gold edges, recomputes both prefix probabilities from scratch, includes once-binding priors/EOS/context resets, directly sums allowable geometric lengths, and evaluates the separate vectorized iid backward guide. It checks scalar values to 1e-7. Complete realistic observer replay reuses the native search implementation and author; it is not independent full-search certification.

## Interpretation and next decisions

An early geometric prebeam loss suggests preserving diversity or improving a bound before an expensive guide. A surviving prebeam followed by final loss implicates guide ordering at this fixed width; it does not establish that the source itself is wrong. A surviving literal path with compatible group omitted from output implicates output coverage. A surviving/returned compatible key with a wrong selected reading would implicate final scoring/decision, which this observation does not change.

The recently completed 95M-parameter neural models did not pass their capacity preference gate. Do not label their activations a decipherment mechanism merely because they predict better than initialization. A future learned search critic could train on synthetic early-loss examples, keep exact source-law fitting unchanged, and require disjoint-key/null tests plus matched causal interventions before mechanistic interpretation. That idea is unimplemented and not authorized as an open-ended new training loop by this registration.
