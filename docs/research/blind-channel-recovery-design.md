# Blind channel recovery: infer the encoding units before polishing the key

Design review: 2026-09-25. **Proposal only: no new benchmark has been generated, no model has been fit, and no new answers have been scored.** This is a design specification to turn into one separately frozen experiment, not an implemented capability or a registration with completed input manifests. The original research charter is unchanged.

## Decision and research question

Finish the already bounded NAIBBE-003 source-prior repair, then retire the present Naibbe publication as a source of fresh confirmation. NAIBBE-002 already recovered all 126 non-y/z assignments and nearly all ordinary text, while its unchanged macro-key gate correctly remained FAIL. Its largest remaining assistance is the legal parse lattice built from known table membership, within-table role links, and the role grammar. Additional rare-letter accuracy would not remove that assistance. See [001 results](../experiments/NAIBBE-001-results.md), [002 results](../experiments/NAIBBE-002-results.md), and NB-190–195.

The next substantial question is: **Can one bounded decoder infer reusable variable-length encoding units, their mappings, and a small amount of channel state from visible ciphertext alone, then recover new text under the same inferred channel?** The deliverable is a source-to-cipher generator and a cipher-to-source decoder with explicit uncertainty. A favorable language-model score without plaintext recovery on controls is insufficient.

Use one locked benchmark with an easy family, an unknown-unit family, an unseen stateful family, and matched nonlanguage controls. Oracle ablations identify the missing capability on the same benchmark. Do not create a new series of increasingly tailored controls after each answer becomes visible.

## Primary-source basis and limits of the analogy

The following sources were opened during this review. The proposed finite model below is our design, not a theorem or implementation copied from these papers.

| Primary source | What it establishes and what we take from it | Limit that matters here |
| --- | --- | --- |
| [Chiang, Graehl, Knight, Pauls & Ravi, 2010, *Bayesian Inference for Finite-State Transducers*](https://aclanthology.org/N10-1068.pdf), §§2–4 | Finite-state cascades support composed derivation lattices, normalized competing transitions, forward-backward estimation, and tasks including substitution decipherment and segmentation. Keep source, channel, and observation assumptions separate. | Their initial cascade specifies the parameter inventory; it does not discover unrestricted cipher structure. §3.2 explicitly skips a Metropolis-Hastings correction for approximate path proposals. We cannot call an analogous approximate sampler exact. |
| [Ravi & Knight, 2011, *Deciphering Foreign Language*](https://aclanthology.org/P11-1002.pdf), §2 | A source model plus an explicit channel can be trained from ciphertext; constrained structure and source/channel separation are useful computational tools. | Their word-substitution setting supplies word units. The final decoder stretches channel probabilities, and the paper also discloses skipped sampling correction. These are not a normalized evidence prescription for our unknown-unit comparison. |
| [Ristad & Yianilos, revised 1997, *Learning String Edit Distance*](https://arxiv.org/pdf/cmp-lg/9610005), §2, Theorem 1, Appendix B | Defines normalized stochastic edit processes, distinguishes summing paths from choosing one path, and makes termination/length modeling explicit. | Their memoryless joint edit model differs from our source-conditioned channel. Proper stopping and length treatment must be derived for our own process; a length penalty added after scoring is not automatically a probability model. |
| [Allman, Matias & Rhodes, 2009, *Identifiability of Parameters in Latent Structure Models with Many Observed Variables*](https://arxiv.org/pdf/0809.5032), §§1,6 | Distinguishes label swapping and generic identifiability under specific latent-model assumptions. | Its HMM results do not establish identifiability of our variable-length transducer, an unknown language, or a finite manuscript. We must exhibit ambiguities and avoid treating finite-data stability as a proof. |
| [Kiefer, 2020, *Notes on Equivalence and Minimization of Weighted Automata*](https://arxiv.org/pdf/2009.01217), definition in §1 and Theorem 2.3 | Provides an exact behavioral notion—equal weights on every string—and an algebraic equivalence test that can produce a distinguishing string. | This applies to finite weighted automata over a field. A test on finitely many strings is weaker; floating-point near-equality is not an exact certificate. Equal ciphertext distributions do not guarantee equal plaintext posteriors. |
| [Grünwald & Roos, 2019, *Minimum Description Length Revisited*](https://arxiv.org/pdf/1908.08484), §§2.1,2.3 | Explains valid two-part codes and why a model description must be paid for along with the data description. | Our proposed discrete code is one explicit practical choice, not optimal universal coding. An arbitrary complexity coefficient is not automatically MDL, and singular latent models do not justify casually substituting a parameter-count penalty. |
| [Ellis et al., 2021, *DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning*](https://www.neurosymbolic.org/papers/EllisWNSMHCST21.pdf), §2 and footnote 1 | Separates neural proposal/search policy from executable candidate programs and their task likelihood. This motivates learned search ordering followed by explicit scoring. | Its beam approximation is not exhaustive inference. Executing a candidate verifies consistency with observations, not historical truth. We propose search acceleration only after the exact scoring problem works. |
| [Song, Zhao & Ermon, 2017, *A-NICE-MC: Adversarial Training for MCMC*](https://papers.nips.cc/paper/7099-a-nice-mc-adversarial-training-for-mcmc.pdf), §4 | Combines learned proposals with a Metropolis-Hastings step to preserve a specified target distribution. | It uses continuous invertible proposals. Our discrete variable-structure proposals would require their own tractable forward/reverse probabilities, support argument, and convergence checks. This is a later option, not an implemented guarantee. |

Voynich-specific precedent also warns against importing assumptions unnoticed. [Reddy & Knight, 2011](https://aclanthology.org/W11-1511.pdf) examine linguistic/statistical questions without establishing the units or a decipherment. [Hauer & Kondrak, 2016](https://aclanthology.org/Q16-1006.pdf) investigate language identification and anagrammed substitution under explicit source-language and cipher constraints; their Voynich suggestion is not a recovered, independently verified manuscript plaintext. The new benchmark must therefore measure recovery under withdrawn assumptions, not convert a candidate-language ranking into a historical conclusion.

## One explicit normalized model

Let `A` be a candidate plaintext alphabet, `G` the observed glyph alphabet, `h` a finite source-model context, and `s` an unnamed channel state. Source alphabets/language candidates remain declared assistance in this first benchmark; they are not inferred historical facts. The observed file contains glyph strings and record boundaries, with no codeword boundaries, table names, role labels, or permitted-parse lattice.

For each candidate source language, freeze a finite-order character source model `p_L(a | h)` trained on independent authors. Add a proper stopping law `rho(h)` with a strictly positive lower bound fixed from source-only development. At each source position:

1. Stop with probability `rho(h)`.
2. Otherwise draw a plaintext character `a` from `p_L(. | h)`.
3. Draw a next channel state and a nonempty glyph string from `q_theta(s', v | s, a)`, normalized over every legal pair `(s', v)` in that row.
4. Emit `v`, update source context with `a`, and move to `s'`.

Use an initial-state distribution `pi_theta`; do not reveal the generator's realized initial states. Start with at most four states, glyph emission lengths one through four, and at most three emission alternatives per `(s,a)` row. These are **proposed computational bounds**, to be checked on development throughput and fixed before the sealed benchmark is generated. They are not discoveries about Voynich. No plaintext deletions, free null glyphs, unbounded copying, or transposition enter this initial family.

Here an encoding unit is an emitted substring `v`; a role is an inferred state-conditioned use of a unit, not a supplied prefix/suffix tag. No true role count or within-role letter linkage is given. A learned model may represent a historical prefix/suffix distinction through states or whole emissions. Recovering the generator's original terminology is not required when a different representation has the same decoding behavior. Discovering a compact factored prefix/suffix dictionary is a possible later compression extension, not a capability silently assumed in this initial model.

For plaintext `x=a_1...a_n`, channel path `s_0...s_n`, and emissions `v_1...v_n` concatenating to observed `y`, the joint probability is

```text
p(x, s, v, y | L, theta)
  = pi_theta(s_0) rho(h_n)
    product_i [(1-rho(h_(i-1))) p_L(a_i | h_(i-1))
               q_theta(s_i, v_i | s_(i-1), a_i)]
    × indicator[concat(v_1,...,v_n) = y].

p(y | L, theta) = sum_(x,s,v) p(x,s,v,y | L,theta).
```

This is normalized: each decision row sums to one, the bounded-below stopping probability gives almost-sure termination, and every terminated path emits exactly one observed string. For fixed `y`, every emission consumes at least one glyph, so at most `len(y)` source steps can contribute. The likelihood is an exact finite forward sum over `(glyph offset, source context, channel state)`. An independently implemented exhaustive enumerator must verify the sum on small alphabets. Viterbi gives a best path for display; it must not replace the sum in the model-comparison score. A MAP path is also not necessarily a MAP plaintext, because multiple paths can produce the same plaintext.

The inferred codebook contains explicit glyph strings and probabilities. Proposed units can be initialized from all observed substrings within the length bound, but their actual symbols/lengths and row assignments must be paid for in the model code. Every row remains normalized over its full declared emission set, including emissions not seen in the fitting text. Do not normalize only over paths that happen to match the current observation. A fixed escape component, if needed, must be a fully specified normalized distribution and included in all models before evaluation; it cannot become unpriced per-token exceptions.

This removes the automatic advantage of making a shorter, more fluent plaintext: an explanation must also emit the entire observed string with accounted probability and pay for its reusable machinery. **Normalization does not prove that the correct explanation wins.** The source model can still be misspecified, or several channels can be observationally indistinguishable. Controls test those failures.

Spaces are not silently discarded or equated to source word boundaries. The first synthetic benchmark uses a space-free visible stream and disclosed record boundaries; its optional visible separators are ordinary emissions under the same channel. Later manuscript transcription uncertainty needs a separately normalized observation transducer. Adding deletion/null operations would require a new termination and inference proof; they are not casual extensions to this acyclic algorithm.

## Structure cost and selection without answer access

Use a two-part code fixed before ciphertext fitting:

```text
J(L, theta; Y_fit) = C(L, theta) - log2 p(Y_fit | L, theta).
```

`C` encodes the candidate-language index, number of states, each row's number of alternatives, each emission length and glyph sequence, next-state destinations, the initial-state distribution, and emission weights. Start with rational weights on a fixed finite grid; the grid denominator is part of the registration. Use explicit prefix codes or finite enumerative codes with stated ranges. Duplicate behavior can have multiple descriptions; selecting the shortest discovered description is acceptable but not proof that the global shortest description was found. Never change a penalty after looking at a sealed answer.

Candidate generation may use source-only scores for speed, but retained winners are compared using the same full `J`. The test record likelihood is evaluated with the fitted structure and probabilities frozen. It is conditional predictive performance of that fitted model, not a Bayes factor unless parameters and structures were actually integrated under specified priors. Report the search budget and discovered scores separately from any claim of optimization completeness.

## The single benchmark and its isolation

Proposed benchmark ID: **BLIND-CHANNEL-001**, reserved here but not yet registered. Corpus acquisition must first produce a rights/provenance/hash manifest. No particular uninspected source file is assumed available. Use Latin, Italian, and English as declared source candidates; this is a bounded engineering choice, not a Voynich shortlist. Each language needs five nonoverlapping author groups:

| Partition | Use | Forbidden reuse |
| --- | --- | --- |
| Prior authors P1/P2 | Source-model estimation and source-only hyperparameter selection | No ciphertext-task source excerpts |
| Development author D | Exposed synthetic instances, runtime measurement, all architecture/search decisions | No final source material |
| Final author F | Ciphertext-only key/channel fit for a final task | No source-prior or neural-proposal training |
| Final author T | New text encrypted under the unchanged task key/channel | No prior, architecture selection, fitting, or selection of the task key |

Author identity must be checked across editions, translations, anthologies, quotations, and duplicate excerpts. Works/editions and exact exclusions are fixed before generation. Translation parallels across partitions are forbidden. Language priors may know their alphabet and training authors; source language, task authors, true text lengths, emitting states, and keys are hidden from the task solver. Known candidate languages and the small channel grammar remain explicit assistance.

One benchmark contains three channel families, each with independently generated opaque keys:

| Family | Construction rule | Development exposure |
| --- | --- | --- |
| A: memoryless homophones | One state; nonempty glyph codewords of equal length; several possible codewords per source character | Training/development examples allowed |
| B: memoryless variable units | One state; variable-length codewords with shared prefixes/suffixes; no supplied boundaries or role links | Training/development examples allowed |
| C: state-conditioned variable units | Two or four states; different normalized output laws depending on state/source character; codewords share components across states | **No generated examples or trajectories used for development or proposal training** |

Family C is expressible by the declared search grammar but absent from development data. This is a family-disjoint transfer test within a bounded superfamily, not discovery of an arbitrary unknown cipher. Its generative specification must be sealed before A/B results can motivate special-case changes. Do not generate C, inspect it, then call it a held-out family.

Proposed final matrix: three languages × two independent keys × three families = 18 positive tasks. Each task fits one reusable channel on records from author F and transfers it to author T. Add 18 paired negative tasks, matched by language reference, glyph inventory, record-length schedule, and broad local-frequency targets; generate them from a finite-order glyph process or bounded copy process with no source message. Negative generator parameters are fitted using development controls only, never final plaintext. Add one explicitly designated many-to-one ambiguity task with unavoidable information loss, outside the 18 recoverable-positive tasks, to test abstention rather than demand impossible key accuracy. Thus the proposed benchmark contains 37 task instances; the ambiguity fixture's exact construction must be fixed before final generation.

Eight fitting records and four transfer records per task are a proposed initial size. Source lengths vary within a published envelope; exact lengths are not supplied to the solver. Record boundaries and the length envelope are disclosed assistance. The generator must preserve the distinction between underlying source length, emitted glyph length, and any length-based inclusion rule. Do not trim through an emission and retain the original source target. Any rejection sampling or conditioning used to select tasks is recorded; the decoder's stopping prior need not equal that selection law, and this mismatch must be stated.

An independent builder retains plaintext, source/channel traces, and emission laws in evaluator-only outputs; the solver receives only ciphertext and the fixed public configuration. Keys, source windows, and RNG streams are independent across development and final partitions. Freeze code, inputs, search settings, and baselines before final fitting, then freeze all inferred channels before opening any final transfer text or answers. No sequential revision between final families. Mechanical data-integrity checks may read gold but must not choose easier cases according to solver performance.

## Oracle ablations on the same cases

| Arm | What is supplied | Question answered |
| --- | --- | --- |
| Blind primary | Visible glyph streams and the bounded grammar only | Can units, mappings, and states be jointly inferred? |
| Unit oracle | True boundaries between emitted codewords, without plaintext mappings or state labels | Is unknown segmentation the dominant failure? |
| Structure oracle | True emission inventory and role/state topology, with mappings still hidden | Is structural search the dominant failure? |
| Channel oracle | True full channel probabilities/key, without plaintext | How much source/path ambiguity remains even with correct machinery? |
| Source-language oracle | True language candidate only; otherwise blind | Is language competition causing the failure? |
| Degenerate baseline | Frequency initialization and no contextual key search | Is expensive search adding actual recovery? |
| Nonlanguage baseline | Normalized glyph ngram/copy generator with its own recorded description cost | Does the language explanation beat a competing process on new observations? |

The oracles are diagnostics, not competitors for an undisclosed headline. Do not tune the blind arm using final oracle gaps and then present its rerun as fresh. On development instances, a recovered channel with better fit objective than gold but worse plaintext identifies objective/identifiability trouble; gold beating all searched candidates identifies a missed better candidate, not a proof that unrestricted optimization would uniquely recover gold.

## What counts as recovery and what can be unidentifiable

Primary outputs are plaintext character edit rate on new-author transfer, correct literal decoded spans, uncertainty/abstention, complete ciphertext accounting, and reuse of the frozen channel. Report every task, including timeouts and failed searches. Weight source/key instances equally in the main table; do not let long easy records hide a failed family. Language choice and glyph predictive code length are separate metrics, not substitutes for plaintext recovery.

State names can be freely permuted. Splitting a state into equivalent copies can change the diagram without changing behavior. Accordingly, report transition/mapping accuracy modulo permitted symmetries only when those quantities are identifiable, and do not punish an equivalent reusable decoder merely because its hidden numbering differs. On tiny rational models, compare the composed ciphertext weighted automata algebraically and return a distinguishing string when they differ. For larger models, finite probe agreement is only empirical agreement.

There are two distinct equivalence questions:

- **Ciphertext observational equivalence:** two source/channel models give the same probability to every observed glyph string.
- **Decoding equivalence:** they also give the same plaintext posterior for each glyph string.

The first does not imply the second. Two different languages/channels can explain the same ciphertext while disagreeing about its message. The benchmark must include such a deliberately constructed ambiguity case, requiring broad uncertainty or abstention. A stable optimum, a visually interpretable hidden state, or high compression does not dissolve that ambiguity. No general identifiability theorem is claimed for this design.

## Implementation ladder, resource bounds, and stop/go decisions

These are planning limits, not launched jobs or empirical runtime predictions. Use no paid compute. First measure a development-only pilot; revise and freeze the final resource plan before sealed task generation. Proposed initial envelope: at most two simultaneous CPU workers, 24 GiB aggregate resident memory, and 12 CPU-hours for the first complete development/search block. A slow job must checkpoint and stop at the declared boundary. The sealed matrix gets a fixed per-task budget determined by that pilot; a timeout counts as an observed failure, not a reason to replace the case.

1. **Exact finite model first.** Implement a small forward/backward solver, Viterbi reporting, a code-cost calculator, and an independent exhaustive reference on hand-specified tiny examples. Prove/local-check row normalization and stopping; compare total mass, summed likelihood, gradients/counts where used, best path, and equivalence under state renaming. **Stop** on any discrepancy, any unpriced exception, or any gold access by search. These are correctness gates, not benchmark achievements.
2. **Fit A/B on exposed development authors.** Start with a source bigram model, explicit emission rows, split/merge unit proposals, local key changes, and bounded state changes. Stronger finite-order source models are allowed only if source-only validation and throughput justify them. Use sparse forward states rather than materializing every theoretical lattice state. **Stop expanding capacity** if the known-channel oracle already fails to recover the required text quality: diagnose source mismatch or intrinsic ambiguity first.
3. **Freeze and run the entire matrix once.** Obtain results for the blind arm and named ablations at matched, registered budgets. **Go toward manuscript candidate work only if** unknown-unit and withheld-family recovery materially exceeds the fixed uninformed baseline on new-author text, the effect is not confined to one key/language, the declared recovery criterion passes, and matched negative/ambiguous controls do not receive confident false-decipherment declarations. **Stop manuscript claims** if only oracle arms succeed, if the model selects fluent false readings for negatives, or if key reuse fails on new authors.
4. **Learned proposals only for a demonstrated search gap.** If exact scoring is valid and development gold objectives expose missed good structures, train a proposal model on A/B development-generated tasks. Initially let it rank the same legal split/merge/state/key moves; independently rescore every retained candidate. Preserve a non-neural exploration component and compare under the same wall-time/evaluation budget. This can improve discovery but cannot prove complete search. A later posterior sampler needs explicit forward/reverse proposal probabilities and the MH correction; never claim exactness from rescoring alone.

This design deliberately does **not** invent a numerical plaintext-success threshold from future results. Before any sealed generation/scoring, the actual experiment registration must choose a practical literal-recovery tolerance, uncertainty coverage target, false-declaration rule, paired effect criterion, family-level aggregation, and interval procedure from task requirements plus exposed development behavior, and state the rationale. Those thresholds then remain fixed. With only two keys per language/family, uncertainty will be coarse; this first matrix can establish or falsify engineering competence, not estimate universal success rates. Expanding the matrix requires a new untouched author/key allocation, not repeated tuning on these answers.

Regardless of the numeric outcome, publish the failure diagnosis and stop the old Naibbe rare-letter loop. The actionable result is which oracle gift can now be removed—or precisely which one still prevents recovery.

## Historical and manuscript use

Borg should proceed in parallel as a historical/transcription control. The [completed ciphertext inventory](borg-ciphertext-inventory.md) resolves the ambiguous download: its pinned bytes contain a ciphertext transcription mixed with original cleartext and annotations, rather than automatic Latin decipherment. **Glyph/code units and the final parser remain unresolved.** The inventory's 66 residual codepoint types are not a validated cipher alphabet; they must not be forced into a letter permutation. Preserve its documented exposures and exclude original cleartext as cribs. The [earlier feasibility review](historical-controls-2026-09-25.md) remains the source-history record. Borg's disclosed language and simple-family assumptions differ from the synthetic structural challenge. Success on Borg would improve historical realism; success on C would improve structural inference. Neither alone identifies Voynich's mechanism.

After the blind benchmark qualifies, a manuscript run must propose a constrained reusable channel and a concrete fresh prediction. For example, two competing unit parses may predict different locations of a visible boundary or different repeated source spans. That supplies a reason to restart direct-image measurement or targeted human review. Mechanistic/neural analysis should test whether learned proposals represent the recovered rules and retain them under key/source interventions. Image features, neuron plots, or larger architecture size are not progress unless they resolve an actual decoding ambiguity.

The objective remains actual decipherment with independently applicable rules and external/fresh evidence. This proposal tests a missing capability required for that objective; completing the benchmark would not complete the research goal.
