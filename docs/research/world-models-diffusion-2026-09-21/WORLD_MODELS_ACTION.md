# World models, world-action models, and a decipherment research program

Research date: 2026-09-21. **Research and architectural ideation only. No implementation, training, manuscript scoring, or experiment was performed.** Source IDs resolve in [WORLD_SOURCES.md](WORLD_SOURCES.md). Reading depth varies and is declared there. Statements introduced as proposals or deductions are this review's reasoning, not published demonstrations of Voynich decipherment.

## 1. The ambitious opportunity is inverse world modeling

A serious world-model approach to Voynich would infer a hidden process that jointly explains the writing, its organization, and any recoverable relationship to depicted or described things. Its output would be a constrained, reusable explanatory mechanism and an uncertainty distribution over readings. It would have to predict something that a language prior, copy-and-mutate generator, page-layout classifier, or researcher-selected translation cannot explain equally well.

That objective is substantially more ambitious than training a larger next-glyph predictor. It also differs from generating plausible medieval scenes from the illustrations. The most valuable lesson from modern world modeling is the division between **state estimation**, **transition learning**, **observation generation**, and **action selection**. These components can be learned jointly, but they answer different questions. Collapsing them into a single fluent model makes it difficult to tell whether the system has recovered historical structure or merely built an attractive simulator.

There are four candidate “worlds,” and they must remain explicit:

| World being modeled | Candidate state | Transition or action | Observable evidence | Principal ambiguity |
| --- | --- | --- | --- | --- |
| Semantic/discourse world | Entities, properties, relations, topics, references, discourse commitments | A proposition or discourse operation changes the represented situation | Textual recurrence, possible labels, layout, independently established image relations | Many semantic assignments preserve the same textual distribution |
| Writer/channel world | Encoding key, spelling convention, abbreviation state, copy buffer, line position, scribe-specific realization | A writing/encoding operation produces glyphs or layout | Glyph sequences, allographs, spacing, corrections, local reuse, page geometry | The same text can have multiple generative histories |
| Procedural/referent world | Ingredients, vessels, body parts, calendar states, locations, quantities | A described procedure or natural process changes referents | Diagrams and texts, if independently linked to procedures | A static picture is not an observed execution trace |
| Solver's epistemic world | Competing hypotheses, uncertainty, evidence provenance, costs | Inspect a region, retrieve a comparator, revise a mapping, request expert annotation | New evidence or a mechanically scored consequence of a proposed rule | Improvement in the solver's score need not improve historical truth |

The first three are hypotheses about the manuscript or its production. The fourth is a real environment we can construct: a solver takes an information-gathering or inference action and receives an observable result. It is the cleanest place to borrow planning machinery, but it cannot validate its own assumptions. A policy that becomes exceptionally good at maximizing a translation model's score may simply become an exceptionally good producer of hallucinated translations.

**Proposed central thesis:** build a large inverse generative system with separate linguistic, channel, visual, and inference-state modules. World models supply state and transition structure; diffusion or flow methods supply joint hypothesis proposals; mechanistic interpretation tests whether the system uses the intended variables; external manuscript evidence decides whether a historical reading survives. None of these jobs can substitute for the others.

## 2. State-space thinking is older and sharper than the current label

In a partially observed dynamical model, the current observation generally does not determine the hidden state. A belief distribution summarizes what the history supports. With latent state `s`, observation `x`, and recorded action `a`, the schematic update is:

`b_next(s') ∝ p(x_next | s') × sum_s p(s' | s, a) b(s)`.

This is a modeling template, not a claim that Voynich is a particular hidden Markov process. A writing model may instead emit observations on transitions; a semantic model may be hierarchical; a copying mechanism may need a growing memory. The important commitment is that uncertainty and update rules are represented explicitly enough to inspect.

Predictive state representations offer a complementary foundation. Littman, Sutton, and Singh represent state through probabilities of future action-observation tests and establish representational relationships with finite POMDPs. Their construction does not identify a human meaning for each latent coordinate, and a representational existence result is not a practical learner for an unknown manuscript. [W01](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf)

Our deduction is that a decipherment system should expose several kinds of future question rather than only its next-symbol distribution. Given a candidate state, what recurrence structure, affix compatibility, diagram relation, discourse continuation, or writing operation should become likely? If two supposed meanings answer all available questions identically, the manuscript does not currently distinguish them. A hundred named latent variables would be weaker than a smaller set with demonstrably different consequences.

The system-identification perspective also distinguishes three unknowns: the state within a mechanism, the mechanism itself, and the observation map. On Voynich, all three can be uncertain simultaneously. A capable recurrent model can track state under a supplied mechanism while failing to infer a new mechanism. It can infer an observationally useful state while assigning it an arbitrary coordinate system. It can recover a channel while retaining uncertainty about the underlying language. A useful architecture should return those uncertainties separately.

The project already has relevant negative evidence. Its recorded CAMPAIGN-0001 results distinguish improved familiar-process prediction from failed blind unfamiliar-key recovery, and broad component interventions from selective state recovery. The latest notebook also records failed synthetic deletion/alignment gates. Those are repository observations, not evidence that world models are hopeless. They are evidence that another architecture must change the inference problem, the supervision, or the cross-domain structure rather than simply increase the size of the same task. This chapter proposes a different program; it does not reactivate any registered experiment.

## 3. What the established world-model line actually learned

### Compression, imagination, and policy optimization

Ha and Schmidhuber's *World Models* learns a compressed spatial-temporal representation of reinforcement-learning environments and uses it to train compact controllers; one demonstration trains a policy inside its learned simulator and transfers it to the actual environment. The model is learned from environmental trajectories, and policy success is evaluated against an environment that can be interacted with. This establishes the usefulness of learned simulation in those settings, not that arbitrary passive text reveals a uniquely correct hidden world. [W02](https://arxiv.org/abs/1803.10122v4)

PlaNet makes the learned state model directly useful for planning from images. It combines stochastic and deterministic latent dynamics and a multi-step training objective, with online planning in latent space. Its control tasks provide action-observation histories and rewards. “From pixels” describes the observation channel; it does not mean that the learner receives no actions or task feedback. [W03](https://arxiv.org/abs/1811.04551v5)

DreamerV3 learns environment models and improves behavior through imagined futures, reporting broad performance with one configuration across more than 150 tasks. The relevant achievement is robust model-based reinforcement learning across varied interactive domains. Neither its reported breadth nor its Minecraft result demonstrates inference of an unknown writing system from one static document. The April 2024 arXiv revision is the specific paper version consulted here; its actor acts without online lookahead, after learning from imagined trajectories. [W04](https://arxiv.org/abs/2301.04104v2)

MuZero sharpens a crucial distinction: a useful model need not reconstruct everything. Its learned dynamics support predictions of reward, value, and action policy for search. This is a task-oriented internal model; successful planning does not imply that every hidden coordinate corresponds to an externally meaningful physical variable. The available environment still supplies actions and feedback. [W05](https://arxiv.org/abs/1911.08265v2)

**Our architectural inference:** importing a Dreamer-like actor without a trustworthy reward is premature. Importing a recurrent belief model, multi-step consistency, and explicit separation between observed and imagined states is useful. MuZero is especially attractive for an inference-search controller, because a search policy need only anticipate the consequences of a hypothesis revision. It is dangerous as the sole historical model, because the task score may be too weak to preserve the details needed for decoding.

### Object-centric dynamics

Slot Attention supplies a differentiable mechanism for organizing inputs into exchangeable slots, while SlotFormer models visual dynamics over object-centered representations. They suggest a division between reusable entities and the relations that change between them. Their empirical setting is visual object discovery and dynamics, not the discovery of morphemes, grammatical roles, or manuscript referents. [W06](https://arxiv.org/abs/2006.15055v2), [W07](https://arxiv.org/abs/2210.05861v2)

Our extension would use slots for uncertain recurring units at more than one level: visual parts, lexical candidates, discourse entities, or writing operators. The same slot must not silently change from “plant” to “glyph family” to “topic” depending on which loss is easiest to satisfy. Exchangeability is helpful for arbitrary names, but a stable slot identity across pages needs explicit correspondence evidence. Binding a role to a filler is a separate capability from recognizing either one. A model might recognize a root-like shape and a frequent word without determining that the word names the root.

A graph of relations is more promising than a list of guessed species. Repeated containment, adjacency, branching, count, and ordering relations can be described with uncertainty and matched across pages. Such relations might support anonymous variables before anyone supplies a translation. This is a proposed research route, not a statement that the manuscript's pictures are accurate botanical observations or that they encode those relations linguistically.

## 4. JEPA: learn predictable structure without reproducing every detail

I-JEPA predicts representations of masked image regions, and V-JEPA extends feature prediction to video. They make it plausible to learn useful representations without forcing a pixel generator to model every nuisance detail. V-JEPA reports learning from two million videos without text, pretrained image encoders, pixel reconstruction, or negative examples; its downstream evaluations still involve defined tasks and evaluation procedures. These results concern representation learning, not automatic extraction of a uniquely interpretable ontology. [W08](https://arxiv.org/abs/2301.08243v3), [W09](https://arxiv.org/abs/2404.08471v1)

V-JEPA 2 is the most informative supervision audit. Its passive pretraining uses over one million hours of video. Its robot-control model subsequently receives video **and end-effector state signals**, with action-conditioned prediction. In §3.1, “unlabeled” explicitly means no task, reward, or success annotations; it does not mean absence of the robot's measured state. Pick-and-place evaluation uses supplied image subgoals. Camera dependence and long-horizon limitations are acknowledged. The full-paper distinction matters more for our purposes than the headline benchmark numbers. [W10](https://arxiv.org/html/2506.09985v1)

A manuscript JEPA could predict a missing region's representation from surrounding text, image structure, and page organization. However, the invariances learned by a visual encoder can be actively harmful for cryptanalysis. A tiny mark, rare allograph, or placement difference might carry information that an ordinary image model learns to ignore. Conversely, exact pixel reconstruction can waste capacity modeling stains and texture. The design question is therefore **which information may safely be discarded**, with a reversible observation path retained whenever that answer is uncertain.

Our proposed division is an evidence-preserving channel branch plus a more abstract semantic branch. The first retains glyph alternatives and coordinates. The second predicts anonymous relations and structured continuations. Agreement between them is informative only when each has access to genuinely different evidence and cannot solve the task by copying a page or section identifier.

A JEPA prediction error is also not automatically a normalized likelihood. If two models learn different feature spaces, their raw distances need not be comparable as evidence for historical hypotheses. A system may use feature prediction to build representations, but model comparison still needs a declared observation model, calibrated scoring procedure, or separately justified decision statistic. Calling an embedding distance an “energy” does not by itself make it a Bayes factor.

## 5. Genie and latent actions: an important idea with an important gap

Genie learns a video tokenizer, latent-action model, and dynamics model from unlabelled video. Its latent-action encoder observes past and future frames during training; a constrained discrete bottleneck carries information about change. At inference, user-selected codes replace the latent-action encoder. The paper's main experiment uses eight codes. This is a concrete way to discover a compact control interface from temporal data, but the code names and their causal meanings are not guaranteed by reconstruction alone. [W11](https://arxiv.org/html/2402.15391v1)

Genie 2's official announcement describes an action-conditioned diffusion world model; Genie 3's announcement reports more persistent real-time interactive generation. These are primary product/research announcements, not fully inspected technical papers. We do not infer Genie 3's undisclosed architecture or use visually convincing demonstrations as proof of causal fidelity. [W12](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/), [W13](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)

For manuscript research, the enticing analogy is to infer “latent actions” between text segments. The immediate danger is that there is no guarantee those segments are successive states of the same underlying system. Two adjacent paragraphs might describe different plants; two similar words might be alternative spellings, morphological relatives, cipher variants, or copying artifacts. A latent code that reconstructs the next word could simply encode the next word.

There are nevertheless two defensible extensions. First, a **writer-world inverse dynamics model** could represent transformations that explain how recurring textual templates differ: substituting a stem, choosing a homophone, inserting a conventional abbreviation, modifying a copied pattern, or changing line-position realization. Second, a **procedural-world inverse dynamics model** could represent changes in an independently described process, provided suitable external corpora actually establish text/state correspondences. Both require limits on how much information a latent action can smuggle from the target into the reconstruction.

The decisive distinction is between a compact code for observed differences and an executable operator. An operator should have typed inputs, predictable effects, composition rules, and defined cases in which it does not apply. If an inferred “pluralize” action works only on the particular forms from which it was extracted, it may be a memorized edit. If a supposed “encode under key K” action changes topic or syntax unpredictably, it is not a separated channel mechanism. A latent-action interface becomes scientifically valuable when it can be translated into such consequences, not when its clusters look intuitive.

## 6. VLA and world-action models: the transferable recipe

A VLA is generally a policy that maps visual observations and language to actions. It need not contain an explicit forward world model. A world-action model adds a connection between predicted future structure and executable actions. These categories overlap, and author terminology is not uniform. For this review the operative question is always: **what is observed, what is predicted, and how is the prediction grounded in execution?**

RT-2 represents robotic actions in a token interface and combines robot learning with prior visual-language knowledge. Its transfer result concerns recognized concepts affecting robot behavior; it does not show that arbitrary symbols can be grounded without aligned demonstrations. [W15](https://arxiv.org/abs/2307.15818v1)

OpenVLA provides an open 7B VLA trained on approximately 970,000 robot demonstrations. Its openness is useful for investigating learned representations and adaptation. The demonstrations are a substantial source of grounding that Voynich lacks. Reusing the checkpoint supplies visual-language and manipulation priors, not historical knowledge of the manuscript's encoding. [W16](https://arxiv.org/abs/2406.09246v3)

LAPA makes the bridge especially explicit: learn latent actions from frame pairs, train a VLM to predict them from observations and task descriptions, then fine-tune with a small set of trajectories containing actual robot action labels. It reduces a particular action-label requirement during pretraining; it does not remove the eventual alignment requirement. Its unsupervised stage also does not mean absence of all language conditioning in the pipeline. [W14](https://arxiv.org/html/2410.11758v1)

π₀ combines a pretrained VLM with a continuous action expert using flow matching, generating coordinated action chunks. The inspected original paper reports about 10,000 hours of robot demonstrations, supplemented by open robot data. This is a substantial behavioral dataset, not evidence that a generic language model can infer arbitrary motor or linguistic actions from a few examples. [W17](https://arxiv.org/html/2410.24164v1)

π₀.₅ expands the co-training recipe across robot data, web information, and semantic subtask supervision. Its inference factorization first predicts a high-level subtask and then conditions lower-level actions on it. The result motivates heterogeneous supervision and hierarchical interfaces, while supplying no direct validation of unknown-language decipherment. [W18](https://arxiv.org/html/2504.16054v1)

Three fresh preprints show where the design space is moving, with appropriately lower evidential weight here. LaWAM uses predicted latent visual subgoals instead of rendering full future video; MoWAM proposes explicit future motion as a compact representation supporting action selection; ZimaBlue proposes scalable video pretraining and a slow/fast architecture for robot control. Their abstracts were verified; their evaluation details and code were not audited in this pass. LAWM, a separate 2025 work, proposes latent-action pretraining through world modeling. Similar names do not make these the same method. [W19](https://arxiv.org/abs/2606.15768v1), [W20](https://arxiv.org/abs/2609.20709v1), [W21](https://arxiv.org/abs/2609.00188v1), [W22](https://arxiv.org/abs/2509.18428v1)

The deep transferable idea is **heterogeneous evidence tied to a common executable structure**. Some external data can teach visual relations; some can teach language realization; some can teach procedural state changes; some can teach unknown-script inference. A task-specific head connects them through a constrained representation. This is a much stronger proposal than feeding a page into a robot model and asking what it means.

For linguistics, the equivalent of an action is not automatically a motor command. It could be an executable semantic operation, a grammatical derivation, a channel transformation, or an inference edit. These inhabit different spaces and need different supervision. Flow matching over continuous motor trajectories is well matched to physical smoothness. Cipher mappings and grammatical structures are often discrete and permutation-sensitive. A continuous latent relaxation may help search, but it must decode to valid structures and preserve alternatives; averaging two incompatible alphabets is not a meaningful partial decipherment.

## 7. Passive data and the missing-intervention problem

The following is our formalization of the difficulty, not a theorem claimed by a robotics paper. A passive sequence observes a mixture over unobserved actions:

`p(x_next | history) = sum_a p(x_next | history, a) p(a | history)`.

Without further restrictions, many action models and action-selection distributions produce the same mixture. Calling the unknown factor a “latent action” does not make the factorization unique. If a code learns camera movement, stylistic variation, or a change of topic, predicting it may be useful while interpreting it as a causal operation is wrong.

For a text-producing system, an even simpler ambiguity is the relabeling of latent meanings. If a transformation is applied to all internal meanings and the renderer compensates, the observations can remain unchanged. Cross-modal evidence may reduce this freedom, but only to the extent that it independently constrains the same variables. A label attached to an ambiguous illustration may still admit many meanings. A diagram relation that recurs systematically across independent pages is potentially stronger than a guessed species name, but it remains an empirical question whether the text encodes it.

Observation can support causal discovery under substantive assumptions; we should not claim a blanket impossibility. The practical obligation is to state those assumptions. For example: are multiple views conditionally independent given a shared referent? Are linguistic and visual rendering mechanisms stable across scribes? Are operator effects invariant across contexts? Is the allowed channel family sufficiently restricted? Does the data cover the states needed to distinguish two proposals? Which facts come from external historical knowledge rather than the manuscript itself?

There are genuine actions available to us: choose a scan region for expert examination, compare an independent transcription, search an external catalog, or run a mechanically specified candidate decoder. There are also interventions on a trained model's activations. Neither type is an intervention on the fifteenth-century source process. The former can reveal additional evidence; the latter can establish how the model computes. Their value is real, but their evidential targets differ.

## 8. Four ambitious architectures worth serious consideration

These are proposed research architectures. They are not an implementation plan, a sequence of small experiments, or a promise that a particular design will decipher Voynich.

### A. A foundation inverse-world model for writing systems

**Question.** Can a system infer an unfamiliar writer/channel mechanism while keeping it separate from language and meaning?

**Inputs.** Manuscript image regions and an uncertainty-preserving transcription lattice; page geometry; uncertain scribe and section metadata; diverse external writing systems, languages, and reversible/irreversible channel families with provenance. The model should be able to operate without image-derived semantic labels.

**Internal interface.** A hierarchical latent description contains a language/grammar hypothesis `G`, channel program `C`, document-level discourse state `D`, and local writer state `S`. A renderer maps their consequences to predicted observations. An inference network proposes distributions over the description from context; an explicit executor checks whether a proposed channel actually produces the observed symbols. A neural state-space backbone amortizes this otherwise expensive inverse problem.

**Inference story.** Broad priors propose mechanisms, then document-specific inference concentrates on mechanisms that explain repeated structure across the manuscript under a compact shared description. A change to a channel rule propagates through every occurrence it governs. The system can return an equivalence class of mechanisms when several fit equally well. This is a proposed structured-inference design; merely naming a neural output a posterior does not guarantee calibrated uncertainty. It should be able to infer “this is a productive state-dependent code with unresolved plaintext labels” without pretending it has recovered semantic content.

**Why it might work.** Large-scale pretraining can teach reusable inference operations across languages, scripts, and channels, rather than memorizing a fixed key. Hierarchical sharing can pool sparse evidence. An executor limits the ability to repair every inconvenient token with an ad hoc explanation.

**Why it might fail.** The true mechanism may be absent from the prior; powerful latent programs may explain everything; historical preprocessing may destroy the distinctions needed for inference; language and channel complexity may trade off without a unique optimum. Synthetic breadth alone cannot establish coverage of the actual historical process. The important deliverable would be a constrained generative explanation with explicit unresolved degrees of freedom.

### B. A multimodal referent-and-procedure world model

**Question.** Can relations that recur across images and text create enough independent structure to anchor anonymous semantic variables?

**Inputs.** Region-level images, text neighborhoods, diagram topology, counts, and uncertain correspondences. External material would include historically relevant illustrated texts with readable language and documented relationships between images and descriptions. Modern photographs could support generic part recognition, but would not be treated as direct medieval semantic supervision.

**Internal interface.** Object/part slots feed a relation graph. A discourse model maintains candidate entities, coreference links, and propositions. A procedural model represents typed transitions only where the document supports a process interpretation. Separate rendering heads generate text, diagram relationships, and visual-part evidence. A channel model relates latent linguistic expressions to manuscript glyphs.

**Inference story.** The system searches for shared anonymous structures before attaching names. For example, it might propose that a textual construction relates a repeated entity to one of its parts, or that labels around a diagram correspond to an ordered set. A later lexical anchor would have to agree with these previously constrained relations. Alternative image readings remain in the posterior.

**Why it might work.** Meaning is relational. Several weak correspondences can jointly constrain a system when they recur with shared rules. Composition could let a rare expression inherit constraints from familiar components. This is the strongest route in this chapter toward semantic decipherment rather than solely channel recovery.

**Why it might fail.** Illustrations may be schematic, composite, copied from unrelated sources, symbolic, or weakly aligned with nearby text. A large multimodal model may impose modern botanical or procedural categories. Diagram position may be explained by layout conventions alone. If every image receives a different bespoke caption, the apparent cross-modal explanation has no restrictive force.

### C. A world-action inference agent with an external verifier

**Question.** Can a solver learn to navigate the enormous space of candidate grammars, channels, alignments, and sources more intelligently than static reranking?

**Inputs.** A versioned evidence store, competing generative models, uncertainty summaries, allowable evidence queries, and an explicit accounting of computational and annotation cost. Its state is our knowledge about the manuscript, not the manuscript's alleged physical world.

**Action interface.** Typed actions include proposing a global symbol-class merge, changing an alignment, adding a finite-state channel operator, retrieving a historical comparator, or requesting a particular ambiguous glyph judgment. Each action has a scope, preconditions, reversible state change, and verifiable consequence. The agent cannot simply overwrite an inconvenient observation.

**Inference story.** A policy proposes multi-step analysis plans; a forward model anticipates whether an action will distinguish leading hypotheses or merely improve local fit. A diffusion-style planner could propose coherent sets of edits, while an executor enforces consistency. Search values should derive from independently computed evidence and information gain, with linguistic fluency treated as a prior rather than the only reward.

**Why it might work.** The real obstacle may be coordinated global search. Many useful discoveries require temporarily accepting a worse local explanation to reveal a better shared rule. An agent trained on a broad collection of inverse problems could learn which observations settle which ambiguities and when to abstain.

**Why it might fail.** Reward misspecification could train a hypothesis beautifier. A learned value function may be confident far outside its training distribution. Repeated adaptive inspection can consume the evidence intended to test final claims. The planner's success therefore remains separate from success of any historical reading it discovers.

### D. A joint posterior model with interpretable causal modules

**Question.** Can we retain the global coordination of a large multimodal model while making the candidate explanation inspectable and causally testable?

**Inputs and state.** Reuse the evidence of A and B. Maintain a structured hypothesis graph containing proposed glyph classes, segmentations, lexical forms, grammar operations, referents, channel rules, and uncertainty. Give each type a separate interface. A stochastic generative proposer updates many linked variables jointly; a world model scores their predicted consequences over the whole document.

**Inference story.** Diffusion or flow does not “clean” the observed text into an assumed plaintext. Instead it proposes complete candidate explanations under explicit constraints. A selected mapping change must propagate to every affected context, and a proposed referent change must alter the corresponding semantic predictions. The system can retain competing complete readings instead of mixing fragments from mutually inconsistent worlds.

**Mechanistic target.** Establish which variables and computational pathways implement state updating, role binding, channel selection, and uncertainty. Intervention specifications should predict several downstream consequences and several invariances. If changing a candidate grammatical role alters only a final verbal answer, that is weaker than changing the independently scored consequences that role is supposed to govern.

**Why it might work.** Separate interfaces make cross-modal evidence and large-scale learned priors usable without making every conclusion a black-box statement. Global proposals address dependencies that tokenwise decoding misses. Mechanistic analysis could reveal whether the model maintains a stable channel mechanism or substitutes context-specific shortcuts.

**Why it might fail.** Modular interfaces can be bypassed; apparent causal variables may be implementation conveniences rather than historical factors; latent proposals can be sharply multimodal and hard to sample. An interpretable computation can still implement a false hypothesis. The external observation model and evidence remain indispensable.

## 9. What “big” should mean here

The credible scale-up is a foundation model of **inference across writing systems and evidence types**, coupled to a historically constrained manuscript model. Its scale lies in external diversity, structured hypothesis search, multimodal alignment, and explicit explanation capacity. It need not mean pretraining a billion-parameter video generator from the few hundred surviving pages.

There are at least three materially different resource regimes: reusing existing visual/language backbones, training adapters and structured inference modules, and pretraining a new world-model foundation system. The last is an institutional-scale proposition compared with this repository's local hardware. None of the cited robot successes supplies a credible dollar estimate for our new task. A future implementation decision would require measured model/data sizes, access and licensing checks, hardware throughput, and a finite cost envelope. No training allocation or expenditure is proposed by this document.

The hardest missing resource may be better aligned evidence rather than more compute: readable historical comparators with images, credible text-region correspondences, multiple transcriptions with manuscript coordinates, and expert judgments that carry explicit uncertainty. These resources could change which latent variables are identifiable. Millions of generated examples can train a solver to use an anchor; they cannot manufacture an authentic anchor for Voynich.

## 10. Prior Voynich applications and the limits of this review

A bounded live search on 2026-09-21 used combinations of “Voynich” with “world model,” “JEPA,” “VLA,” and “diffusion.” It did not locate a substantive, evaluated application of learned action-conditioned world modeling to Voynich decipherment. That is a search outcome, not a universal novelty claim.

It did locate a public Voynich AI platform proposing diffusion glyph morphing, CLIP alignment, synthetic decipherment pretraining, and related approaches; its changelog describes research specifications. These are relevant precedents at the level of proposed methods. We did not inspect evidence that the proposals achieved decipherment or validate the site's novelty, feasibility, glyph-count, or historical assertions. Morphological proximity under a learned image transformation would not, by itself, identify ancestry or linguistic value. [W23](https://voynich-ai-production.up.railway.app/research)

The immediate intellectual conclusion is therefore affirmative but specific: world models and world-action methods open a credible, ambitious architectural direction when they model **how structured meanings and writing mechanisms generate the evidence**, and when the inference agent has a separate, honest action space. A conventional robot or video checkpoint does not bridge the gap automatically. The promising research program is the construction of that bridge, including the linguistic and historical constraints that make its outputs distinguishable from well-coordinated fiction.
