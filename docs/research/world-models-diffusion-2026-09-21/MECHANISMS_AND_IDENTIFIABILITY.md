# Mechanistic interpretation of a decipherment world model

Research and architectural ideation, 2026-09-21. Nothing proposed here was implemented or run. `M` citations resolve in [the source ledger](MECHANISM_SOURCES.md). Source reports are attributed; architectural deductions and examples are ours. This chapter complements the [world/action-model review](WORLD_MODELS_ACTION.md), [diffusion review](DIFFUSION_INFERENCE.md), and [linguistics review](LINGUISTICS_GROUNDING.md).

## The ambitious target: extract an executable account of what the model knows

Mechanistic interpretation could be part of the decipherment engine itself. A successful learner might internally represent an entity being discussed, an abbreviation rule, a pending grammatical dependency, a copy source, or a changing encoding state. If those variables can be isolated, their update rules could become a compact candidate decoder. This is substantially more useful than finding a neuron whose favorite strings resemble plant names.

There are three separate claims to establish:

1. **Behavioral competence:** the learner can recover unfamiliar systems in settings where the answer is independently known.
2. **Computational explanation:** an extracted representation and transition rule explain how that learner works, including responses to controlled changes.
3. **Historical explanation:** those rules explain the manuscript, rather than merely one model trained on it.

The first two can be established in constructed or solved systems. The third requires manuscript evidence and independent anchors. None entails the next automatically. Their separation is what makes an ambitious program scientifically productive: even a failure of the historical hypothesis can leave behind a useful universal decipherment model or a stronger mechanism-extraction method.

## What the Othello line of work actually establishes

**Source report.** Li and colleagues trained a sequence model on moves and recovered information about board state using probes and interventions. Their synthetic training set contains 20 million games. The board and rules were available to the researchers for evaluation; probe training and intervention targets used those known states. The intervention procedure also modified successive layers, rather than demonstrating that a single untouched latent coordinate universally controlled the game. [M01](https://arxiv.org/html/2210.13382v4)

**Source report.** Nanda, Lee, and Wattenberg found a more revealing coordinate system: ownership relative to the current player, rather than absolute black/white color. This made relevant board representations linearly accessible and enabled simpler interventions. [M02](https://aclanthology.org/2023.blackboxnlp-1.2/)

**Our inference.** A failed probe for a guessed Voynich category does not imply that no useful representation exists. The wrong reference frame can hide a simple mechanism. Candidate frames could include “same as the last occurrence,” “relative to the current paragraph,” “current versus alternative referent,” or “position within the writer's emission cycle.” These are search hypotheses, not evidence that the manuscript uses such devices.

Othello is an existence proof for extracting structured information from symbolic sequences under favorable conditions. It is not evidence that a similarly sized network trained on one short unknown manuscript will recover a language. The observation budget, diversity of state coverage, and available truth labels are radically different. The appropriate scale-up is to train across many independently generated or solved symbolic worlds, then attempt inference on a new one.

## Recovering a world means recovering distinctions and equivalences

**Source report.** Vafa and colleagues evaluate whether different histories reaching the same state support the same continuations, and whether different states can be distinguished by suitable continuations. Their deterministic-automaton framework exposes failures missed by next-token legality and state probes. Computing the proposed truth-based metrics assumes access to the underlying automaton. [M03](https://arxiv.org/html/2406.03689v1)

**Our adaptation.** A decipherment world model needs both abilities:

| Requirement | Linguistic/channel example | Consequence if missing |
| --- | --- | --- |
| Merge equivalent histories | Two spellings or paraphrases leave the same referent and discourse state | Memorizes surface form instead of meaning |
| Preserve important distinctions | Two prefixes predict the same immediate glyph but different later agreement or encoding states | A convincing local predictor hides a wrong mechanism |
| Update consistently | Reading the same next observation transforms equivalent beliefs equivalently | Clusters have no usable transition rule |
| Preserve uncertainty | Several entities or keys remain compatible with the observations | Invents premature semantic certainty |

For an unknown manuscript, we cannot declare two passages semantically equivalent because our model says so and then use that declaration to validate the model. Truth-based equivalence belongs first to solved cases; on Voynich it becomes a conditional prediction to be checked through evidence not used to invent the equivalence.

A probabilistic extension should compare distributions of joint continuations, not only accepted strings or separate future-position marginals. A state can preserve the probability of each individual future token while destroying dependencies between them. “Good future prediction” needs a stated horizon and a stated joint distribution.

## Three state spaces, three kinds of intervention

Our proposed architecture has distinct state spaces:

- **Referent state:** objects, parts, properties, relations, procedures, or a diagrammatic system being discussed.
- **Writer/channel state:** current discourse plan, language realization, abbreviation conventions, glyph choices, layout and possible encoding mechanism.
- **Solver belief state:** alternative explanations and uncertainty over those hidden variables.

Changing an abbreviation rule in a simulator is a causal intervention on a known writer process. Patching an activation is an intervention on a modern neural model. Changing a proposed reading is an action in the solver's search process. These are different experiments with different conclusions.

We have no ability to intervene on the historical author. We can inspect additional observations, commission independent annotations, or measure another transcription. Those can distinguish explanations, but calling them medieval interventions would overstate the evidence.

**Source report.** Causal representation learning studies how causal variables might be recovered from low-level data, emphasizing structure, changes of environment, and additional assumptions beyond ordinary correlations. Richens and Everitt show a relationship between robust policies and recoverable causal models under a rich intervention family; their stated conditions include an unmediated decision setting and finite-dimensional variables. The result does not establish that passive next-token training on one document identifies its hidden cause. [M06](https://arxiv.org/html/2102.11107v1), [M07](https://arxiv.org/html/2402.10877v2)

The productive use of these ideas is architectural: deliberately expose many independently varied mechanisms during external training, so invariances become learnable. Keep the much weaker manuscript inference claim separate.

## A causal atlas of language and writing systems

**Proposed research direction.** Build a learner whose mechanistic organization can be compared across independently trained worlds. Instead of asking whether two models have similar-looking features, ask whether they implement the same *counterfactual operation*.

Consider a constructed document about objects with detachable parts. A change to the described part should alter the semantic role, the appropriate noun or bound morpheme, and the corresponding image region. A change to the scribal hand should alter the marks while preserving those content relations. A change to a cipher key should alter the entire family of affected spellings while leaving object and event structure intact. The intervention's intended effects and invariants are specified by the construction, not by a language model judging its own prose.

The atlas would contain behavioral signatures for:

| Hypothesized variable | Counterfactual signature | Important competing explanation |
| --- | --- | --- |
| Referent identity | Changes later references and linked visual entity; preserves unrelated entities | Topic bias or cached lexical association |
| Morphological feature | Changes compatible forms across multiple lexical roots | Memorized suffix table or line position |
| Pending argument | Changes which role can be completed next | Local collocation alone |
| Encoding state | Changes emissions under the same latent message | Different underlying language or scribe |
| Copy source | Changes copied descendants according to a source pointer | Semantic repetition |
| Null-production state | Changes assigned surplus marks without changing recovered message | Decoder deleting all difficult evidence |
| Belief uncertainty | Preserves multiple predictions until discriminating evidence arrives | Diffuse logits unrelated to epistemic uncertainty |

The atlas is not a classifier that declares “this resembles a morphology circuit, therefore Voynich is linguistic.” Similar computation can support different tasks. Its role is to generate *sharper candidate explanations* and identify which next observations separate them.

An ambitious endpoint is an interpretable virtual machine: learned neural inference proposes a small typed state-transition program; the program independently reconstructs or predicts observations with an explicit alignment and exception budget. The inferred program should survive symbol renaming and transfer to unseen documents under the same rules. Its semantic names remain hypotheses until grounded.

## Causal abstraction supplies the interface

**Source report.** Geiger and colleagues formalize relations between detailed computations and higher-level causal models. Interchange interventions provide a way to ask whether replacing an internal variable has the effect predicted by a proposed abstraction. This formalism can describe distributed variables rather than demanding one neuron per concept. [M05](https://www.jmlr.org/papers/v26/23-0058.html)

**Our design.** Suppose a high-level interpreter has state `s`, update `F`, and observation model `G`; the neural world model has hidden state `h` and update `f`. An extraction map `phi` should satisfy two independent requirements on unseen cases:

`phi(f(h, observation)) approximately equals F(phi(h), observation)`

`predicted observations from G(phi(h)) approximately match the neural model's joint predictions`

Interventions add a third requirement: replacing one high-level variable and its aligned low-level representation should yield matching downstream changes while preserving nuisance behavior. Neither `phi` nor `F` should be defined so that this agreement holds tautologically. A very flexible extracted program can simply memorize the teacher.

Comparisons should concern equivalence classes of computations. Neuron indices and arbitrary rotations are not stable semantic identifiers. A transition system recovered up to permutation may be entirely adequate for structural recovery. A literal translation additionally needs a mapping to independently grounded meanings.

The promise is substantial: the model becomes an engine for discovering an algorithm that can subsequently be inspected without trusting every neural activation. The limitation is equally concrete: a faithful explanation of the model can still faithfully explain a wrong model.

## Interpretation of diffusion has an additional time axis

A diffusion-based decipherment system has at least three axes: manuscript position, neural layer, and denoising step. A dynamical world model adds event time; a search agent adds decision time. These axes must not be collapsed into one notion of “temporal understanding.”

**Source report.** DLM-Scope studies SAEs for Dream-7B and LLaDA-8B. It distinguishes features learned from masked versus unmasked token positions and interventions applied repeatedly during denoising. These are useful precedents for inspecting a text-diffusion solver. Its steering results concern modern-language models and selected features, not discovery of an unknown grammar. [M10](https://arxiv.org/html/2602.05859v1)

**Source report.** Tinaz, Fabian, and Soltanolkotabi study concepts and interventions in Stable Diffusion v1.4, reporting different opportunities for composition and style control across diffusion stages. Their concept labeling uses other vision systems; separately fitted timestep dictionaries are not automatically aligned feature identities. [M11](https://arxiv.org/html/2504.15473v1)

**Our extension.** Inspect when a candidate solver commits to its language family, channel operation, alignment, and semantic graph. A useful discovery would be that global channel information stabilizes before local lexical choices, or that a late contradiction actually revises a previously favored key. The opposite outcome—semantic narrative fixed early and evidence subsequently forced into it—would diagnose a system that rationalizes its prior.

For this purpose, retain immutable observations in a separate evidence stream. Record every inferred deletion, expansion and reassignment. Intervene on an internal *hypothesis*, then ask whether later evidence corrects it. Compare a single intervention with sustained forcing. A variable that only works while continually forced may be a control input to generation, not a naturally maintained belief.

For paired trajectory comparisons, hold denoising randomness fixed where possible. Causal claims also require repeated comparisons that quantify sampling variation; randomized comparisons can establish an effect without identical noise. Matching noise alone does not make an off-distribution intervention historically meaningful.

## Identifiability gives us constructive design requirements

**Source report.** Zhang, Chen, and Chen study latent recovery under invertible generation, Boolean representations, a rich multi-task setting, and a particular low-degree complexity bias. Their conclusions are conditional and include representation symmetries. They are not a theorem that a smaller network necessarily recovers meaning. [M04](https://arxiv.org/html/2502.09297v2)

**Our use.** Design external training worlds with multiple tasks that require the same latent structure: understand a description, render a diagram, predict an action outcome, recover a writer rule, and answer a counterfactual query. A representation explaining all those views economically has a stronger reason to reflect the shared structure than one predicting characters alone. Whether this generalizes to an unknown manuscript remains a research question.

**Source report.** Work on content/style separation and multimodal contrastive learning supplies conditional identification results for shared latent factors. Assumptions about invariant content, changing nuisance factors, mixing functions and paired observations matter. Recovering a shared subspace up to an invertible transformation does not name its coordinates. [M08](https://proceedings.neurips.cc/paper/2021/hash/8929c70f8d710e412d38da624b21c3c8-Abstract.html), [M09](https://arxiv.org/abs/2303.09166)

**Our consequence.** Re-rendering a synthetic message under many keys provides controlled invariance. Cropping, swapping, or altering real Voynich drawings does not automatically preserve content. A leaf shape might be diagnostic rather than decorative; a single stroke might distinguish graphemes. The augmentation policy itself can silently decide the answer. The ambitious solution is explicit uncertainty and separately modeled channels, not more aggressive indiscriminate augmentation.

Two glyph-level models can be observationally equivalent while assigning different semantic labels. Even perfect optimization then leaves a family of answers. Multiple independently generated views can reduce that family; they cannot be assumed to eliminate it. A promising system should expose the remaining ambiguity and identify which external fact would distinguish the alternatives.

## Program induction, posterior diversity, and the searcher's world model

**Source report.** DreamCoder combines program search, learned recognition, and reusable library construction. Its tasks have specifications such as input/output examples and its search starts from supplied primitives. A manuscript supplies neither a ready-made evaluator for intended meaning nor a guaranteed correct primitive library. [M12](https://www.neurosymbolic.org/papers/EllisWNSMHCST21.pdf)

**Our synthesis.** Use such machinery to propose bounded writer programs: explicit substitution, abbreviation, segmentation, copying, and limited state transitions. Let an independent executor calculate what the proposed program explains. The neural proposer may be enormous; the final account should be compact enough for another scholar to apply.

**Source report.** GFlowNets target diverse composite objects with sampling proportional to a specified nonnegative reward under their training conditions. This is relevant to maintaining several plausible programs rather than optimizing one story. Correctness is relative to the supplied reward and successful learning. [M13](https://jmlr.org/papers/volume24/22-0364/22-0364.pdf)

**Our synthesis.** A diffusion proposer, a GFlowNet over channel programs, and conventional search are competing inference tools, not mutually exclusive historical hypotheses. They can target the same joint model. Calling the reward a “posterior” is justified only if it represents a documented probabilistic target and the inference approximation is evaluated. An arbitrary weighted sum of fluency, visual similarity and researcher preference is a heuristic score.

**Source report.** Bayesian inverse planning infers goals and beliefs by inverting a model of action choice under environmental and rationality assumptions. [M14](https://web.mit.edu/9.s915/www/classes/cognition2009.pdf)

**Our proposed high-risk use.** Treat a historical writer as communicating instructions to a competent reader under costs of writing, ambiguity and memory. Reader success could constrain candidate encodings. But cost functions are hypotheses: aesthetic copying, ritual, deception, error and incomplete knowledge remain possible. A rational-reader prior is useful only if it cannot explain every observed text by retuning the imagined reader.

The solver's own world model is less historically speculative. Its actions can be “inspect an ambiguous glyph,” “request an independent motif judgment,” or “test whether a proposed rule explains all its occurrences.” It should forecast how the possible answers would change competing hypotheses. That is a concrete VLA-inspired perception–action loop even when the actions are intellectual operations rather than robot movements.

## What would make this a major research contribution

The target is a reusable system that infers unfamiliar writing processes, reports ambiguity, extracts an executable explanation and relates it to independent evidence. Four substantial outputs would matter even before a full Voynich reading:

1. A foundation model for inference across unknown linguistic and channel systems, with demonstrated transfer across entire families rather than new keys alone.
2. A compiler that translates a learned hypothesis into inspectable rules with complete observation alignment.
3. A causal atlas distinguishing linguistic, referential, channel and copying mechanisms by interventions and transferred behavior.
4. A manuscript explanation whose independent predictions survive alternative transcriptions, visual confounds and competing nonsemantic accounts.

Simulation-based inference provides a framework for learning inference from generated worlds, but the simulator determines what worlds the inference system can know. Calibration under that simulator checks the inference procedure; it does not validate the simulator's applicability to Voynich. [M15](https://arxiv.org/html/1911.01429v3), [M16](https://arxiv.org/abs/1804.06788)

The research ambition is therefore not “interpret a network and hope it speaks Latin.” It is to build a system whose inferred mechanisms can be compiled, challenged, transferred and grounded. The exact semantic labels may be the last part recovered, while structural and procedural relations supply the constraints that make those labels recoverable.
