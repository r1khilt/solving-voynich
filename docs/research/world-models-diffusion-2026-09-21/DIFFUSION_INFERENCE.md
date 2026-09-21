# Diffusion as an inference engine for an unknown manuscript

Research and architectural ideation, 2026-09-21. **No implementation, training, decoding run, or new manuscript result.** Source IDs resolve in [DIFFUSION_SOURCES.md](DIFFUSION_SOURCES.md). Literature summaries below are separated from proposed extensions. This chapter complements the world-model and linguistics reviews; it does not assume that Voynich encodes a natural language or that its illustrations identify particular plants.

## The opportunity is joint revision, not denoising a mysterious alphabet

The strongest diffusion proposal is to maintain many competing, globally consistent explanations of the manuscript and revise their plaintext, alignment, encoding rules, and latent state together. It is not to feed the transcription into an image generator, ask for Latin, and interpret the output as recovered evidence. A capable generative prior can produce convincing language under an incorrect channel. The scientific question is whether observed glyphs and independent evidence constrain the explanation tightly enough that a solver must converge on substantially the same rules.

Diffusion can supply a learned search distribution over complete hypotheses. World models can supply predictions about how a proposed latent process generates sequences and responds to interventions. Explicit channel models can enforce correspondence with the actual marks. Linguistic models can assign probabilities to morphology, syntax, discourse, and historical variation. These components have different jobs. Combining them is promising precisely because no component has to impersonate the others.

Here is the consequential distinction. A manuscript likelihood measures how well a proposed process predicts the observed artifact. A language prior measures how typical a hypothetical plaintext is. A sampler determines which explanations are explored. A verifier determines whether their claimed relationships actually hold. Improving the sampler does not automatically improve the truth of the generative assumptions; improving fluency does not automatically improve evidence fit.

The research direction worth taking seriously is therefore **diffusion-assisted, uncertainty-preserving inverse program inference**, with an explicit forward account of writing. That is a considerably larger undertaking than a larger next-token predictor. It also creates concrete points at which a beautiful but false explanation can be rejected.

## What the diffusion families actually provide

### Continuous diffusion and scores

The original diffusion construction learns a reversal of an artificial corruption process; DDPM makes this practical with a noise-prediction objective and repeated denoising. A score model estimates the gradient of log density for noisy data, and the score-SDE framework relates stochastic reversal, probability-flow ODEs, and conditional generation. These are different routes to learning and sampling a probability distribution. Their time coordinate is a computational noise level, not necessarily physical or historical time. [D01](https://arxiv.org/abs/1503.03585v8), [D02](https://arxiv.org/html/2006.11239v2), [D03](https://arxiv.org/html/2011.13456v2)

For our proposed use, continuous states could represent uncertain scene configurations, vectors of morphological features, or soft assignments within a search procedure. They should not be treated as exact symbolic answers. A continuous vector near the embedding of a plausible word is not proof that the observed glyphs encode that word. A discrete realization and its complete correspondence to the manuscript are still required.

Diffusion-LM explicitly confronts this issue: it learns continuous word embeddings and introduces a rounding mechanism back to tokens; its inspected method discussion reports that naive rounding does not reliably commit to individual words. Its value here is the possibility of differentiable control over intermediate representations. The rounding problem is a reason to keep a symbolic verifier outside that representation, especially when a single letter or boundary changes the claimed decoding rule. [D05](https://arxiv.org/html/2205.14217v1)

### Discrete diffusion and masked language modeling

D3PM defines categorical corruption through transition matrices, including uniform and absorbing-mask processes. It makes the corruption geometry an explicit design choice. Its experiments also supply a useful negative result: embedding-neighbor transitions did not automatically improve text generation. Similar-looking or semantically nearby symbols are not guaranteed to define a good diffusion neighborhood. [D04](https://arxiv.org/html/2107.03006v3)

SEDD learns discrete probability ratios using score entropy; MDLM derives a practical masked-diffusion objective as a weighted collection of masked-language losses. These provide principled distributional training beyond simply running a masked encoder repeatedly. Their likelihood-related objectives are bounds with assumptions and estimators, not interchangeable with exact autoregressive likelihoods. For scientific model comparison, the objective, estimator, and uncertainty must remain explicit. [D06](https://arxiv.org/html/2310.16834v3), [D07](https://arxiv.org/html/2406.07524v2)

The useful distinction for decipherment is between an **unknown value** and an **unknown position**. A masked-token model naturally represents uncertainty about which symbol occupies a specified slot. It does not, by itself, represent uncertainty about whether the slot exists, whether two observed glyphs jointly encode one morpheme, or whether a delimiter belongs between them. Those are alignment and structure variables, not ordinary vocabulary choices.

LLaDA establishes that masked diffusion can support large-language-model capabilities after substantial pretraining: the inspected report describes an 8B model trained on 2.3 trillion tokens and 0.13 million H800 GPU hours. Dream 7B instead adapts an autoregressive initialization, retains a shifted prediction relationship, and uses context-adaptive token-level loss weighting; its additional corpus comprises 580 billion tokens. These are evidence for viable large-scale model classes. Neither resource scale is evidence that a manuscript-sized corpus can teach language, semantics, and an unknown writing system from scratch. [D08](https://arxiv.org/html/2502.09992v3), [D09](https://arxiv.org/html/2508.15487v1)

A pretrained diffusion LM could be an external plaintext prior or proposal generator. It should not silently become a judge of historical truth. Its tokenizer, language distribution, modern spelling conventions, and instruction tuning are inherited assumptions. A character- or grapheme-level channel can be kept independent of that tokenizer, allowing the same candidate to be evaluated under several plaintext priors without changing the observed evidence.

### Revision is not automatic

A particularly important correction to the usual sales pitch is that standard absorbing-mask samplers often freeze a token after revealing it. They permit arbitrary-order filling, but this is not the same as repeatedly correcting committed mistakes. ReMDM introduces an explicit remasking construction, preserves the relevant masked marginals, and studies schedules that allow revision and additional inference computation. The inspected version is v4, February 2026; a search result incorrectly associated one older attachment with ICLR 2025, whereas the current arXiv record says NeurIPS 2025. [D10](https://arxiv.org/html/2503.00307v4)

Our proposed solver needs revision of **linked assignments**. If one key entry changes, all its dependent glyph occurrences must change together. Independently remasking fifty plaintext positions can repeatedly rediscover the same inconsistency. A global key variable should control those positions, or a proposal should jointly update its dependency set. Likewise, a change in segmentation should revise the affected alignment and morphology rather than leave incompatible old decisions in place.

This is a reason to diffuse over structured hypotheses, not merely printed plaintext. Some variables can use continuous proposals; others require categorical changes, permutations, tree edits, or explicit finite-state transitions. A mixture of proposal types is scientifically cleaner than forcing every uncertainty into one token sequence. The benefit to seek is coordinated movement between explanations separated by many local edits.

### Variable length is central, not an optimization detail

Johnson and colleagues extended denoising models to insertion and deletion, demonstrating arithmetic-sequence and spelling-correction applications. Edit Flows later formulates insertion, deletion, and substitution as continuous-time jumps over variable-length sequence space, using auxiliary alignments for tractable training. Its method separates operation rates from token distributions and includes ordinary token-wise and autoregressive constructions as restricted cases. This directly expands the space of representable hypotheses. [D11](https://arxiv.org/abs/2107.07675), [D12](https://arxiv.org/html/2506.09018v3)

For Voynich, an editable latent sequence could contain phonological segments, morphemes, abbreviations, or writing operations. An observed glyph sequence would remain fixed. Inserting a latent vowel would mean proposing that the channel omits it; deleting a latent token would mean changing a proposed plaintext, not erasing inconvenient ink. Each operation needs a probability or complexity cost and an explicit alignment explanation.

The attractive capability is to revise language and alignment jointly. The danger is unrestricted flexibility: if arbitrary words can be inserted, arbitrary glyphs ignored, and arbitrary many-to-many rules introduced, almost any readable passage can be made compatible. The architecture therefore needs a restricted operation vocabulary, shared rules across pages, an accounting of exceptions, and a meaningful comparison with nonsemantic processes. Variable length increases expressive power and the burden of identification at the same time.

## A formal target that preserves the evidence

The following is our proposed probabilistic specification, not a result established by the cited papers. Let C denote observed glyph evidence, V independently represented image/layout evidence, and M metadata. Let H choose a hypothesis family; K denote shared encoding rules; X plaintext or another latent message; A alignment and segmentation; S evolving channel state; W a possible world/procedural representation. One possible factorization is:

`p(H,K,X,A,S,W | C,V,M) ∝ p(H,K|M) p(W|H,M) p(X|W,H,M) p(A,S|X,K,H,M) p(C|X,A,S,K,H,M) p(V|W,H,M)`.

This is an architectural hypothesis. In particular, the final image factor asserts conditional structure that may fail if pictures are copied, decorative, symbolic, or only loosely related to text. Alternative factorizations should represent these cases. H must include structured nonsemantic production and hybrid possibilities; the variables X and W then require a different interpretation or may be absent.

The factorization makes several commitments visible. K is shared across a declared scope rather than freely selected for each word. A accounts for every observed glyph, including ambiguous readings and explicitly modeled nulls. S has a defined transition law rather than serving as an unlimited explanation. W does not acquire a plant name because a language model finds that story appealing. M may condition priors, but a historical guess cannot be counted again as independent observational evidence.

There are two legitimate ways to add diffusion. A conditional model can learn a proposal `qφ(H,K,X,A,S,W | C,V,M)` from controlled task families. Alternatively, a diffusion prior over some latent variables can be combined with explicit likelihood factors at inference time. The former risks amortization failure outside its training mechanisms; the latter makes likelihood evaluation and posterior correction difficult. Hybrid use is plausible: fast amortized proposals, explicit evidence scoring, and occasional structured transitions that escape the proposal model's preferred explanations.

In both cases the manuscript is conditioning information. Corrupting a copy of C during self-supervised training does not authorize replacing C at evaluation. Preserve original marks, alternative transcriptions, and their uncertainty. The latent variables may be edited. The evidence may be marginalized under a documented observation model. It must not drift toward the answer that the model finds easiest to generate.

### The noise process is not the historical channel

Diffusion training introduces a convenient path from data to masks or noise. A historical scribe may have used abbreviation, homophony, transposition, copying, or an ordinary writing system; none of those is established by successful artificial denoising. Even if a training corruption resembles a cipher, learning its reversal demonstrates competence only under that specified family and distribution.

Our proposed design should therefore name two channels separately: the **artificial training corruption** used to learn a proposal, and the **hypothesized writing process** whose output must match the manuscript. Their parameters, latent states, and time indices are different. Confusing them would turn a computational convenience into an unsupported historical theory.

## Constraints, posterior sampling, and the difference between valid and true

The July 2026 Dang–Ermon paper supplies an unusually relevant connection: combine a diffusion step's factorized token probabilities with an automaton and perform joint inference. Its exactness is local to the constrained mean-field distribution at that step, not the entire conditioned diffusion generative distribution. Deterministic automata implement the desired indicator constraint; nondeterministic automata can introduce accepting-path multiplicity weights. Its Sudoku constraint enforces format and givens, not all puzzle rules. [D23](https://arxiv.org/html/2607.07026v1)

Our proposed extension is to compile a candidate channel and observed glyph sequence into an acceptor or weighted transducer over possible latent strings. Neural probabilities propose plausible alternatives; the compiled structure imposes alignment and encoding consistency. For a fixed finite-state channel this can be tractable. Searching unknown keys and rule grammars can make the state space explode. Global substitution consistency, long-range copying, and hierarchical grammar cannot simply be declared cheap regular constraints.

A practical division of labor is consequently important. Use exact dynamic programming for the portion that really is tractable. Use explicit shared variables or a constraint solver for global requirements. Use diffusion for proposing difficult correlated changes. Maintain a final verifier that reruns the declared encoder from the candidate explanation and checks its correspondence to the observations. A verifier can establish consistency with a hypothesis, not that the hypothesis is historical fact.

There is also a distinction between **search** and **posterior sampling**. A guided search may deliberately prioritize attractive candidates without sampling any stated probability distribution. That can be useful, but its output frequencies are not posterior confidence. If a method claims calibrated uncertainty, its proposal probabilities, corrections, support, and convergence assumptions matter. Soft relaxation, greedy confidence schedules, projection to validity, and selecting the best of many samples generally change the distribution.

Twisted Diffusion Sampler offers a relevant statistical precedent: weighted particle simulation incorporates approximate guidance while retaining asymptotic correctness for its stated conditional target. That is a property of a particular algorithm under its assumptions, not a transferable guarantee for any pipeline containing diffusion and resampling. Its demonstrations concern conditional images and protein design, not unknown writing systems. [D20](https://arxiv.org/abs/2306.17775)

In our synthesis, particle populations would represent different language/channel/alignment explanations. Resampling would concentrate on evidence-supported regions, while rejuvenation moves would alter keys, segmentation, and rule structure. Such a design must guard against particle collapse: a hundred copies of the same early mistake do not represent a hundred independent explanations. Proposal support is especially important when external language priors heavily favor familiar languages or modern genres.

Bayesian notation does not solve the calibration problem either. A likelihood built from an arbitrary weighted sum of grammaticality, image similarity, and cipher consistency produces a scoring model until its factors and normalization are justified. A classifier score is not automatically a likelihood. Reporting a sharply peaked posterior under an unrealistic model would reveal the model's certainty, not the manuscript's meaning.

## How this connects to world models and action models

A world model and a diffusion model describe different axes of a system. “World model” concerns what is modeled: states, observations, transitions, and possibly actions. “Diffusion” concerns how a distribution is represented or sampled. A world model can use diffusion for observations or trajectories; a diffusion language model need not represent an external world at all. A shared neural architecture does not establish shared causal content.

Four precedents make this separation concrete. Diffuser models whole state/action trajectories and biases their generation with goals or rewards. Diffusion Policy models a conditional action distribution and executes it with receding-horizon control. DIAMOND predicts observations conditional on prior observations and actions, with separate reward/termination machinery. Diffusion Forcing trains causal sequence prediction with independently varied noise levels, allowing uncertain future tokens to coexist with cleaner context. These are not interchangeable modules. [D15](https://arxiv.org/html/2205.09991v2), [D16](https://arxiv.org/abs/2303.04137v5), [D17](https://arxiv.org/html/2405.12399v2), [D18](https://arxiv.org/html/2407.01392v4)

The inspected DIAMOND method conditions a next-observation denoiser on past observations/actions and trains its policy in imagined environments. Its relevance is not that Atari resembles Voynich. It demonstrates how a diffusion observation model can participate in a larger system with distinct state, action, and learning roles. For a manuscript solver, those roles would have to be supplied and justified afresh; a static page is not an action-labeled replay buffer. [D17](https://arxiv.org/html/2405.12399v2)

Diffusion Forcing's maze comparison is a useful narrow warning: the authors report that sampled Diffuser actions and sampled states can be inconsistent when actions are directly executed, and contrast that with using a separate controller. This does not invalidate trajectory diffusion generally. It highlights that a plausible joint sample and an executable causal account are separate achievements. Our counterpart is a candidate translation whose claimed writing rules actually regenerate its glyph evidence. [D18](https://arxiv.org/html/2407.01392v4)

The conceptual bridge to language is a latent event or procedure trajectory. Suppose a page describes a preparation: an ingredient is selected, transformed, combined, and administered. A world/procedure model could constrain which event sequences are possible; a language model could map those sequences to several linguistic realizations; an encoding channel could map a realization to the manuscript. This provides multiple levels at which evidence can disagree with a candidate. The example is a hypothetical research construction, not an interpretation of a Voynich passage.

Diffusion would be attractive for smoothing such trajectories because evidence can constrain both ends and intermediate steps. A late state may rule out an earlier action; a repeated label may tie distant passages together; an independently encoded diagram may constrain an event order. Autoregressive models can also participate in smoothing through external inference. Diffusion's attraction is a convenient global proposal mechanism, not an exclusive ability to reason backward.

There are three different meanings of action that should remain separate. An action can be part of the historical content of the manuscript, such as mixing ingredients. It can be part of a simulated writing process, such as choosing an abbreviation. Or it can be an epistemic action by a researcher, such as checking a glyph crop or seeking an independent source. Only the third produces a route to additional evidence now; the first two are latent hypotheses.

An ambitious solver could learn which external observation would most distinguish its leading hypotheses. If two candidate rules disagree on one damaged glyph or a particular repeated label, it could request a targeted paleographic review. That is a principled transfer from active world-model reasoning: act to reduce uncertainty among models. It should not fabricate a physical experiment on the past or treat an imagined action outcome as a newly observed fact.

### Flow matching, inverse problems, and learned combinatorial proposals

Flow Matching generalizes the training-path viewpoint through vector-field regression, while Discrete Flow Matching supplies analogous probability-path constructions for categorical data. The practical implication for our design is freedom to choose an inference geometry appropriate to the variables. There is no obligation to add Gaussian noise to arbitrary glyph identifiers or to treat edit distance as historical probability. A computationally convenient path is an engineering choice whose endpoint and conditioning still need scientific justification. [D13](https://arxiv.org/abs/2210.02747v2), [D14](https://arxiv.org/abs/2407.15595v2)

Diffusion Posterior Sampling explicitly describes an approximate method for noisy nonlinear inverse problems; Feynman–Kac steering uses interacting particles and intermediate potentials to steer text/image diffusion toward specified rewards. They motivate separating a learned prior from evidence or objectives, while also making it necessary to name what is approximate and what distribution is actually targeted. [D19](https://arxiv.org/abs/2209.14687v4), [D21](https://arxiv.org/abs/2501.06848v5)

DIFUSCO provides a complementary precedent: diffusion proposes binary solution structures for graph optimization tasks. Its relevance is learned search over combinatorial objects, not a proof that denoising finds the exact optimum. A decipherment system could similarly propose key assignments, alignment edges, or operation choices while a separate mechanism enforces validity. The task distribution and representation would need a new design. [D22](https://arxiv.org/abs/2302.08224v2)

Our synthesis is to treat these as a menu of inference tools with different contracts. A posterior sampler should preserve a stated target, a reward-steered generator should improve a stated reward, and an optimizer should find high-scoring feasible explanations. The same neural backbone might participate in all three, but the resulting uncertainties and claims are different. In particular, best-of-many optimization produces a selected candidate; it does not estimate the probability that the candidate is true.

### Separate three clocks and three kinds of uncertainty

A combined system has at least three clocks: manuscript sequence or event order, artificial denoising time, and the research process through which additional evidence arrives. A transition in one clock does not identify a transition in another. One hundred denoising steps do not imply one hundred scribal operations. Reordering computational inference does not imply that the manuscript should be read in that order. Updating a hypothesis after a new scan is ordinary evidence accumulation, not a reconstructed event in the text.

It also has at least three uncertainties. Observation uncertainty concerns the marks themselves. Model uncertainty concerns the language/channel/process family. Conditional latent uncertainty concerns alternative messages or alignments within one family. A method may sharply reduce the last while leaving the first two unresolved. Reporting a confident reading conditional on one assumed alphabet and language as unconditional confidence would hide most of the real problem.

The architecture should therefore retain uncertainty at the right scale. Multiple glyph readings can form a local observation lattice. Multiple alignments can share substructure. Multiple keys or grammars need distinct global hypotheses. Multiple semantic interpretations can remain unnamed relational graphs until independent anchors warrant labels. A single entropy score over plaintext tokens cannot adequately summarize all four.

This suggests a distinctive use of inference computation: allocate it to unresolved global alternatives and their discriminating consequences, not only to making a favored paragraph smoother. More samples are valuable when they reveal different explanations, expose hidden assumptions, or discover a discriminating observation. Repeatedly generating paraphrases under the same hidden assumptions offers much less information.

## Three large architectures worth serious consideration

### 1. A global explanation sampler with a symbolic channel core

This is the strongest primary candidate. The global state contains a hypothesis-family choice, a bounded writing program, key or lexicon variables, and distributions over local alignments. A neural proposal network reads evidence and suggests coordinated changes. A discrete diffusion or edit-flow component proposes plaintext and operation sequences conditioned on the global state. The channel core evaluates the evidence and rejects invalid transitions. A population of explanations survives rather than one winning translation.

The important novelty would be the integration and the scope, not inventing every component. Trainable proposals would need broad external language and known-script experience plus controlled encodings. The model would learn how to search a family of explanations; the manuscript would determine which explanations remain credible. An explicit nonsemantic branch prevents the procedure from requiring language even when the evidence does not.

This architecture could eventually produce a decipherment claim in a reviewable form: a frozen encoding program, shared parameter table, aligned readings, remaining ambiguity, accounted exceptions, and predictions on excluded material. It could also produce a valuable negative conclusion: multiple incompatible language/channel families remain observationally indistinguishable. Either outcome is more informative than a fluent generated paragraph.

The central risk is a proposal network that never reaches the correct mechanism because its training task distribution excludes it. Another is a verifier that accepts so much flexibility that it cannot distinguish hypotheses. The ambitious scientific challenge is to learn broad proposals while making final explanations compact and falsifiable.

### 2. A multimodal latent procedure model with text realization

This is a parallel, more speculative candidate. Construct a latent graph of entities, attributes, transformations, and spatial relations. Its observations include text, drawing components, layout, and repeated motifs; any semantic names initially remain uncertain. Diffusion proposes complete graph or trajectory configurations and associated linguistic realizations. A cross-view model predicts withheld observations rather than merely embedding all modalities into a visually pleasing cluster.

The reason to consider this is that ambiguity can be reduced by multiple views of the same latent object or operation. The reason for caution is that manuscript imagery may be schematic, copied from another source, or weakly related to local text. The graph must permit those explanations. Conditioning on a guessed plant identity and then celebrating a botanical translation would simply close a circle of assumptions.

A particularly ambitious extension is to learn translation between procedure structure and language across many independently understood historical texts, then transfer only the abstract relational constraints. This is a substantial corpus and philology project. Modern internet recipes and photorealistic plant images would be weak substitutes for the target genre and visual conventions.

### 3. An inverse writing simulator with interpretable latent operations

Instead of making plaintext the only latent object, model the writing process itself: select a lexical or formulaic unit, apply an abbreviation or coding operation, choose a surface form, position it on the page, and optionally copy or mutate prior material. Diffusion or edit-flow inference searches over these operation traces and shared rules. A learned world model predicts the consequences of choosing operations, while symbolic restrictions keep those consequences checkable.

This admits meaningful, nonsemantic, and mixed mechanisms within a common modeling framework. It can represent the possibility that the text has strong structure for reasons other than ordinary language. Mechanistic interpretation can ask whether the solver carries a belief about a shared key, a copy source, or a morphological boundary, and whether changing that belief selectively changes dependent predictions.

The limitation is historical nonuniqueness. Several operation programs can generate the same observed distribution. Recovering one compact simulator would identify a possible mechanism, not the actual medieval procedure. Independent manuscript evidence and constrained historical plausibility would still be needed before promoting an inferred operation to a historical claim.

## Direct decipherment precedents and what they do not establish

OBSD is a genuine diffusion-related ancient-script paper, published at ACL 2024. It translates oracle-bone glyph images toward modern Chinese character images and evaluates against known correspondences. Diff-Oracle uses controlled glyph generation to improve oracle-character recognition. These are relevant precedents for visual assistance, but their title-level use of “decipherment” should not be confused with discovering an unknown language and encoding system from an unaligned manuscript. [D24](https://aclanthology.org/2024.acl-long.831/), [D25](https://arxiv.org/abs/2312.13631)

A September 2026 preprint, FROD, explicitly frames assistance as cross-era image translation and reports a character-disjoint evaluation. This is a useful current lead, currently read at abstract depth in this review. Its claims do not establish sentence-level decipherment, and no reproduction was performed. [D26](https://arxiv.org/abs/2609.17227)

Bounded searches for combinations of Voynich, diffusion model, denoising diffusion, decipherment, cryptanalysis, and substitution cipher found no inspected primary paper demonstrating successful diffusion-based Voynich decoding. One public platform proposes training on cropped Voynich glyphs and morphing them toward known scripts. That is evidence that the idea has been proposed; the inspected page does not supply a validated decoding result. Similarity to a known alphabet would in any event require independent historical and linguistic support. Search absence is not proof that no relevant attempt exists. [D27](https://voynich-ai-production.up.railway.app/research)

## What would make the ambitious program intellectually successful

The decisive output is an explanation whose constraints travel: across passages, sections, transcriptions, and independently examined evidence. Diffusion sample quality, a compelling latent trajectory, and a strong language score are tools for finding explanations. They are not the criterion that establishes one.

For architecture selection, the large questions are whether global proposals improve exploration of genuinely different mechanisms, whether explicit structure prevents plausible but incompatible readings, and whether uncertainty remains honest when information is insufficient. This is not a proposed schedule of small experiments. It is the research specification for a serious system whose components may require extensive external training data, historical expertise, and substantial computation.

The most promising investment is a broad inference system with compact, inspectable final explanations. The least justified investment would be training a giant generative model on the manuscript alone and expecting scale to create missing anchors. The right ambition is to infer writing rules, semantic structure where identifiable, and the limits of identification together.
