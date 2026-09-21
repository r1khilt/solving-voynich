# Mechanism, identification, and synthesis sources

Accessed **2026-09-21**. These 16 records support [the mechanisms chapter](MECHANISMS_AND_IDENTIFIABILITY.md) and [the synthesis](SYNTHESIS.md). IDs are local to this review; earlier reviews also used `M` IDs. Use the containing directory when referring to them. All records are primary papers or author-hosted manuscripts. Reading means the explicitly listed portions, **not a complete paper/proof audit or independent replication**. No source files, models, or datasets were downloaded into Git. Web-rendered paper access is not a license to redistribute code, weights, or data.

## M01 — Sequence learning and emergent board state

- **Work:** Kenneth Li, Aspen K. Hopkins, David Bau, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg, *Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task*. ICLR 2023; arXiv first submitted 2022.
- **Consulted:** [arXiv 2210.13382v4](https://arxiv.org/html/2210.13382v4), §§2–4 including datasets, probe setup, intervention procedure and evaluation. Selected-section read.
- **Supports:** Board-state information and behavior-changing interventions in models trained on Othello move sequences; synthetic training scale of 20 million games.
- **Limits:** The true game state is available for supervised probes and evaluation. Not unsupervised discovery of arbitrary semantics; repeated layer edits and next-move evaluation are narrower than complete mechanism extraction.
- **Availability:** Paper links code and data; artifacts not downloaded or licensing-audited here.

## M02 — The reference frame changes interpretability

- **Work:** Neel Nanda, Andrew Lee, and Martin Wattenberg, *Emergent Linear Representations in World Models of Self-Supervised Sequence Models*. BlackboxNLP 2023, pp.16–30, December 7, 2023.
- **Consulted:** [ACL record](https://aclanthology.org/2023.blackboxnlp-1.2/) and [proceedings PDF](https://aclanthology.org/2023.blackboxnlp-1.2.pdf), introduction, §§2–4 and probing table. Selected-section read.
- **Supports:** Player-relative board encoding and linear steering in the studied Othello model.
- **Limits:** Reuses a known synthetic game model; does not prove universal linearity or resolve reference-frame search without ground truth.
- **Availability:** Author code link is in paper; runtime and license untested.

## M03 — Coherent state recovery needs more than local legality

- **Work:** Keyon Vafa, Justin Y. Chen, Ashesh Rambachan, Jon Kleinberg, and Sendhil Mullainathan, *Evaluating the World Model Implicit in a Generative Model*. NeurIPS 2024. Author ordering differs between first arXiv and proceedings records.
- **Consulted:** [arXiv 2406.03689v1, June 6, 2024](https://arxiv.org/html/2406.03689v1), §§1–3 and §4 discussion; [proceedings record](https://papers.nips.cc/paper_files/paper/2024/hash/2f6a6317bada76b26a4f61bb70a7db59-Abstract-Conference.html). Selected-section read; use v1 for claims here.
- **Supports:** Compression and distinction diagnostics grounded in deterministic automata; accurate local predictions can coexist with incoherent implicit structure.
- **Limits:** Metrics use a known automaton and acceptance thresholds. No direct ground-truth automaton for Voynich.
- **Availability:** [Author repository](https://github.com/keyonvafa/world-model-evaluation) inspected at README level; code/data/checkpoint links exist, not executed or license-audited.

## M04 — A conditional theory of latent recovery

- **Work:** Tianren Zhang, Guanyu Chen, and Feng Chen, *When Do Neural Networks Learn World Models?* ICML 2025, PMLR 267:74639–74670.
- **Consulted:** [PMLR record](https://proceedings.mlr.press/v267/zhang25j.html); [arXiv 2502.09297v2](https://arxiv.org/html/2502.09297v2), §§2–4, definitions and theorem statements. Selected-section read; proofs not fully audited.
- **Supports:** Identification under a multi-task Boolean formulation, invertible generation and particular low-degree biases.
- **Limits:** Requires specified task/support/architecture conditions; no general guarantee for natural language, noninvertible writing, or stochastic channels.
- **Availability:** Publication and [author project](https://trzhang0116.github.io/projects/world-models/index.html) accessible; implementation not inspected.

## M05 — Causal abstraction

- **Work:** Atticus Geiger, Duligur Ibeling, Amir Zur, Maheep Chaudhary, Sonakshi Chauhan, Jing Huang, Aryaman Arora, Zhengxuan Wu, Noah Goodman, Christopher Potts, and Thomas Icard, *Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability*. JMLR 26, 2025.
- **Consulted:** [JMLR record](https://www.jmlr.org/papers/v26/23-0058.html) and [64-page published PDF](https://www.jmlr.org/papers/volume26/23-0058/23-0058.pdf), introduction and definitions 44–48, distributed/recursive interchange interventions and accuracy, selected path-patching discussion. Selected-section read.
- **Supports:** A formal interface between low-level neural mechanisms and proposed high-level causal explanations.
- **Limits:** Faithfulness depends on chosen variables, alignment and intervention family; it does not certify the external historical process.
- **Availability:** Open paper. No software artifact audited.

## M06 — Causal representation learning

- **Work:** Bernhard Schölkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke, Nal Kalchbrenner, Anirudh Goyal, and Yoshua Bengio, *Towards Causal Representation Learning*. Proceedings of the IEEE, 2021.
- **Consulted:** [arXiv 2102.11107v1, February 22, 2021](https://arxiv.org/html/2102.11107v1), causal mechanisms and representation-learning discussion. Selected-section read of a research synthesis.
- **Supports:** Distinguishing causal variables/mechanisms from statistical correlations, and using invariance or distribution changes as additional structure.
- **Limits:** Research agenda and conditional arguments, not a plug-in identification algorithm for historical documents.
- **Availability:** Open manuscript; no task-specific code/data claimed.

## M07 — Robust policies and causal models

- **Work:** Jonathan Richens and Tom Everitt, *Robust Agents Learn Causal World Models*. ICLR 2024.
- **Consulted:** [arXiv 2402.10877v2, February 23, 2024](https://arxiv.org/html/2402.10877v2), §§2–4, including explicit limitations. Selected-section read; proofs not independently verified.
- **Supports:** Recoverability of causal information from sufficiently robust policy oracles under the paper's intervention and decision assumptions.
- **Limits:** Rich families of shifts, unmediated decision tasks, and finite-dimensional variables; passive manuscript prediction does not supply that oracle.
- **Availability:** Open manuscript; experimental code not inspected.

## M08 — Content versus style under augmentation

- **Work:** Julius von Kügelgen, Yash Sharma, Luigi Gresele, Wieland Brendel, Bernhard Schölkopf, Michel Besserve, and Francesco Locatello, *Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style*. NeurIPS 2021.
- **Consulted:** [published PDF](https://proceedings.neurips.cc/paper/2021/file/8929c70f8d710e412d38da624b21c3c8-Paper.pdf), introduction and formal setup; [author workshop presentation](https://icml21ssl.github.io/pages/files/ICML_2021__Self_supervised_learning_with_data_augmentations_provably_isolates_content_from_style.pdf), assumptions and theorem statement. Selected-section read.
- **Supports:** Content identification up to transformations under explicit augmentation and smoothness conditions.
- **Limits:** Knowing which transformations preserve content is substantive supervision. Real glyph edits may change the message.
- **Availability:** Paper describes Causal3DIdent; dataset/code/rights not inspected.

## M09 — Shared factors across heterogeneous modalities

- **Work:** Imant Daunhawer, Alice Bizeul, Emanuele Palumbo, Alexander Marx, and Julia E. Vogt, *Identifiability Results for Multimodal Contrastive Learning*. ICLR 2023.
- **Consulted:** [arXiv PDF](https://arxiv.org/pdf/2303.09166), inspected copy identifies **v1, March 16, 2023**; §§2–3, generative assumptions, abstract and theorem framing. Selected-section read. OpenReview PDF access encountered a browser challenge; no claim of reading through that route.
- **Supports:** Block identification of shared factors under specified multimodal generative assumptions.
- **Limits:** Smooth invertible mixing, content sharing and nuisance assumptions; no guarantee that a Voynich page's text and image describe identical content or identify named concepts.
- **Availability:** [Author repository](https://github.com/imantdaunhawer/multimodal-contrastive-learning) README inspected, code and separate data-generation repository linked. No license audit or execution.

## M10 — Mechanistic tools for diffusion language models

- **Work:** Xu Wang, Bingqing Jiang, Yu Wan, Baosong Yang, Lingpeng Kong, and Difan Zou, *DLM-Scope: Mechanistic Interpretability of Diffusion Language Models via Sparse Autoencoders*. 2026 preprint.
- **Consulted:** [arXiv 2602.05859v1, February 5, 2026](https://arxiv.org/html/2602.05859v1), §§2–3, training/steering definitions, §4 framing. Selected-section read.
- **Supports:** Masked/unmasked activation selection and repeated diffusion-time interventions; studied backbones Dream-7B and LLaDA-8B.
- **Limits:** Modern-language feature steering is not unknown-language mechanism recovery; automated concept labels and intervention success need independent validation.
- **Availability:** Paper links [DLM-Scope artifacts](https://huggingface.co/DLM-Scope); weights and licenses not audited or downloaded.

## M11 — Interpretable concepts across image-diffusion time

- **Work:** Berk Tinaz, Zalan Fabian, and Mahdi Soltanolkotabi, *Emergence and Evolution of Interpretable Concepts in Diffusion Models*. 2025 preprint.
- **Consulted:** [arXiv 2504.15473v1, April 21, 2025](https://arxiv.org/html/2504.15473v1), intervention equations and §§4.1–4.3. Selected-section read.
- **Supports:** Time-sensitive concept analysis and image layout/style interventions in SD v1.4.
- **Limits:** Concept labels involve external vision models; timestep-specific dictionaries are not automatically aligned. Image generation is not causal historical reconstruction.
- **Availability:** Open paper; code/checkpoint rights not checked.

## M12 — A reusable program library and neural search

- **Work:** Kevin Ellis, Catherine Wong, Maxwell Nye, Mathias Sablé-Meyer, Lucas Morales, Luke Hewitt, Luc Cary, Armando Solar-Lezama, and Joshua B. Tenenbaum, *DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning*. PLDI 2021, DOI 10.1145/3453483.3454080. The arXiv title differs.
- **Consulted:** [author-hosted published paper](https://www.neurosymbolic.org/papers/EllisWNSMHCST21.pdf), introduction, §2 and wake/sleep/library figures. Selected-section read.
- **Supports:** Combining supplied primitives, specified tasks, library induction and learned search.
- **Limits:** Training problems have evaluable specifications; Voynich semantics do not. A learned library's compression does not establish its historical use.
- **Availability:** Open paper; no software installed.

## M13 — Sampling diverse programs or graphs

- **Work:** Yoshua Bengio, Salem Lahlou, Tristan Deleu, Edward J. Hu, Mo Tiwari, and Emmanuel Bengio, *GFlowNet Foundations*. JMLR 24, 2023; published June 2023.
- **Consulted:** [published PDF](https://jmlr.org/papers/volume24/22-0364/22-0364.pdf), introduction, reward/flow framing, §4.6–4.7 discussion. Selected-section read; convergence proofs not audited.
- **Supports:** Amortized sampling of composite objects relative to a defined nonnegative reward; relevance to retaining multiple hypotheses.
- **Limits:** Learned flow matching and chosen target can be wrong; diversity is not posterior calibration or historical evidence.
- **Availability:** Open paper; no implementation or model availability claim.

## M14 — Inverting a model of intentional action

- **Work:** Chris L. Baker, Rebecca Saxe, and Joshua B. Tenenbaum, *Action Understanding as Inverse Planning*. Cognition 113(3):329–349, 2009; DOI 10.1016/j.cognition.2009.07.005.
- **Consulted:** [MIT author manuscript](https://web.mit.edu/9.s915/www/classes/cognition2009.pdf), abstract, introduction and belief/goal/action formulation; [MIT bibliographic record](https://dspace.mit.edu/entities/publication/2e23673d-f720-4720-98bc-6db894ba27e1). Selected-section read.
- **Supports:** Bayesian inference over goals and beliefs using a specified model of approximately rational planning.
- **Limits:** Goals, environment and costs constrain the inference. No evidence that the Voynich writer fits a chosen rational-reader utility.
- **Availability:** Author manuscript accessible; no historical manuscript dataset.

## M15 — Simulation-based inference

- **Work:** Kyle Cranmer, Johann Brehmer, and Gilles Louppe, *The Frontier of Simulation-Based Inference*. PNAS 2020; arXiv initial submission 2019.
- **Consulted:** [arXiv 1911.01429v3](https://arxiv.org/html/1911.01429v3), introduction and likelihood/posterior/active-inference discussion. Selected-section read of a methods review.
- **Supports:** Learning inference from simulator-generated observations and exposing latent structure to inference procedures.
- **Limits:** The simulator defines the statistical model; inference can be precise under a misspecified world.
- **Availability:** Open paper; no Voynich simulator supplied.

## M16 — Calibration within a generative model

- **Work:** Sean Talts, Michael Betancourt, Daniel Simpson, Aki Vehtari, and Andrew Gelman, *Validating Bayesian Inference Algorithms with Simulation-Based Calibration*. arXiv initial submission 2018.
- **Consulted:** [arXiv 1804.06788v2](https://arxiv.org/html/1804.06788v2), §§2–4 on joint-distribution consistency and rank diagnostics. Selected-section read.
- **Supports:** Repeated prior-predictive simulations can reveal faulty approximate posterior computation or model implementation.
- **Limits:** Passing checks does not show that a simulator describes real manuscript evidence. Rank checks alone also do not establish useful posterior concentration.
- **Availability:** Open manuscript; no experiments rerun here.

## Audit notes

The 2025 and 2026 preprints above were inspected at explicit versions, rather than treated as settled consensus. Source M03 deliberately uses the first preprint for its inspected technical content and the proceedings for bibliographic identity. Some HTML requests for old papers failed; working PDFs and official proceedings were used where recorded. No result in this ledger is a new result from this repository.

Recurring works shared with the earlier `deep-review-2026-09-21` collection are rechecked or extended here, not counted as novel discoveries. Records in this ledger describe the current reading depth; they do not silently upgrade the reading depth of earlier records.
