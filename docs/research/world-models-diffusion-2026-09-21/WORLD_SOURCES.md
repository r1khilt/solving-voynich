# World-model and action-model source ledger

All records accessed **2026-09-21**. Stable IDs W01–W23 are local to this review. This is a curated primary-source reading ledger, not an exhaustive bibliography. `selected_sections` means the listed material was inspected; it never means all proofs, experiments, or appendices were read. `abstract_only` supports high-level descriptions only. Corporate announcements are primary statements by their authors, with weaker methodological detail than a paper. No source result was replicated.

Authors are recorded in full for short lists and explicitly abbreviated for large collaborations; the linked primary record supplies the complete list. ArXiv versions refer to the **version actually used**, not an implicit claim to have reviewed every later revision. Dates below are the dates displayed by the source. Where a metadata oddity was encountered, it is retained rather than silently corrected. Code/data/license statements describe what was checked; a public paper is not a license to redistribute its training data or model.

## W01 — Predictive representations of state

- **Authors:** Michael L. Littman, Richard S. Sutton, Satinder Singh. The PDF names all three; the proceedings landing-page metadata omits Singh.
- **Title/date/version:** *Predictive Representations of State*. NIPS 2001, proceedings volume 14; proceedings PDF, no arXiv version.
- **Primary URL:** [paper](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf).
- **Depth:** `selected_sections` — abstract and introduction. Not a complete proof audit.
- **Load-bearing claim:** State can be represented through multi-step action-conditional predictions; representation existence does not solve learning or historical interpretation.
- **Limitations:** Controlled-system assumptions and finite representational results must not be re-described as ciphertext-only identifiability.
- **Availability:** Open proceedings PDF verified; code, datasets, and redistribution license not checked.

## W02 — World Models

- **Authors:** David Ha, Jürgen Schmidhuber.
- **Title/date/version:** *World Models*. First submitted 2018-03-27; v4 2018-05-09.
- **Primary URLs:** [arXiv v4](https://arxiv.org/abs/1803.10122v4), [author interactive paper](https://worldmodels.github.io/).
- **Depth:** `selected_sections` — author interactive paper: Agent Model (VAE, MDN-RNN, Controller), Car Racing data collection and training procedure; arXiv metadata checked separately.
- **Load-bearing claim:** Compressed learned spatial-temporal models can support compact controllers and imagined-environment policy training in the demonstrated RL environments.
- **Limitations:** Interactive evaluation and environment trajectories do not establish unknown-script recovery or semantic uniqueness.
- **Availability:** Author interactive paper inspected; implementation code and license not audited.

## W03 — PlaNet

- **Authors:** Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson.
- **Title/date/version:** *Learning Latent Dynamics for Planning from Pixels*. First submitted 2018-11-12; v5 2019-06-04.
- **Primary URL:** [arXiv v5](https://arxiv.org/abs/1811.04551v5).
- **Depth:** `selected_sections` — [full HTML](https://arxiv.org/html/1811.04551v5) §3 recurrent state-space model, latent dynamics equations, deterministic/stochastic comparison; abstract and metadata.
- **Load-bearing claim:** Stochastic/deterministic latent dynamics and latent overshooting support online planning for image-based control.
- **Limitations:** Observing pixels is not the same as lacking actions and reward; no manuscript transfer claim.
- **Availability:** Open paper verified; official code/data/license not audited.

## W04 — DreamerV3

- **Authors:** Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap.
- **Title/date/version:** *Mastering Diverse Domains through World Models*. First submitted 2023-01-10; v2 2024-04-17. This record concerns that arXiv revision.
- **Primary URL:** [arXiv v2](https://arxiv.org/abs/2301.04104v2).
- **Depth:** `selected_sections` — [full HTML](https://arxiv.org/html/2301.04104v2) world-model learning, actor/critic learning, input/action/reward supervision, robustness and scaling discussion excerpts.
- **Load-bearing claim:** A common configuration supports model-based RL across a broad interactive task suite; imagined rollouts improve behavior.
- **Limitations:** Reported task breadth does not demonstrate novel-language grounding or decipherment. Later publication/revisions are not silently substituted.
- **Availability:** Author project page linked by paper; code/data/license not audited.

## W05 — MuZero

- **Authors:** Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, Timothy Lillicrap, David Silver.
- **Title/date/version:** *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model*. First submitted 2019-11-19; v2 2020-02-21; related Nature DOI 10.1038/s41586-020-03051-4.
- **Primary URL:** [arXiv v2](https://arxiv.org/abs/1911.08265v2).
- **Depth:** `selected_sections` — [full HTML](https://arxiv.org/html/1911.08265v2) Figure 1 and model description (representation, dynamics, prediction), hidden-state semantics, selected input-encoding appendix excerpts.
- **Load-bearing claim:** A learned model can support search through reward, policy, and value predictions without reconstructing all observations.
- **Limitations:** Task sufficiency does not guarantee a physically or semantically complete internal model.
- **Availability:** ArXiv lists ancillary evaluation JSON and pseudocode; contents and license not audited.

## W06 — Slot Attention

- **Authors:** Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, Thomas Kipf.
- **Title/date/version:** *Object-Centric Learning with Slot Attention*. First submitted 2020-06-26; v2 2020-10-14.
- **Primary URL:** [arXiv v2](https://arxiv.org/abs/2006.15055v2).
- **Depth:** `abstract_only`.
- **Load-bearing claim:** Exchangeable slots are an architectural route toward object-centered representations.
- **Limitations:** Linguistic roles, manuscript parts, and stable cross-page correspondence are proposed extensions, not demonstrated results here.
- **Availability:** Paper verified; code/data/license not audited.

## W07 — SlotFormer

- **Authors:** Ziyi Wu, Nikita Dvornik, Klaus Greff, Thomas Kipf, Animesh Garg.
- **Title/date/version:** *SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models*. First submitted 2022-10-12; v2 2023-01-21.
- **Primary URL:** [arXiv v2](https://arxiv.org/abs/2210.05861v2).
- **Depth:** `abstract_only`.
- **Load-bearing claim:** Object-centered representations can support learned visual dynamics and downstream reasoning.
- **Limitations:** Visual entity decomposition is not evidence that anonymous linguistic or historical entities have been recovered.
- **Availability:** Paper verified; code/data/license not audited.

## W08 — I-JEPA

- **Authors:** Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas.
- **Title/date/version:** *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. First submitted 2023-01-19; v3 2023-04-13.
- **Primary URL:** [arXiv v3](https://arxiv.org/abs/2301.08243v3).
- **Depth:** `abstract_only`.
- **Load-bearing claim:** Masked-region feature prediction supplies a self-supervised alternative to pixel reconstruction.
- **Limitations:** Learned invariance can discard distinctions needed by a writing system; feature distance is not automatically evidence likelihood.
- **Availability:** Paper verified; code/data/license not audited.

## W09 — V-JEPA

- **Authors:** Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, Nicolas Ballas.
- **Title/date/version:** *Revisiting Feature Prediction for Learning Visual Representations from Video*. Primary metadata displays submitted 2024-02-15, v1; identifier 2404.08471. The unusual date/identifier pairing is recorded as displayed, not resolved here.
- **Primary URL:** [arXiv v1](https://arxiv.org/abs/2404.08471v1).
- **Depth:** `abstract_only`.
- **Load-bearing claim:** Video feature prediction without text or pixel reconstruction learns useful visual representations in the reported evaluation.
- **Limitations:** The evaluated representations do not establish a uniquely identified generative world or decipherment capability.
- **Availability:** Paper verified; code/data/license not audited.

## W10 — V-JEPA 2 and V-JEPA 2-AC

- **Authors:** Mahmoud Assran (arXiv abstract metadata also uses Mido Assran), Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, Nicolas Ballas.
- **Title/date/version:** *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*. arXiv v1 banner 2025-06-11; retrieved HTML additionally displays a 2026-08-24 document date. This inconsistency is retained; the inspected content is at the linked versioned URL.
- **Primary URLs:** [full HTML v1](https://arxiv.org/html/2506.09985v1), [official code link](https://github.com/facebookresearch/vjepa2).
- **Depth:** `selected_sections` — §§1, 2.1, 2.3–2.4, 3.1, 4.2–4.3; official repository checkpoint tables and license section. Not all appendices.
- **Load-bearing claim:** Passive pretraining and action-conditioned robot adaptation are distinct stages. The latter uses measured end-effector states; “unlabeled” excludes task/reward/success metadata, not all action information.
- **Limitations:** Supplied subgoals, camera sensitivity, finite-horizon robot tasks; no direct static-manuscript transfer.
- **Availability:** Official repository lists code, pretrained encoders, action-conditioned weights and configs. README states majority MIT with specified Apache-2.0 portions; individual checkpoint/data rights not independently audited.

## W11 — Genie

- **Authors:** Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, Yusuf Aytar, Sarah Bechtle, Feryal Behbahani, Stephanie Chan, Nicolas Heess, Lucy Gonzalez, Simon Osindero, Sherjil Ozair, Scott Reed, Jingwei Zhang, Konrad Zolna, Jeff Clune, Nando de Freitas, Satinder Singh, Tim Rocktäschel.
- **Title/date/version:** *Genie: Generative Interactive Environments*. arXiv v1 2024-02-23; later ICML 2024 proceedings exists, but v1 HTML is the method text inspected.
- **Primary URL:** [full HTML v1](https://arxiv.org/html/2402.15391v1).
- **Depth:** `selected_sections` — §2.1 latent-action model, §2.2 inference, §3.4 ablations, §4 related work, selected implementation appendix excerpt.
- **Load-bearing claim:** A constrained discrete bottleneck learns latent changes from past/future frames; user-selected codes control generated frames at inference.
- **Limitations:** Reconstruction and apparent controllability do not prove uniquely recovered causal actions. Future observations enter the training encoder.
- **Availability:** Paper gives a reproducible case study; no official main-model code/weight release was verified.

## W12 — Genie 2 official announcement

- **Authors:** Jack Parker-Holder et al., Google DeepMind; collaborative byline inspected.
- **Title/date/version:** *Genie 2: A large-scale foundation world model*. Official announcement, 2024-12-04; mutable webpage.
- **Primary URL:** [announcement](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/).
- **Depth:** `official_announcement` — introduction, action interface, consistency duration, and technical paragraph specifying autoregressive latent diffusion and action conditioning.
- **Load-bearing claim:** Developer describes action-conditioned interactive world generation with a diffusion model.
- **Limitations:** Announcement-level evidence; no full training/data/evaluation reproduction in this review.
- **Availability:** Demos public; code/weights/data/license not established.

## W13 — Genie 3 official announcement

- **Authors:** Jack Parker-Holder and Shlomi Fruchter, Google DeepMind.
- **Title/date/version:** *Genie 3: A new frontier for world models*. Official announcement, 2025-08-05; mutable webpage.
- **Primary URL:** [announcement](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/).
- **Depth:** `official_announcement` — byline/date, opening capability description and world-simulation comparison. No undisclosed technical architecture inferred.
- **Load-bearing claim:** Developer reports real-time interactive generation with greater consistency/persistence than earlier systems.
- **Limitations:** No architecture inferred from the announcement; demonstrations are not causal-ground-truth validation.
- **Availability:** Access/release status can change; no weights, training data, or license established here.

## W14 — LAPA

- **Authors:** Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, Minjoon Seo.
- **Title/date/version:** *Latent Action Pretraining from Videos*. v1 2024-10-15 inspected; metadata also verified v2 2025-05-15, which was not the full text used.
- **Primary URLs:** [full HTML v1](https://arxiv.org/html/2410.11758v1), [official repository](https://github.com/LatentActionPretraining/LAPA).
- **Depth:** `selected_sections` — §§3.1–3.3, preceding supervision/related-work discussion; official repository latent inference, fine-tuning, pretraining and license sections.
- **Load-bearing claim:** Video-derived latent actions support pretraining, followed by actual action-labeled robot adaptation.
- **Limitations:** Action-free pretraining is not an entirely ungrounded pipeline; task descriptions condition latent policy learning.
- **Availability:** Official repository supplies training/inference code and checkpoint links and states code/model weights are MIT licensed. Data and inherited dependency rights not independently audited; nothing downloaded or executed.

## W15 — RT-2

- **Authors:** Anthony Brohan et al.; 54-author primary list inspected.
- **Title/date/version:** *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*. v1 2023-07-28.
- **Primary URL:** [arXiv v1](https://arxiv.org/abs/2307.15818v1).
- **Depth:** `abstract_only`.
- **Load-bearing claim:** A shared token interface transfers visual-language knowledge to robot control when combined with robot supervision.
- **Limitations:** Robot behavior is grounded in aligned experience; unknown symbols and missing historical meanings create a different problem.
- **Availability:** Paper verified; code/weights/data/license not audited.

## W16 — OpenVLA

- **Authors:** Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, Chelsea Finn.
- **Title/date/version:** *OpenVLA: An Open-Source Vision-Language-Action Model*. First submitted 2024-06-13; v3 2024-09-05.
- **Primary URLs:** [arXiv v3](https://arxiv.org/abs/2406.09246v3), [author project page](https://openvla.github.io/).
- **Depth:** `abstract_only` for paper; author project adaptation/semantic discussion and official repository checkpoint, dataset and license sections also inspected.
- **Load-bearing claim:** An open 7B VLA is trained on approximately 970,000 robot demonstrations; the scale of aligned grounding matters to transfer comparisons.
- **Limitations:** A reusable robotic checkpoint is not a decipherment model; semantic and physical priors may not match the manuscript.
- **Availability:** [Official repository](https://github.com/openvla/openvla) provides code/checkpoints/configs and dataset-mixture instructions. Code is MIT; README says pretrained weights inherit Llama-2 Community License restrictions. Dataset-specific rights not audited.

## W17 — π₀

- **Authors:** Kevin Black et al. (Physical Intelligence); 24-author list verified in arXiv metadata.
- **Title/date/version:** *π₀: A Vision-Language-Action Flow Model for General Robot Control*. v1 2024-10-31 method text inspected; metadata verifies later v4 2026-01-08, not fully inspected.
- **Primary URL:** [full HTML v1](https://arxiv.org/html/2410.24164v1).
- **Depth:** `selected_sections` — §§III–IV, dataset overview, Appendix A inference/action-expert excerpts.
- **Load-bearing claim:** A VLM plus flow-matching action expert models coordinated continuous action chunks using substantial robot demonstration data.
- **Limitations:** Flow over motor trajectories does not directly define valid distributions over discrete channel programs or grammar structures.
- **Availability:** [Official openpi repository](https://github.com/Physical-Intelligence/openpi) lists base/expert weights and training/inference code. Repository displays Apache-2.0 and a separate Gemma license file; checkpoint/data terms not individually audited.

## W18 — π₀.₅

- **Authors:** Physical Intelligence; Kevin Black et al. Full collaborative author list provided on primary page, not reproduced here.
- **Title/date/version:** *π₀.₅: a Vision-Language-Action Model with Open-World Generalization*. v1 2025-04-22.
- **Primary URL:** [full HTML v1](https://arxiv.org/html/2504.16054v1).
- **Depth:** `selected_sections` — §IV architecture and training recipe, §V-C co-training discussion, §VI limitations/discussion excerpts.
- **Load-bearing claim:** Heterogeneous co-training and high-level-subtask/low-level-action factorization support broader robot generalization.
- **Limitations:** Useful hierarchy is not proof that unsupervised linguistic or historical semantics can be discovered.
- **Availability:** [Official openpi repository](https://github.com/Physical-Intelligence/openpi) lists π₀.₅ base and adapted checkpoints. Repository license labels inspected as in W17; checkpoint and dataset-specific rights not individually audited.

## W19 — LaWAM

- **Authors:** Jialei Chen, Kai Wang, Kang Chen, Shuaihang Chen, Feng Gao, Wenhao Tang, Zhiyuan Li, Weilin Liu, Zhuyu Yao, Boxun Li, Yuanbo Xu, Chao Yu.
- **Title/date/version:** *LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies*. v1 2026-06-14.
- **Primary URL:** [arXiv v1](https://arxiv.org/abs/2606.15768v1).
- **Depth:** `abstract_only`.
- **Load-bearing claim:** Authors propose exposing predicted latent visual subgoals to robot policies instead of rendering full future video.
- **Limitations:** Fresh preprint; evaluation details, baseline comparability, latency conditions, and generalization claims not audited. No reported score is adopted as an established ranking.
- **Availability:** Paper verified; release contents and licenses not checked.

## W20 — MoWAM

- **Authors:** Jiayu Wang, Bin Zhu, Yue Yu, Jingjing Chen.
- **Title/date/version:** *MoWAM: Explicit Future Motion Prediction for Efficient World Action Models*. v1 2026-09-17.
- **Primary URL:** [arXiv v1](https://arxiv.org/abs/2609.20709v1).
- **Depth:** `abstract_only`.
- **Load-bearing claim:** Authors propose explicit compact future-motion prediction and candidate selection as an alternative to inference-time video generation.
- **Limitations:** Four-day-old preprint at access; no independent replication, detailed experiment audit, or manuscript relevance demonstrated.
- **Availability:** Paper verified; code/data/license not checked.

## W21 — ZimaBlue

- **Authors:** Xionghao Wu, Yijun Yang, Shiyang Zhou, Haoze Sun, Jianhui Liu, Songsong Yu, Jiyao Zhang, Wenbo Li, Bo Wang, Guoqing Ma, Lin Song, Renjie Liao, Shenghe Zheng, Wei Tang, Xiaojuan Qi, Yanwei Li, Yuan Zhang, Zhuotao Tian, Haoyang Huang, Nan Duan.
- **Title/date/version:** *ZimaBlue: Evolving Generalizable World Action Models through Scalable Video Pre-training*. v1, displayed submitted 2026-08-31, identifier 2609.00188. Date retained as displayed.
- **Primary URL:** [arXiv v1](https://arxiv.org/abs/2609.00188v1).
- **Depth:** `abstract_only`; primary arXiv abstract confirms the three-stage training curriculum and asynchronous slow/fast architecture. Publication metadata verified.
- **Load-bearing claim:** Current work investigates scaling video priors into executable robot policies.
- **Limitations:** Fresh preprint; training-scale and real-time performance claims are not audited or used for our cost estimate.
- **Availability:** Paper verified; code/data/license not checked.

## W22 — LAWM (distinct from LaWAM)

- **Authors:** Bahey Tharwat, Yara Nasser, Ali Abouzeid, Ian Reid.
- **Title/date/version:** *Latent Action Pretraining Through World Modeling*. v1 2025-09-22.
- **Primary URL:** [arXiv v1](https://arxiv.org/abs/2509.18428v1).
- **Depth:** `abstract_only`.
- **Load-bearing claim:** A model-agnostic latent-action pretraining framework learns from video through world modeling.
- **Limitations:** No performance number or broad superiority claim is adopted; title similarity with W19 must not collapse separate works.
- **Availability:** Paper verified; code/data/license not checked.

## W23 — Direct Voynich proposal precedent

- **Author:** Voynich AI platform; individual authorship not established from the inspected page.
- **Title/date/version:** *Voynich AI — Manuscript Analysis Platform*, Research and Changelog pages; mutable webpages. Changelog entries displayed 2026-03-07.
- **Primary URLs:** [research](https://voynich-ai-production.up.railway.app/research), [changelog](https://voynich-ai-production.up.railway.app/changelog).
- **Depth:** `selected_page_sections` — proposed diffusion glyph analysis, CLIP alignment, synthetic pretraining, and implementation-specification changelog entries.
- **Load-bearing claim:** Similar diffusion/multimodal ideas have been publicly proposed for Voynich, so universal novelty claims are unjustified.
- **Limitations:** This is not verified decipherment or an audited scientific result. Feasibility/novelty claims, counts, scribe claims, and historical statements were not validated.
- **Availability:** Public webpages verified; ownership, underlying datasets, code, and licenses not audited.

## Reading-depth totals

- **23 source records:** 20 papers, two official announcements, one direct-Voynich proposal website record (two related pages).
- **Paper depth:** 10 selected-section readings and 10 abstract-only readings. W16 additionally includes official project/repository documentation; this does not upgrade its paper reading to selected sections.
- **Release checks:** selected official repository documentation checked for V-JEPA 2, LAPA, OpenVLA and openpi. No model, dataset, or implementation was run. These are availability observations, not complete licensing or reproducibility audits.
- **Coverage boundary:** repository discovery also surfaced V-JEPA 2.1; it was not added to the reviewed paper set. This chapter does not claim an exhaustive survey of every model available on the access date.

## Search and exclusion record

- **Bounded direct-application searches:** `"Voynich" "world model"`, `"Voynich" "VLA"`, `"Voynich" "JEPA"`, `"Voynich" "diffusion" model decipherment`.
- **Observed outcome:** Relevant public proposals found (W23); no substantive evaluated learned-action-world-model decipherment located in this bounded pass. Search absence is not proof of nonexistence.
- **Discovery-only material:** Search results for 2026 WAM surveys/tutorials and third-party paper summaries were used to locate primary works. Their claims are not adopted as technical evidence. Forum speculation, generated plant images, and metaphorical uses of “world model” were not counted as decipherment results.
- **Shared review boundary:** Mechanistic world-model interpretation and causal-identifiability sources are covered by the parent review; this chapter uses its own deductions for the static-data/action distinction and avoids pretending that a robotics benchmark resolves those questions.
