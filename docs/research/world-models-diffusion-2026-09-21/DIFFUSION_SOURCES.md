# Diffusion source ledger

Accessed **2026-09-21**. Stable IDs D01–D27. This is a targeted review: 26 papers and one direct-proposal website, not 27 full-paper reads. `Selected sections` means the named material was inspected; `abstract + metadata` means deeper claims are deliberately not relied upon. Publication dates and versions below distinguish initial submissions from inspected revisions. “Availability not audited” means no claim about usable code, checkpoints, dataset rights, or reproducibility. No listed system was installed or run.

Source-derived summaries are kept short; proposed manuscript applications in the companion chapter are our synthesis. arXiv preprints are not treated as independent verification of their own empirical claims. Paper-version links identify what was inspected, not necessarily the newest possible version of every source.

## D01 — Original diffusion construction

- **Authors/title:** Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, Surya Ganguli. *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*.
- **Date/version:** first 2015-03-12; inspected metadata for v8, 2015-11-18.
- **Primary URL:** https://arxiv.org/abs/1503.03585v8
- **Reading depth:** abstract + metadata.
- **Supports:** learned reversal of a deliberately destructive forward process as generative modeling.
- **Limit:** historical origin only; no manuscript claim or new derivation taken from this abstract.
- **Availability:** not audited.

## D02 — DDPM

- **Authors/title:** Jonathan Ho, Ajay Jain, Pieter Abbeel. *Denoising Diffusion Probabilistic Models*.
- **Date/version:** first 2020-06-19; v2, 2020-12-16.
- **Primary URL:** https://arxiv.org/html/2006.11239v2
- **Reading depth:** selected sections 2, 3.1–3.4; training/sampling algorithms.
- **Supports:** artificial Gaussian corruption, learned reverse transitions, variational and simplified noise-prediction objectives.
- **Limit:** image-generation evidence; computational diffusion time is not historical time.
- **Availability:** paper links an implementation; repository and license not audited.

## D03 — Score SDE and inverse problems

- **Authors/title:** Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole. *Score-Based Generative Modeling through Stochastic Differential Equations*.
- **Date/version:** first 2020-11-26; v2, 2021-02-10; ICLR 2021.
- **Primary URL:** https://arxiv.org/html/2011.13456v2
- **Reading depth:** abstract and selected Appendix I.2–I.4, especially general inverse-problem assumptions.
- **Supports:** conditional score formulation; the inspected inverse-problem construction uses tractability and approximation assumptions.
- **Limit:** cannot transfer exactness to an unknown discrete historical channel.
- **Availability:** not audited.

## D04 — D3PM

- **Authors/title:** Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, Rianne van den Berg. *Structured Denoising Diffusion Models in Discrete State-Spaces*.
- **Date/version:** first 2021-07-07; v3, 2023-02-22.
- **Primary URL:** https://arxiv.org/html/2107.03006v3
- **Reading depth:** selected sections 2–3 and 5.1–5.2; absorbing-state formulas in Appendix A.
- **Supports:** categorical transition matrices, tractable corruption posteriors, and the empirical importance of corruption design. Embedding-neighbor transitions did not consistently help text.
- **Limit:** fixed-position token corruption does not discover writing-system alignment.
- **Availability:** not audited.

## D05 — Continuous text diffusion

- **Authors/title:** Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, Tatsunori B. Hashimoto. *Diffusion-LM Improves Controllable Text Generation*.
- **Date/version:** v1, 2022-05-27.
- **Primary URL:** https://arxiv.org/html/2205.14217v1
- **Reading depth:** selected sections 3–4.2, embedding/rounding formulation and objective qualifications.
- **Supports:** learned continuous embeddings and explicit conversion back to discrete text; naive rounding can fail to commit to words.
- **Limit:** fluent controlled text is not cryptanalytic correspondence.
- **Availability:** not audited.

## D06 — SEDD

- **Authors/title:** Aaron Lou, Chenlin Meng, Stefano Ermon. *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution*.
- **Date/version:** first 2023-10-25; v3, 2024-06-06; ICML 2024.
- **Primary URL:** https://arxiv.org/html/2310.16834v3
- **Reading depth:** selected sections 3.1–3.3, score-entropy consistency and likelihood bound.
- **Supports:** ratio-based discrete score estimation and a likelihood-bound training/evaluation formulation.
- **Limit:** asymptotic consistency assumes support, capacity, and data conditions; no scarce-manuscript guarantee.
- **Availability:** paper links code; repository and license not audited.

## D07 — MDLM

- **Authors/title:** Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T. Chiu, Alexander Rush, Volodymyr Kuleshov. *Simple and Effective Masked Diffusion Language Models*.
- **Date/version:** first 2024-06-11; v2, 2024-11-10; NeurIPS 2024.
- **Primary URL:** https://arxiv.org/html/2406.07524v2
- **Reading depth:** selected introduction and background/objective discussion; not all appendices.
- **Supports:** practical masked-diffusion training; weighted masked-language losses and a variational objective.
- **Limit:** objective bounds, sampler quality, and exact generative likelihood remain distinct.
- **Availability:** paper links code; repository and license not audited.

## D08 — LLaDA

- **Authors/title:** Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, Chongxuan Li. *Large Language Diffusion Models*.
- **Date/version:** first 2025-02-14; v3, 2025-10-18.
- **Primary URL:** https://arxiv.org/html/2502.09992v3
- **Reading depth:** selected introduction and sections 2.1–2.2; resource statement and loss equation.
- **Supports:** large-scale masked-diffusion language modeling; report states 8B parameters, 2.3T tokens and 0.13M H800 GPU hours.
- **Limit:** this is the original LLaDA paper, not a claim about the latest LLaDA-branded release; no low-resource transfer demonstrated here.
- **Availability:** official repository found at https://github.com/lrbob/LLaDA ; no checkpoint downloaded or license audit performed.

## D09 — Dream 7B

- **Authors/title:** Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, Lingpeng Kong. *Dream 7B: Diffusion Large Language Models*.
- **Date/version:** v1, 2025-08-21.
- **Primary URL:** https://arxiv.org/html/2508.15487v1
- **Reading depth:** selected sections 4.1–4.3 and benchmark setup.
- **Supports:** autoregressive initialization, preserved prediction shift, context-adaptive weighting, 580B-token additional corpus.
- **Limit:** additional training tokens exclude the inherited pretrained model's full training expenditure; paper benchmarks do not establish decipherment.
- **Availability:** paper announces base/instruct releases; files and licenses not audited.

## D10 — ReMDM

- **Authors/title:** Guanghan Wang, Yair Schiff, Subham Sekhar Sahoo, Volodymyr Kuleshov. *Remasking Discrete Diffusion Models with Inference-Time Scaling*.
- **Date/version:** first 2025-03-01; v4, 2026-02-07; current arXiv metadata says NeurIPS 2025.
- **Primary URL:** https://arxiv.org/html/2503.00307v4
- **Reading depth:** selected section 3.1, marginal-preservation theorem and remasking bounds; sampler overview.
- **Supports:** revisiting committed tokens needs an explicit mechanism; remasking construction can reuse pretrained masked models.
- **Limit:** extra inference is not proof of posterior calibration or monotonic historical accuracy.
- **Availability:** project links present; implementation not audited. An older indexed attachment's venue label conflicted with current metadata and was not used as authoritative.

## D11 — Insertion/deletion diffusion

- **Authors/title:** Daniel D. Johnson, Jacob Austin, Rianne van den Berg, Daniel Tarlow. *Beyond In-Place Corruption: Insertion and Deletion In Denoising Probabilistic Models*.
- **Date/version:** first 2021-07-16; v1 referenced; ICML 2021 workshop publication.
- **Primary URLs:** https://arxiv.org/abs/2107.07675v1 ; https://research.google/pubs/beyond-in-place-corruption-insertion-and-deletion-in-denoising-probabilistic-models/
- **Reading depth:** abstract and workshop introduction/background excerpt.
- **Supports:** diffusion over changing sequence lengths; arithmetic-sequence and text8 spelling-correction demonstrations.
- **Limit:** error correction under known-language training differs from discovering an unknown historical channel.
- **Availability:** not audited.

## D12 — Edit Flows

- **Authors/title:** Marton Havasi, Brian Karrer, Itai Gat, Ricky T. Q. Chen. *Edit Flows: Flow Matching with Edit Operations*.
- **Date/version:** first 2025-06-10; v3, 2025-11-12. Some conference listings use the expanded subtitle “Variable Length Discrete Flow Matching with Edit Operations.”
- **Primary URL:** https://arxiv.org/html/2506.09018v3
- **Reading depth:** selected sections 2.2, 3.1–3.3; auxiliary alignment construction and theorem statement.
- **Supports:** insertion/deletion/substitution CTMC over sequence space; rates and token values parameterized separately; auxiliary alignments enable training.
- **Limit:** generative flexibility does not identify the true alignment or a historical edit process; cap and resource assumptions still matter.
- **Availability:** not audited.

## D13 — Continuous flow matching

- **Authors/title:** Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le. *Flow Matching for Generative Modeling*.
- **Date/version:** first 2022-10-06; v2, 2023-02-08.
- **Primary URL:** https://arxiv.org/abs/2210.02747v2
- **Reading depth:** abstract + metadata.
- **Supports:** vector-field regression along prescribed probability paths, including paths beyond diffusion.
- **Limit:** no claim here that a transport path is a true writing process or a calibrated inverse solution.
- **Availability:** not audited.

## D14 — Discrete flow matching

- **Authors/title:** Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky T. Q. Chen, Gabriel Synnaeve, Yossi Adi, Yaron Lipman. *Discrete Flow Matching*.
- **Date/version:** first 2024-07-22; v2, 2024-11-05.
- **Primary URL:** https://arxiv.org/abs/2407.15595v2
- **Reading depth:** abstract + metadata; its token-wise construction also inspected as background in D12.
- **Supports:** discrete probability-path design and sampling through learned posterior quantities.
- **Limit:** this record does not support implementation-level or theorem claims beyond the inspected summary.
- **Availability:** not audited.

## D15 — Diffuser

- **Authors/title:** Michael Janner, Yilun Du, Joshua B. Tenenbaum, Sergey Levine. *Planning with Diffusion for Flexible Behavior Synthesis*.
- **Date/version:** first 2022-05-20; v2, 2022-12-21; ICML 2022.
- **Primary URL:** https://arxiv.org/html/2205.09991v2
- **Reading depth:** selected introduction, section 2.1, section 3 overview and 3.1.
- **Supports:** trajectory denoising as planning; conditioning and guidance reinterpretation.
- **Limit:** known control tasks differ from inferring an unknown language/channel; proposed manuscript transfer is ours.
- **Availability:** paper links project/code; repository and license not audited.

## D16 — Diffusion Policy

- **Authors/title:** Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*.
- **Date/version:** first 2023-03-07; v5, 2024-03-14.
- **Primary URL:** https://arxiv.org/abs/2303.04137v5
- **Reading depth:** abstract + metadata.
- **Supports:** visually conditioned multimodal action distributions and receding-horizon execution.
- **Limit:** a policy samples actions; it does not by itself identify world dynamics or historical semantics.
- **Availability:** paper links code/data/training information; artifacts and rights not audited.

## D17 — DIAMOND

- **Authors/title:** Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, François Fleuret. *Diffusion for World Modeling: Visual Details Matter in Atari*.
- **Date/version:** first 2024-05-20; v2, 2024-10-30; NeurIPS 2024.
- **Primary URL:** https://arxiv.org/html/2405.12399v2
- **Reading depth:** selected sections 2.3–3.2, conditioning and imagination training.
- **Supports:** an RL agent trained inside a diffusion world model; visual fidelity can matter to control performance.
- **Limit:** interactive/action-labeled game evidence is unavailable for the historical manuscript; imagery quality is not semantic identification.
- **Availability:** release statement present; code/agents/playable models not audited or run.

## D18 — Diffusion Forcing

- **Authors/title:** Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, Vincent Sitzmann. *Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion*.
- **Date/version:** first 2024-07-01; v4, 2024-12-10.
- **Primary URL:** https://arxiv.org/html/2407.01392v4
- **Reading depth:** abstract; selected sections 3.3 and 4.1–4.2 on uncertainty and tested planning behavior.
- **Supports:** independently varied token noise levels with causal sequence modeling; flexible generation and trajectory guidance.
- **Limit:** the paper's subsequence-bound statement is not independently audited here; do not treat the causal model as an unrestricted bidirectional smoother.
- **Availability:** project link present; implementation not audited.

## D19 — Diffusion posterior sampling

- **Authors/title:** Hyungjin Chung, Jeongsol Kim, Michael T. McCann, Marc L. Klasky, Jong Chul Ye. *Diffusion Posterior Sampling for General Noisy Inverse Problems*.
- **Date/version:** first 2022-09-29; v4, 2024-05-20; ICLR 2023.
- **Primary URL:** https://arxiv.org/abs/2209.14687v4
- **Reading depth:** abstract + metadata.
- **Supports:** approximate posterior sampling for noisy nonlinear measurement problems.
- **Limit:** approximation is explicit; differentiable image measurements differ from unknown discrete rules.
- **Availability:** paper links code; repository and license not audited.

## D20 — Twisted Diffusion Sampler

- **Authors/title:** Luhuan Wu, Brian L. Trippe, Christian A. Naesseth, David M. Blei, John P. Cunningham. *Practical and Asymptotically Exact Conditional Sampling in Diffusion Models*.
- **Date/version:** first 2023-06-30; NeurIPS 2023 proceedings PDF used as stable publication reference.
- **Primary URLs:** https://arxiv.org/abs/2306.17775 ; https://proceedings.neurips.cc/paper_files/paper/2023/file/63e8bc7bbf1cfea36d1d1b6538aecce5-Paper-Conference.pdf
- **Reading depth:** abstract and publication/search metadata; full method and proof not audited. Attempted full-text retrieval returned an internal error.
- **Supports:** weighted-particle SMC with approximate twists and asymptotic correctness for the stated conditional target.
- **Limit:** no finite-particle exactness claim; no automatic guarantee for our proposed discrete/hybrid adaptation.
- **Availability:** not audited.

## D21 — Feynman–Kac steering

- **Authors/title:** Raghav Singhal, Zachary Horvitz, Ryan Teehan, Mengye Ren, Zhou Yu, Kathleen McKeown, Rajesh Ranganath. *A General Framework for Inference-time Scaling and Steering of Diffusion Models*.
- **Date/version:** first 2025-01-12; v5, 2025-07-18.
- **Primary URL:** https://arxiv.org/abs/2501.06848v5
- **Reading depth:** abstract + metadata.
- **Supports:** interacting particles, intermediate potentials, resampling, and reward-based steering of text/image diffusion.
- **Limit:** maximizing a chosen reward is not evidence-calibrated decipherment; no theorem beyond inspected material is relied upon.
- **Availability:** paper links code; repository and license not audited.

## D22 — DIFUSCO

- **Authors/title:** Zhiqing Sun, Yiming Yang. *DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization*.
- **Date/version:** first 2023-02-16; v2, 2023-12-02; NeurIPS 2023.
- **Primary URL:** https://arxiv.org/abs/2302.08224v2
- **Reading depth:** abstract + metadata.
- **Supports:** learned denoising proposals for binary combinatorial solution representations, evaluated on graph optimization tasks.
- **Limit:** useful solution quality does not imply exact optimization or posterior sampling; training problem distribution is material.
- **Availability:** not audited.

## D23 — Finite-automaton constraints for diffusion LM decoding

- **Authors/title:** Meihua Dang, Stefano Ermon. *Constrained Decoding for Diffusion Language Models via Efficient Inference over Finite Automata*.
- **Date/version:** v1, 2026-07-08; preprint.
- **Primary URL:** https://arxiv.org/html/2607.07026v1
- **Reading depth:** selected sections 3.1–4.3, Algorithm 1, and 5.1–5.3.
- **Supports:** exact per-step constrained mean-field inference for DFA constraints; joint rather than independent position sampling.
- **Limit:** §4.1 explicitly disclaims exact full-DLM conditional sampling; NFAs add accepting-path weights. Sudoku constraints omit row/column/box rules. Constraint satisfaction is not semantic correctness.
- **Availability:** not audited.

## D24 — OBSD ancient-script assistance

- **Authors/title:** Haisu Guan, Huanxin Yang, Xinyu Wang, Shengwei Han, Yongge Liu, Lianwen Jin, Xiang Bai, Yuliang Liu. *Deciphering Oracle Bone Language with Diffusion Models*.
- **Date/version:** ACL proceedings, August 2024, pp. 15554–15567; arXiv first 2024-06-02.
- **Primary URLs:** https://aclanthology.org/2024.acl-long.831/ ; https://aclanthology.org/2024.acl-long.831.pdf
- **Reading depth:** publication metadata, introduction, and indexed evaluation excerpt describing generated modern-character OCR and known ground-truth comparison.
- **Supports:** conditional glyph-image assistance using established oracle-bone/modern-character correspondences.
- **Limit:** not blind unknown-language or unknown-cipher decipherment; entire method and dataset audit incomplete.
- **Availability:** paper points to https://github.com/guanhaisu/OBSD ; repository/data rights not audited.

## D25 — Diff-Oracle

- **Authors/title:** Jing Li, Qiu-Feng Wang, Siyuan Wang, Rui Zhang, Kaizhu Huang, Erik Cambria. *Diff-Oracle: Deciphering Oracle Bone Scripts with Controllable Diffusion Model*.
- **Date/version:** first 2023-12-21; v1 referenced.
- **Primary URL:** https://arxiv.org/abs/2312.13631v1
- **Reading depth:** abstract + metadata.
- **Supports:** content/style-conditioned generation for oracle-character data augmentation and recognition.
- **Limit:** “deciphering” in the title includes recognition; not discovery of an unknown linguistic code.
- **Availability:** not audited.

## D26 — FROD current follow-up

- **Authors/title:** Yanbin Hou, Biao Xiong, Guojun Xu, Jianwen Xiang, Cheng Tan, Yanchao Yang, Junwei Zhou. *FROD: Feature Matching Residual Denoising Oracle Bone Decipher*.
- **Date/version:** v1, 2026-09-15; preprint.
- **Primary URL:** https://arxiv.org/abs/2609.17227v1
- **Reading depth:** abstract + metadata.
- **Supports:** explicitly frames assistance as cross-era image translation; reports character-disjoint evaluation.
- **Limit:** fresh preprint; no replication, full split audit, or sentence-level decipherment result established here.
- **Availability:** not audited.

## D27 — Direct Voynich diffusion proposal, not a validated result

- **Author/title:** operator of *Voynich AI — Manuscript Analysis Platform*, research page; named authorship and publication date not established.
- **Date/version:** undated mutable web page accessed 2026-09-21.
- **Primary URL:** https://voynich-ai-production.up.railway.app/research
- **Reading depth:** “Diffusion Model Glyph Analysis” proposal and surrounding research list.
- **Supports:** this platform proposes diffusion over Voynich glyph crops and transformations toward known scripts.
- **Limit:** self-published ideation; crop count, assets, and claimed feasibility not verified; no validated diffusion decoding result inspected.
- **Availability:** no asset availability claim made.

## Search boundary and coverage gaps

Direct-application queries inspected on 2026-09-21 included `"Voynich" "diffusion" model cipher`, `site.arxiv.org "diffusion" "cryptanalysis"`, `site.arxiv.org "Denoising Diffusion" "decipherment"`, `"Voynich" "diffusion model"`, `"diffusion" "substitution cipher" model`, and `"oracle bone" "diffusion" ACL 2024 decipherment`. Ordinary cryptographic “diffusion” in Shannon's sense was excluded from evidence about generative diffusion models. Search hits were followed to primary publications when used technically.

No inspected primary source established diffusion-based Voynich decipherment. This bounded search does not establish worldwide absence or novelty. Additional literature worth auditing before implementation includes discrete guidance, diffusion bridges, block diffusion, corrector theory, variable-length transducer inference, and sampler-calibration failures. We prioritize the distinctions supported above over a claim to exhaustive coverage.

Reading-depth count: 14 papers inspected at selected-section depth, two at excerpt depth (D11, D24), ten at abstract/metadata depth, and one proposal webpage. No complete proof audit or replication is claimed. Full-text retrieval attempts for D20 and a targeted PDF search within D24 failed; the narrower labels above reflect what was actually available.
