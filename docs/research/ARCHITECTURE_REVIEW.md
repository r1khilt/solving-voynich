# Architecture review: an interpretable Voynich language model

Reviewed 2026-09-20 PDT / 2026-09-21 UTC. This is a design review, not a benchmark result. Source statements below are attributed to their authors; suggested transfers to Voynich are project hypotheses. No source reviewed here demonstrates that a modern architecture can decipher Voynich.

## Recommendation

Build a small, instrumented, dense causal transformer and a controlled family of variants. The ambitious part should be the quality of the measurements and causal experiments, not the number of mechanisms combined in the first model. Use random initialization and Voynich transcription only for the manuscript predictors. Keep synthetic calibration models separate.

The initial candidate is four layers, width 192, four attention heads, SwiGLU hidden width 512, pre-RMSNorm, rotary positions, full causal attention, and an ordinary additive residual stream. At a 64-unit vocabulary with untied embeddings this is approximately 1.80 million parameters, before optional modules; the actual vocabulary and implementation determine the exact count. This is a planning estimate, not a measured optimum. Start with 256 transcription units of context; compare 512 and shorter contexts after measuring data coverage. Include a two-layer small dense model and a two-layer attention-only control.

The first research extensions worth testing are **two-token prediction**, **per-head query/key normalization**, and **head-specific attention-output gating**. Implement them as independent switches and retain a plain control. Add a third prediction horizon only if the second is useful. Do not infer that a combination of individually promising papers is automatically superior.

Exact executable settings, implementation status, and measurements belong in the committed configs and run manifests. The architecture above is a recommended candidate; it does not assert that every suggested ablation has already been implemented.

## What is actually current in DeepSeek

DeepSeek-V4 is an official release, not an assumed future model. DeepSeek's announcement is dated **2026-04-24** and identifies V4-Pro (1.6T total / 49B active parameters) and V4-Flash (284B / 13B), both with million-token context. The official technical report describes compressed sparse attention, heavily compressed attention, mHC residual connections, and Muon, retaining MoE and multi-token prediction. These are designs for a radically different scale from this project. [Official launch](https://deepseek.com/en/news/v4-preview/), [technical report, version 1](https://arxiv.org/html/2606.19348v1), [official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro).

The official site also lists **V4.1-Flash, released 2026-09-10**, with a causal encoder-decoder design and native vision support. Its announcement says the older V4-Pro API identifier began routing to V4.1-Flash on September 14. Thus API aliases are not reliable experiment model-version identifiers. No DeepSeek API was called for this review. The V4.1 announcement was inspected for currency; its full architecture report was not analyzed here. [Official V4.1 announcement](https://deepseek.com/en/news/deepseek-v4-1-flash/).

Bibliographic caveat: the inspected V4 arXiv landing page displays submission on April 26, 2026, despite its identifier beginning `2606`. Preserve the exact versioned source URL and displayed date; do not silently infer a corrected publication date from the identifier.

## Transfer assessment

Decisions in this table are engineering recommendations for this corpus, not findings established by the cited authors.

| Component | What the primary source supports | Voynich decision and test |
| --- | --- | --- |
| Pre-RMSNorm, RoPE, SwiGLU, no QKV biases | Qwen3 uses these in its dense models. | **Adopt** as a compact modern default. Preserve a simpler attention-only model to expose the contribution of nonlinear feed-forward computation. [Qwen3 report, §2](https://arxiv.org/html/2505.09388v1#S2). |
| Full independent attention heads | The transformer-circuits framework separates QK selection and OV information transport and studies small attention-only models. | **Adopt** full causal MHA with separately accessible Q, K, V and head contributions. Avoid shared/compressed KV paths initially. [Elhage et al., 2021](https://transformer-circuits.pub/2021/framework/index.html). |
| Query/key normalization | Henry et al. study L2-normalized Q/K with learned scaling in low-resource translation. Qwen3 uses QK-Norm for training stability. | **Ablate** per-head RMS Q/K normalization; name its exact formula. It is not identical to Henry's original method. It could improve optimization but changes the scale information available to attention. [Henry et al., 2020-10-08](https://arxiv.org/abs/2010.04245), [Qwen3, 2025-05-14](https://arxiv.org/abs/2505.09388). |
| Multi-token prediction (MTP) | Gloeckle et al. find benefits in several settings, including induction experiments spanning 1M–1B nonembedding parameters. DeepSeek-V3/V4 use sequential MTP modules. | **Prioritize an ablation** with a direct +2 prediction head. The small-model evidence is especially relevant, but corpus and supervision differ. Do not call a simplified parallel-head implementation an exact DeepSeek replica. [Gloeckle et al., 2024-04-30, §4](https://arxiv.org/html/2404.19737v1#S4), [DeepSeek-V3, 2024-12-27](https://arxiv.org/abs/2412.19437). |
| Sigmoid gating after attention | Qiu et al. compare many gating placements; head-specific SDPA-output gates improve their large dense/MoE models. Their paper distinguishes scalar headwise from vector elementwise gates. | **Ablate** a scalar gate for each head and query position. It is cheap and can itself be patched; keep the ungated model. A low gate does not mean the corresponding text is filler. [Qiu et al., 2025-05-10](https://arxiv.org/html/2505.06708v1), [official implementation](https://github.com/qiuzh20/gated_attention). |
| Muon | Moonshot reports improved compute efficiency over AdamW in its tested scaling setup; DeepSeek-V4 uses it. | **Defer to an optimizer ablation** after a stable AdamW reference. Changing optimization preserves the model's causal graph, making this more attractive than many structural additions. Verify implementation and matrix/non-matrix parameter treatment before use. [Liu et al., 2025-02-24](https://arxiv.org/abs/2502.16982), [Moonshot implementation](https://github.com/MoonshotAI/Moonlight). |
| Compressed/sparse attention, MLA, GQA | Frontier models reduce attention and KV-storage cost, particularly at long contexts. | **Defer.** Our first contexts are hundreds of units. Compression complicates mapping a causal effect back to a particular source position; an efficiency need has not been demonstrated. [DeepSeek-V4 §2](https://arxiv.org/html/2606.19348v1#S2). |
| mHC | DeepSeek expands the residual stream and constrains learned stream mixing to improve signal propagation at scale. | **Defer.** Multiple dynamically mixed residual streams complicate path analysis, while a four-layer model does not inherit the paper's demonstrated scaling bottleneck. [mHC, 2025-12-31](https://arxiv.org/abs/2512.24880). |
| MoE and hash routing | V4 uses routed/shared experts and token-ID-based routing in early layers. | **Defer.** Sparse expert specialization is not evidence for historical symbol categories; routing would add another latent mechanism to identify with very little text. [Official V4 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro). |
| Gated DeltaNet / Kimi Delta Attention | Qwen3-Next combines recurrent linear attention with gated full attention. Kimi Linear compares a KDA/MLA hybrid against full MLA. | **Defer to a separate architecture family.** Recurrent state may be useful for finite-state mechanisms, but full token-to-token causal paths are simpler to inspect first. These reports do not establish superiority at Voynich scale. [Qwen3-Next model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct), [Kimi Linear, 2025-10-30](https://arxiv.org/abs/2510.26692). |
| Recurrent depth / weight sharing | Looped-transformer work shows that repeated layers can approximate deeper models on several synthetic reasoning tasks. | **Later ablation.** Parameter sharing is appealing for a small corpus. Every loop needs separately addressed hooks, and comparisons need both matched-parameter and matched-compute controls. Repeated computation is not evidence of a historical cipher state machine. [Saunshi et al., 2025-02-24](https://arxiv.org/abs/2502.17416). |
| Engram conditional lookup memory | DeepSeek introduces hashed n-gram memory and reports large-model gains with a 27B-parameter memory module. | **Study through controls first.** An explicit n-gram predictor measures local lookup effects without embedding a large memory into the research model. A small lookup-residual branch is a later test, not a discovery of the manuscript's codebook. [Cheng et al., 2026-01-12](https://arxiv.org/abs/2601.07372), [official repository](https://github.com/deepseek-ai/Engram). |
| Low-precision quantization and distributed kernels | Frontier systems use these for resource efficiency. | **Defer.** Preserve float32 analysis and straightforward numerical comparisons until hardware profiling shows a need. Mixed precision training, if introduced, needs recorded settings and full-precision analysis checks. |

The Qwen3.5 official announcement confirms that Qwen continued combining recurrent gated attention, MoE, and MTP in 2026. That is useful corroboration of research direction; it is not a controlled test of which component helps our task. [Qwen3.5 announcement, 2026-02-15](https://qwen.ai/blog?id=qwen3.5).

## Why tiny-data performance needs its own evidence

The frontier papers' broad-language benchmarks are not the objective here. The manuscript supplies one highly constrained historical corpus, not millions of independent documents. More passes through its text do not create new evidence. Data-constrained scaling experiments find diminishing returns from repeated data, but their quantitative laws were fitted at different scales and should not be numerically extrapolated into this project. [Muennighoff et al., 2023-05-25](https://arxiv.org/abs/2305.16264).

A model can improve loss by learning transcription conventions, line boundaries, token frequency, local copying, or section differences. These are valid predictive mechanisms but do not by themselves identify plaintext or the manuscript's historical production process. This is why the initial predictor should expose rather than conceal its computation.

The closest small-model evidence for MTP deserves a specific qualification. Gloeckle et al.'s induction comparison reports two seeds and selects epochs using the test metric; its benefit depends on dataset and model size, and stronger training data can remove much of the advantage. We should use a development split for checkpoint selection and never copy their test-oracle choice. Our simplified independent horizon heads also differ from their full experimental heads. [MTP §4.1 and Appendix J](https://arxiv.org/html/2404.19737v1#S4.SS1).

## Proposed model contract

This section specifies project design choices; it is not a claim of novel architecture or observed performance.

**Inputs.** Start with reversible transcription-level units, explicit spaces and line markers, and explicit page segmentation. Keep the original transcription and uncertainties in a separate immutable source record. EVA characters are transcription symbols, not automatically atomic manuscript glyphs. Do not use an off-the-shelf English subword tokenizer. Alternative ligature-aware or word-based views should be separately versioned experiments, not silent preprocessing changes.

**Backbone.** For each layer, compute an attention update from normalized residual state and add it back; then compute a SwiGLU update from the new normalized residual state and add that back. Apply a final norm and an untied linear readout. Independent embedding and readout weights cost little at this vocabulary size and avoid an unnecessary geometric constraint. Weight tying remains an explicit ablation.

**Attention.** Use ordinary masked softmax and full MHA. Apply optional per-head RMS normalization to Q/K before rotary positions, recording epsilon and affine parameters. Optional headwise gating should multiply the head's attended value vector before the output projection. Record gate input, bias choice, initialization, and whether scores are scalar or elementwise. Do not combine QK normalization with a second undocumented temperature rule.

**Auxiliary objective.** Let `h[t]` depend only on `x[0:t+1]`. The main head predicts `x[t+1]`; a distinct optional head predicts `x[t+2]`. A +3 head can be added later. An initial candidate loss is `CE(+1) + 0.2 * CE(+2)`; the weight is a proposed hyperparameter, not literature-derived or tuned here. Compute each horizon over its own valid targets, exclude padding, and never predict across a page/split boundary. Auxiliary heads must not receive their target or intervening future tokens as inputs. Report the main next-unit metric separately from the combined training objective.

**Analysis execution.** Dropout must be disabled in analysis mode. Training and analysis use the same learned weights and mathematical forward computation. Caching is opt-in; it must not change outputs. Save checkpoints at initialization and selected training stages, with optimizer/RNG state where continued training is supported. Record tokenization, data/split hashes, exact config, versions, seeds, precision, steps, and wall time.

### Hook surfaces

Every site should support read-only capture and value replacement without editing model code. Cache only requested sites to control memory use.

| Surface | Reason to expose it |
| --- | --- |
| Token embedding output | Separate input identity effects from later computation. |
| Residual before attention, after attention, after MLP | Localize which stage changes a prediction. |
| Norm input/output, including final norm | Account for normalization when interpreting contributions. |
| Q/K before and after normalization/rotation, plus V | Distinguish content selection, positional selection, and transported information. |
| Masked attention logits and attention probabilities | Analyze selection with causal masking made explicit. |
| Per-head attended value vectors | Patch a head before output projection. |
| Optional gate values and gated vectors | Separate gate-mediated effects from attention selection. |
| Per-head contributions after each slice of the output projection | Verify their sum reproduces the attention update and support targeted head interventions. |
| MLP gate/up branches, product activation, down-projected output | Inspect and intervene on nonlinear computation. |
| Final residual and prediction heads | Relate interventions to actual logits and horizon-specific behavior. |

TransformerLens's identity-hook/caching model is a useful API precedent. Implementing similarly named hooks does not establish TransformerLens checkpoint compatibility; an actual adapter would need numerical parity checks. [Official hook documentation](https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.hook_points.html).

## Mechanistic interpretation plan

The attention-only control is valuable because fewer mechanisms mediate behavior. The original circuits work demonstrates detailed decomposition in simplified small models, including induction-like copying. Our normalized, position-aware variants retain complications absent from some of those derivations; do not treat raw weight products as a complete explanation. [Transformer circuits framework](https://transformer-circuits.pub/2021/framework/index.html).

Start from a behavior measured on reserved development material: for example, a repeated pattern changes the probability of a subsequent unit, or the same local context behaves differently near a line boundary. Construct matched clean and altered contexts that differ in the proposed causal factor while controlling length, frequency, and token positions where feasible.

Patch clean activations into the altered run at selected positions and components, then measure the change in the prespecified target log-probability or logit difference. Include identity patches, random/mismatched donor patches, direction reversal, and an intervention that should be irrelevant. Preserve the original clean/altered performance gap: normalized recovery can become unstable when that gap is small. Report absolute effects as well.

Corruption and metric choice can materially change activation-patching conclusions. Results therefore need several matched pairs and multiple seeds, not an appealing heatmap. Zero ablation is useful as a diagnostic but can place the model outside its normal activation distribution; resampling interventions provide a complementary test. [Zhang and Nanda, 2023-09-27; revised 2024-01-17](https://arxiv.org/html/2309.16042v2).

Sparse autoencoders or transcoders can be added after the predictor and basic causal pipeline work. Transcoders approximate a component's input-output map and have shown advantages on interpretability assessments; neither method makes feature labels automatically correct. Other work demonstrates fragile SAE interpretations under small input changes. Measure reconstruction, behavioral faithfulness, donor controls, feature stability, and intervention effects before attaching a historical meaning. [Paulo et al., 2025-01-31](https://arxiv.org/abs/2501.18823), [Li et al., 2025-05-21](https://arxiv.org/abs/2505.16004).

## Cipher-related calibration

ALICE is particularly relevant adjacent work: an encoder-only transformer solves substitution cryptograms, with a Gumbel-Sinkhorn head explicitly representing bijective mappings and layerwise analyses. It uses paired encryption/decryption supervision and assumes a substitution-cipher family. It is not evidence that a ciphertext-only next-unit predictor learns a decoder. [Shen and Smith, 2025-09-08](https://arxiv.org/abs/2509.07282).

Use that distinction to separate two future synthetic tests:

1. **Predictor interpretation:** give a randomly initialized model only synthetic ciphertext. Test which known generator properties its internal computations recover without plaintext supervision.
2. **Supervised decoder calibration:** give a separate model paired data and an explicit cipher family, testing recovery on held-out keys, plaintexts, and generator families. This tests our tools under favorable identifiability conditions.

Include monoalphabetic substitution as an easy positive control; homophonic substitution, finite-state encodings, controlled null insertion, copy/mutate generators, and randomized text as harder or negative controls. Keep synthetic plaintext out of the Voynich-only model's training set. A failed calibration is informative: it can show that the architecture or interpretability method cannot recover a known mechanism at matched data scale.

Do not impose ALICE's bijection on the manuscript without independent evidence. A head constrained to represent a one-to-one substitution can only discover the best mapping within that assumption, even if the assumption is historically wrong.

## Selection and validation gates

Before substantive training, freeze an experiment record and page-level train/development/test assignment; keep recto/verso or larger physical blocks together where the claim requires it. Keep transcription alternatives of the same text in the same split. Audit exact and near-duplicate material, and do not put editorial metadata or source comments into the language-model stream. Choose a primary transcription first; alternative transcriptions are robustness tests, not extra independent manuscript samples.

Check the following before interpreting any learned behavior:

- Future-token changes leave earlier logits unchanged in evaluation mode.
- Padding and page boundaries never supply a target or context across the intended isolation boundary.
- Cached and uncached forward passes agree; identity patches do not change outputs.
- Summed per-head projected contributions equal the attention update to numerical tolerance.
- Each horizon aligns with its stated target and receives gradients only through valid causal paths.
- Initialization and checkpoint reload reproduce outputs under the recorded environment.
- A tiny overfit smoke test verifies learnability; it is not evidence of manuscript structure.

Compare unigram/character n-gram models and copying controls using the same representation and held-out units. Evaluate both overall next-unit cross-entropy in bits and per-page variability. Also inspect errors by line position, token frequency, section, and transcription uncertainty, retaining an aggregate primary metric to avoid choosing only attractive slices.

Use development data for model size, checkpoint, context, and ablation selection. Prefer the smallest model on a predeclared predictive/interpretability tradeoff; do not search thousands of configurations against one small holdout. Compare at matched token exposure and report compute differences rather than confusing more optimization with a better architecture. Require multiple seeds before treating an apparent architectural advantage as stable. Reserve test pages for a frozen final comparison.

## Research limits and next state

This is a targeted primary-source review covering relevant architecture families and interpretability practice through the access date. It is not an exhaustive survey of every 2026 model or an independently reproduced literature result. Several sources were inspected through abstracts, model cards, or targeted sections rather than full-paper replication; the linked source and specificity of each claim reflect that scope. Public repositories were read as documentation, not executed or installed. No paid research/training API was used.

The immediate deliverable is an auditable model/training/analysis implementation with these controlled options and a reproducible corpus pipeline. Subsequent model ranking, synthetic calibration, and historical interpretation require separate experiment records and measured results.
