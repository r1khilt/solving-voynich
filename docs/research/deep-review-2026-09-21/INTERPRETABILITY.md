# Mechanistic interpretation: deeper review and falsifiable next steps

Reviewed 2026-09-21. **Research and design only; no new model run or decipherment.** The [source ledger](interpretability.sources.json) contains 24 distinct primary papers or original research reports, exact versions where verified, all authors as given in the cited version, read depth, limitations and decisions. Nineteen sources had body text inspected: one full short note and eighteen selected-section reads, some limited to retrieved primary-source excerpts. Five are explicitly abstract-only. This is a focused review, not a claim to have exhaustively read the field.

Four sources are verifiably newer than April 2026: identifiable SAEs on May 29, reproducible subspaces on June 10, natural-language autoencoders on May 7, and interference-weight analysis on August 21. Search-engine relative dates were not used as publication evidence. A later OpenReview CLT critique was blocked; M24 cites the accessible original February author report instead.

The resulting change in direction is specific: **recover a compact representation whose updates and joint predictions survive new encodings, then test whether that representation causally mediates the model's future behavior.** A hidden-state readout, a named feature, a low reconstruction error and a steerable answer each test different things.

## What our existing experiments actually demand

| Prior result | Missing capability | Literature-informed next question |
| --- | --- | --- |
| [EXP-0004](../../experiments/EXP-0004-results.md): oracle-supervised state readout works, probe-direction intervention does not | A readout is not necessarily a causal coordinate system | Can a candidate representation support selective, multi-step interchange? |
| [EXP-0006](../../experiments/EXP-0006-results.md): late learned steering nearly matches an output-weight span | Manipulating an answer does not locate the natural algorithm | Can a change affect later predictions with output interventions held as a negative control? |
| [EXP-0008](../../experiments/EXP-0008-results.md): familiar partitions work selectively; new keys fail; copy success is mainly one step | Reusable updates and joint-future sufficiency | Can one inference procedure adapt without labels to genuinely new encodings, using explicit update constraints? |
| [EXP-0009](../../experiments/EXP-0009-results.md): broad early synthetic patches help; manuscript confirmation fails | Selectivity and a defined latent intervention | Can we move a particular predictive variable while preserving unrelated ones, across fresh continuations? |

A useful correction to our intuition: a transformer's final vector need not be its complete state for future computation. Later predictions also access earlier representations through attention. Searching only the final residual can therefore miss temporally relevant information. Conversely, concatenating everything can improve a probe by making it larger without revealing a usable state. We must compare compact representations at matched dimensions and distinguish diagnostic access to all layers from an implementable recurrent update.

## Theoretical target: three different objects

Let \(x_{\le t}\) be the observed prefix, \(h_t\) a model's computational state, and \(z_t=\phi(h_t)\) a candidate explanatory state. For a transformer, the chosen \(h_t\) must include whichever prefix activations or cached keys/values the proposal actually needs. Write \(F_a\) for a proposed update after symbol \(a\), and \(q(w\mid z_t)\) for the probability of an entire future string \(w\).

Our proposed evidential ladder is:

1. **Readable information:** a diagnostic decoder can estimate something from \(h_t\). With oracle labels this is supervised calibration.
2. **Predictive state:** \(z_t\) preserves useful joint-future information and has consistent updates, \(\phi(h_{t+1})\approx F_{x_{t+1}}(\phi(h_t))\).
3. **Causal abstraction:** an intervention on \(z_t\) corresponds to a specific intervention on the model, including its later behavior.
4. **Generator identification:** the recovered process is identified within an explicit model class, up to justified equivalences.
5. **Historical decipherment:** a constrained mapping produces independently checkable readings of the manuscript.

These are our proposed claim levels, not a theorem that every level can be reached. An explanation of our trained predictor is not automatically an explanation of the manuscript's author. Multiple generators can induce the same observations; a model can approximate those observations with an algorithm different from the source process.

Known-generator belief geometry is useful calibration, but its affine readouts use supplied posterior labels. The RRXOR example also distinguishes information spread across layers from information available at the output. [M14, Shai et al.](https://arxiv.org/html/2405.15943v2). Our failed blind RRXOR result is therefore evidence against that tested pipeline, not a disproof of every belief-state approach.

The Belief State Transformer is a different proposal: bidirectional training constrains a forward representation. Its sufficiency theorem assumes exact conditional predictions for every supported prefix/suffix. That assumption is much stronger than achieving a good average validation loss on a tiny corpus. [M19, Hu et al., v3](https://arxiv.org/html/2410.23506v3). It is a candidate objective ablation, not an identification guarantee.

## What the newer methods add, and what they do not

| Sources | Practical contribution | Decision for this project |
| --- | --- | --- |
| [M01 DAS](https://arxiv.org/html/2303.02536v3), [M02 causal abstraction](https://jmlr.org/papers/v26/23-0058.html), [M03 critique](https://openreview.net/pdf?id=Ebt7JgMHv1), [M04 reply](https://arxiv.org/html/2401.12631v1) | A language for specifying interventions and competing explanations; the nullspace critique is contested | Retain alignment, but fit and confirm against explicit transition counterfactuals. Never use one steering score or one nullspace test as the verdict. |
| [M05 patching methodology](https://arxiv.org/html/2309.16042v2), [M21 causal scrubbing](https://www.alignmentforum.org/s/h95ayYYwMebGEYN5y/p/JvZhhzycHu2Yd57RN) | Corruption choice matters; an interpretation implies behavior-preserving replacements | Design donor equivalence classes on synthetic ground truth first. Manuscript “same state” is a hypothesis, not a label we possess. |
| [M06 circuit tracing](https://www.transformer-circuits.pub/2025/attribution-graphs/methods.html), [M08 attention interactions](https://www.transformer-circuits.pub/2025/attention-qk/index.html), [M09 crosscoders](https://www.transformer-circuits.pub/2024/crosscoders/index.html) | Shared feature descriptions and local computational graphs | Small optional comparator. Attention routing, error terms and original-model interventions stay in the evaluation. |
| [M07 toy unfaithfulness](https://www.transformer-circuits.pub/2025/faithfulness-toy-model/index.html), [M24 CLT critique](https://www.lesswrong.com/posts/6CS2NDmoLCFcEJMor/cross-layer-transcoders-are-incentivized-to-learn-unfaithful) | Repetition and sparsity can produce explanatory shortcuts absent from the original mechanism | Especially relevant to our repeatedly sampled corpus. Compare finite interventions and intermediate steps, not just output reconstruction. |
| [M10 SAEBench](https://arxiv.org/html/2503.09532v1), [M11 canonical units](https://arxiv.org/abs/2502.04878), [M15 random-transformer SAEs](https://arxiv.org/html/2501.17727v1), [M22 seed dependence](https://arxiv.org/abs/2501.16615) | Different evaluation questions need different metrics; convincing descriptions and one dictionary are insufficient | Keep independent predictive, causal and stability metrics. Do not use English feature naming as a manuscript correctness signal. |
| [M12 identifiable SAEs](https://arxiv.org/pdf/2605.31245), [M13 reproducible subspaces](https://arxiv.org/html/2606.12138v1) | Code stability and subspace agreement can reveal distinctions missed by decoder-vector matching | Candidate signed/multistep SAE versus TopK/PCA comparison on synthetic cases. Evaluate both individual features and spans. |
| [M18 Jacobian SAEs](https://arxiv.org/html/2502.18147v1), [M17 interference weights](https://www.transformer-circuits.pub/2026/interference_effectiveness_helpfulness/index.html) | Study connections and functional effects rather than only activation sparsity | Promising after we have a task with demonstrable transfer. Confirm local gradient predictions using finite interventions. |
| [M20 automata extraction](https://proceedings.mlr.press/v80/weiss18a/weiss18a.pdf) | Refine proposed states using concrete counterexamples | Prefer explicit update models with recorded disagreement strings to attractive clusters alone. An approximate oracle is not an equivalence proof. |
| [M23 weight-sparse transformers](https://arxiv.org/html/2511.13653v1) | Architecture can simplify circuits before interpretation | One bounded synthetic comparator, matched by predictive quality as well as resources. Do not confuse zero weights with cheap dense-optimizer training. |
| [M16 natural-language autoencoders](https://transformer-circuits.pub/2026/nla/) | A language bottleneck can describe activations using pretrained language-capable modules | Defer for decoding. Our Voynich-only predictor does not already contain a known mapping to English meanings. |

Two distinctions prevent misleading implementations. A local attribution graph can freeze attention even though its ordinary replacement model runs attention normally. Also, a sparse Jacobian is a local sensitivity description; finite changes may cross nonlinear boundaries. Neither a graph's neatness nor derivative sparsity establishes the temporal mechanism. These are reasons to add tests, not reasons to abandon the tools.

Natural-language explanations deserve extra restraint here. A verifier who already knows English can test whether a feature activates on English constructions. We have no comparable semantic labels for Voynich. The explanatory language model could supply a persuasive interpretation that the tiny manuscript model never learned. Measured reconstruction or usefulness as a search aid must be kept separate from translation truth.

## Proposed pipeline, with stopping gates

This is a design recommendation. Exact datasets, budgets and thresholds require a separate experiment registration before any run. No current final pool is fresh: the previous synthetic evaluations have been exposed.

### A. Establish that the proposed information is observable

Build fresh controls where distinct hidden beliefs have identical immediate output distributions but different joint futures. Include delayed parity, noisy state transitions, variable copy lags, nonstationary mixtures, and deliberately observationally equivalent hidden descriptions. Keep new emission keys and new mechanism families separate. An alphabet permutation alone does not define a genuinely new distribution for every generator.

For each controlled process, compute oracle distinguishability over short future strings. Measure how much ambiguity remains even for the oracle; do not require perfect recovery where observation cannot support it. The nonidentifiable controls should return uncertainty or an equivalence class, not fabricated unique labels.

### B. Separate failed prediction from failed extraction

First compare the teacher predictor with the known oracle on familiar and unseen keys. EXP-0008's frozen predictor itself deteriorated on new keys, so a perfect extractor could not simply assume a faithful teacher. Compare direct text-only explicit models, dense transformers and a small weight-sparse or recurrent comparator under matched data exposure. Report their actual prediction gaps before assessing extraction.

For each teacher, compare an output-distribution baseline, final-layer representation, a dimension-matched multi-layer representation, and any proposed recurrent summary. Fit extraction using visible sequences only. Hidden state labels may score a fully frozen result; they must not choose the clustering, rank or architecture.

### C. Learn update rules and score joint futures

Fit \(\phi,F_a,q\) with an explicit state bottleneck and a fixed complexity sweep on development data. A cluster must carry information that updates reproducibly after symbols, not merely summarize the next answer.

Score joint future strings separately at several lengths, as well as individual offsets. For a manageable synthetic alphabet, exact short-string enumeration can reveal parity dependencies invisible to separate marginals. Also report probability normalization, calibration and the gap to a direct observed-text baseline. Reject a representation whose average gain is entirely the first symbol, as happened in the copy control.

Add counterexample search: find prefixes assigned similar states whose continuation distributions disagree, and retain those witness strings. Adaptive refinement belongs to development; final evaluation remains untouched. New-key adaptation must be declared explicitly, including its available text and fitted parameters. Label-free per-key fitting is not zero-shot transfer.

### D. Test a causal state rather than a donor-shaped answer

The decisive synthetic comparison should use donor/recipient cases matched on immediate predictions and measured nuisances but separated in later joint predictions. A whole-prefix transplant is a positive reference; it does not count as selective success.

A proposed intervention must:

- change the relevant future behavior after new observations, with no patch to future positions;
- preserve predeclared unrelated variables and behavior on matched-state pairs;
- generalize to new prefixes, continuation strings, keys and independently trained models under the stated alignment protocol;
- agree with the proposed update: advancing the patched model after symbol \(a\) should match the intervention corresponding to the updated candidate state \(F_a(z)\).

That last test is stronger than showing that a patch changes several outputs. It ties the change to an explicit transition rule. Conditions imposed algebraically by the intervention, such as a projection's idempotence, are implementation checks rather than independent scientific evidence.

Controls must include output-weight steering, equally sized random subspaces with matched displacement, shuffled counterfactual supervision, an untrained backbone, wrong donors matched on observable nuisance statistics, unchanged recipients, identity patches and full restoration. Where possible add covariance-aware perturbations and both directions of interchange. Use held-out, naturally occurring donor activations; assess whether a proposed hybrid state lies within supported synthetic combinations. Arbitrary off-manifold movement can manufacture behavior.

For transformers, explicitly identify which cached keys/values or earlier components change. A single final-layer patch cannot generally carry a state into the future merely because its immediate logits are correct. Compare the proposed state variable with the broad early-component reference from EXP-0009.

### E. Only then test unlabeled manuscript hypotheses

Freeze an analysis that succeeded on recoverable and unrecoverable synthetic controls. Use manuscript development leaves to generate hypotheses about measurable properties: separators, glyph sequences, repetition distance, transcription alternatives, section/scribe metadata where independently established. Avoid inventing semantic state labels.

A manuscript success at this stage would mean a reproducible predictive abstraction and a causal account of our model on held-out data. To prefer a historical generator, compare competing explicit mechanisms with complexity penalties and external constraints independent of the fitted explanation. A claim of plaintext still requires a constrained decoding rule and independent validation under the repository protocol.

## Falsification and resource priorities

The following are **proposed failure criteria**, not retroactive changes to earlier experiments:

| Proposed claim | Observation that defeats it |
| --- | --- |
| Information beyond immediate output | No advantage over matched output features on delayed/joint predictions |
| Complete predictive state | Good first-symbol results but persistent later/joint errors; update inconsistency |
| Encoding transfer | Gains disappear on genuinely new keys after a declared equal adaptation budget |
| Selective causal variable | Broad patches work, but narrow patches fail controls or damage independent variables |
| Natural computation | Output-span or shuffled/random/untrained controls explain the same effect |
| Robust feature identity | Individual names change across seeds without stable function; report only a subspace if that is what survives |
| Faithful replacement circuit | Accurate clean reconstructions but wrong responses to original-model interventions |
| Historical mechanism | Several incompatible source processes fit equally well or need unlimited exceptions |

The highest-value next work is the joint-future/transition benchmark and its controlled causal test. It reuses our tested intervention machinery and can reveal whether a more complex interpretability tool has anything useful to extract. A sparse architecture, an iSAE and a Jacobian basis should be optional bounded ablations of that question, not three independent open-ended projects.

Large pretrained-model CLT or NLA training is deferred: its compute requirements and language supervision do not match our current inference problem. Our machine's available memory is useful for cached multi-layer activations and carefully paired controls; filling memory is not an experimental objective.

No reviewed source supplies an established ciphertext-only Voynich decoder. The promising opportunity is a stricter bridge from observable sequence constraints to transferable, inspectable rules. This review leaves implementation, numerical thresholds, runtime profiling, proof verification for the longer theoretical papers, and independent replication of recent preprints as explicit next work.
