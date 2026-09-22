# From readable directions to a mechanism that transfers

Written during JSPACE-0001 calibration, before its development/final causal results. Mathematical deductions below are our analysis under stated assumptions. They are not findings from the manuscript or unreported results of another paper.

## 1. A coordinate swap has a surprisingly simple geometry

Let u and v be two distinct unit readout directions, with c=uᵀv and n=(u-v)/sqrt(2-2c). The minimum-displacement edit exchanging uᵀh and vᵀh is

δ = Dᵀ(DDᵀ)⁻¹(P-I)Dh = -2(nᵀh)n,

where D stacks uᵀ,vᵀ and P exchanges the two coordinates. Thus the operation is a Householder reflection. It preserves ||h||, preserves every direction orthogonal to n, and applying it twice restores h. These facts are tested independently in the geometry suite.

This gives useful controls, but not a semantic theorem. The reflection can be exactly correct geometrically while exchanging two quantities that are not the model's operative country variable. A high baseline donor coordinate can even make the edit push in the opposite direction from the intended historical/semantic story. Therefore a report must include the initial contrast, not only successful examples. Near-parallel country vectors can have individually stable directions and an unstable *difference*; stability of u and v alone is insufficient.

The norm preservation also clarifies a confound: a swap is not equivalent to erasing information by shrinking a hidden state. Conversely, preserving global norm does not imply preserving grammatical competence, local distribution, or unrelated semantic features. An orthogonal rotation can radically change those.

## 2. Average sensitivity is different from a portable internal variable

For prompt x, write g(x)=R J(x), where R is a chosen vocabulary row with final normalization scale folded in. The mean lens is E[g], while the sensitivity energy is E[||g||²]. The decomposition

E[||g||²] = ||E[g]||² + E[||g-E[g]||²]

separates a stable signed effect from context-dependent variation. A small ratio ||E[g]||²/E[||g]||² is not proof that the concept is absent; large positive and negative sensitivities can cancel. Nor is a high ratio proof of a semantic workspace: the residual identity path makes late Jacobians close to the identity, and the final output directions are then inherited almost automatically.

We therefore measure independent-article agreement, country-contrast agreement, token-form agreement, and the raw-output-direction comparator. Larger calibration samples are especially relevant for early layers, where long downstream computations and attention redistribution make derivatives more context-dependent. These diagnostics cannot replace behavioral interventions.

A second limitation is extrapolation. The first derivative predicts only infinitesimal changes. For a finite edit δ and twice-differentiable scalar output f,

f(h+δ)-f(h) = ∇f(h)ᵀδ + ∫₀¹(1-t) δᵀH_f(h+tδ)δ dt.

The remainder can be substantial when the edit changes attention or a gating pattern. A smooth-looking projection plot does not bound it. Actual finite edits and dose curves are the appropriate check. Choosing the dose after seeing desired final answers would change the hypothesis; JSPACE-0001 fixes strength one and treats later dose work as exploratory.

## 3. Raw neuron magnitude is not a stable importance measure

For a SwiGLU MLP, define a_j=SiLU(g_jᵀx)(u_jᵀx) and output y=Σ_j w_j a_j. Scaling u_j by a nonzero scalar c and w_j by 1/c leaves y unchanged while scaling a_j by c. Consequently, ranking neurons solely by |a_j| or |Δa_j| can change under an exactly function-preserving reparameterization.

For this restricted scaling symmetry, |Δa_j| ||w_j|| and the signed projected write Δa_j qᵀw_j are invariant. They are better descriptive quantities when comparing the contribution of changed neurons. They still do not solve arbitrary feature mixing, redundant units, or polysemanticity.

The exact MLP-write difference is Δy=Σ_j w_j Δa_j. Contributions may cancel or reinforce, so Σ_j ||w_j Δa_j||² is not generally ||Δy||². A participation count based on contribution energies is a description of this decomposition, not the number of independent causal mechanisms. We must validate the sum and separately intervene on selected coordinates.

At a Qwen post-MLP residual site, adding an activation change η to selected MLP neurons is mathematically equivalent to adding W_down η to that block's residual output. No nonlinear function occurs between those two sites. This permits efficient, auditable neuron interventions while preserving exactly which neuron coordinates generated the edit. A numerical equality check is still necessary under finite precision.

## 4. A stronger test is transferring a change across query functions

Country→capital and country→currency can have separate answer representations. Transplanting donor activations from another capital question can copy an answer-like state. It is a useful positive control, but a weak test of a reusable country variable.

Use four executions: source country z with query q; donor country z' with q; source z with another query q'; donor z' with q'. At a chosen MLP layer, form η=a(z',q')-a(z,q'). Add its selected coordinates to a(z,q). The intended output is f(z',q), *not* f(z',q'). A currency-derived country change must therefore yield the donor's capital in a capital question. A direct currency-answer transplant will fail this criterion.

This tests an approximate factorization of the country effect across query functions. It remains an assumption, not a guaranteed property of neural representations: a legitimate distributed representation can be nonlinear or query-dependent. A failed transfer does not prove that the model lacks country knowledge. Positive evidence requires same-query controls, matched random-coordinate changes, unrelated copying preservation, and new paraphrases excluded from unit selection.

Select candidate units using natural country changes in one paraphrase, then evaluate on the other paraphrase. Ranking by a statistic that rewards effects across all four relations reduces the chance of selecting only capital-answer units. It does not create a new language or country holdout; all those facts remain shared. This is a controlled mechanistic extension, not a broad reasoning benchmark.

## 5. What relevant external results do—and do not—supply

[Naganna, Sijan, and Kalita (2026)](https://arxiv.org/html/2607.16693v1) study arithmetic in three base Llama models across symbolic, word-problem, and code forms. Their selected-section methods/results use attribution plus activation patching, shared-neuron sets, and cross-format transfer. The paper explicitly restricts circuit discovery to correctly answered prompts and MLPs at the final input position; it leaves attention-routing invariance unresolved. This motivates cross-form rescue tests, not importing their arithmetic success rates into Qwen geography or decipherment.

[Marinov et al. (2026)](https://arxiv.org/html/2606.14347v1) examine bilingual directions with a covariance-adjusted inner product. The inspected methods and limitations concern eight Latin-script languages from two Indo-European families and filtered single-token translation pairs. Their own discussion reports limited, context-sensitive generation control despite clear geometric structure. This is a direct reason to separate separability plots from useful causal manipulation. A covariance metric estimated from our 32 chosen words would be low rank and task selected; it cannot be substituted for their vocabulary covariance without a new method definition.

The [Anthropic multilingual circuit study](https://transformer-circuits.pub/2025/attribution-graphs/index.html) provides examples of shared operation/operand features with language-specific routing, while acknowledging mechanisms missed by the circuit approximation. Its [September 2025 update](https://www.transformer-circuits.pub/2025/september-update/index.html) also investigates how apparent cross-language feature overlap changes with context length. These are reasons to control surface form, language, position, and context length separately. Shared active features can reflect task or positional context rather than common meaning.

[ALICE v2](https://arxiv.org/html/2509.07282v2), already reviewed in this repository, supplies a supervised substitution-cipher precedent with an explicit permutation head. Its symbol pooling and fixed cipher assumptions offer a concrete path to inspectable decoding; they do not license imposing a bijection on Voynich. Early-exit guesses remain observations about intermediate readout, not a proof of an executed algorithm. The relevant transfer is to build competent known-system teachers and compare executable internal variables against alternatives.

## 6. Why inspecting a model cannot manufacture a missing historical anchor

Suppose two candidate historical systems induce the same distribution over every observation available to the learner. A model trained only on those observations, with independent random seed, cannot distinguish which system generated them merely by inspecting its own activations. Its activations are downstream computations of the same evidence. They may expose useful sufficient statistics or a compact algorithm, but they cannot break an observational symmetry without additional assumptions or information.

Pretrained weights can add information, but that information comes from external training and its priors. It must be treated as an additional source, not as something discovered uniquely in the manuscript. The country labels in this experiment are externally supplied and independently known. No corresponding verified country/plant/action labels are currently available for Voynich.

For a linguistics solver, the ambitious next object is an identifiable *family of decoding systems*: explicit segmentation, mapping, morphology/syntax, and optional world/action constraints, with uncertainty over equivalent descriptions. A learned representation can propose variables and speed search. Exact execution and independent observations must determine whether those variables explain the historical artifact. A world model, diffusion sampler, VLA-style action policy, or activation oracle is useful only to the extent it improves that evidence-constrained process.

The present investigation therefore has two distinct outputs: a numerical/causal method qualified on known concepts, and constraints on the kind of internal representation worth seeking in a future decipherment model. Neither output is itself a translation.
