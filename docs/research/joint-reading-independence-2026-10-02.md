# Change the reading and emission inventory together without a key-likelihood lattice

Own derivation and prospective finite qualification; no trained sampling, source-archive scoring, recovery or manuscript reading has been performed here. The independent 96M SOURCE-ACTION-TRAIN-002 campaign continues with its existing gates unchanged.

## Why this follows from the actual failures

The closed [tempered recovery diagnosis](../experiments/TEMPERED-RECOVERY-DIAG-001-results.md) found the generating used emission inventory missing from every final bank on its four exposed known-answer cases. Its known-key source scores exceeded every bank best. Merely transposing row labels preserves the inventory and cannot fix those omissions. The earlier [auxiliary key swap](auxiliary-key-swap-2026-10-01.md) already proved a cheap terminal joint move, but that move ALSO preserves the inventory. The existing source-state, guided-particle, lookahead and whole-key neural proposal programs are not new ideas to rerun under new names.

The [reading-action environment](source-action-inverse-2026-10-02.md) instead constructs a reading and the visited dictionary entries together. A complete stochastic proposal can replace many emission units, segmentation choices and source labels in one move while literally reproducing all records. Its returned path probability cannot be substituted for a whole-key marginal. Keeping the reading as part of the inference state is the missing distinction that makes a different correction available.

This concerns the fixed non-erasing one/two-glyph, shared-dictionary calibration family. It does not establish that this family represents Voynich. The earlier empirical source is retained because the four diagnosed known keys already score strongly; this is evidence for addressing search first on THOSE cases, not a universal source-quality theorem.

## Source review and limits

[Chiang et al. 2010, §3.2 and §5](https://aclanthology.org/N10-1068.pdf), selected sections read, use blocked derivation proposals and disclose an omitted MH correction. Their fixed-cascade, bigram substitution example does not validate our variable-length duplicate-unit channel. We retain the correction.

[Neklyudov et al. 2020, §2 Proposition2, §3 and §3.1 Trick1](https://proceedings.mlr.press/v119/neklyudov20a/neklyudov20a.pdf), selected sections read, supplies the general joint-density/involution perspective and warns that added latent dimensions can worsen convergence. Our discrete independence kernel is elementary MH, without a continuous Jacobian. Avoiding a marginal integral can shift difficulty into mixing.

[Mengersen and Tweedie 1996](https://www.stat.rice.edu/~dcox/Stat552/Mcmc/MengersonTweedieAnnStat1996.pdf), abstract reviewed through indexed primary-paper text; scanned body did not yield usable text. The abstract links independence convergence to domination of the target by the proposal and cautions that bounds can be conservative. We do NOT claim a selected-section reading of its theorem. The finite minorization below is derived directly here and checked by rational enumeration.

Voynich-specific boundaries: [Reddy and Knight 2011, §2.2, §2.3 and §§8–9](https://aclanthology.org/W11-1511.pdf), selected sections read, describe transcription ambiguity, text partitions and the limits of vague generation/decoding explanations. [Hauer and Kondrak 2016, introduction and §§5.3–5.4](https://aclanthology.org/Q16-1006.pdf), selected sections read, work with substitution/anagram constraints and report that their manuscript outputs lack coherent syntax/semantics. No primary precedent for THIS exact neural-reading joint kernel on Voynich was established. Our synthetic supervision, known alphabet and bounded unit inventory differ from an unknown historical script. Neither cited result supplies a historical Latin assumption.

## State and proposal law

Fix observations C, a source alphabet of R rows, a U-element pool of nonempty units, an iid-uniform row prior, and one frozen stochastic neural policy. Let X be the tuple of whole source records and B the dictionary restricted to the rows visited in X. Unvisited rows are exactly unbound. The deterministic consumed-glyph scheduler makes (B,X) identify one action path; equal emitted units are still distinct source letters. No extra path multiplicity is needed. For a valid terminal state z=(B,X), define

    t(z) = Q(X) U^(-m(z)),
    q(z) = product of the ACTUALLY SAMPLED legal-action probabilities,
    w(z) = t(z)/q(z),

where m is the number of distinct visited rows and Q includes independent record resets and geometric EOS. Failed prefixes carry proposal probability but no terminal target mass. Success mass s=Σ_z q(z) can be less than one.

Independently completing the R−m unused rows uniformly gives a FULL joint state (K,X) with

    q_full(K,X) = q(z) U^(-(R−m)),
    t_full(K,X) = Q(X) U^(-R) 1[encode_K(X)=C],
    t_full/q_full = Q(X) U^(-m)/q(z).

Thus unused completion factors cancel in the importance ratio. Equivalently, integrate unused rows out and operate directly on (B,X). Both constructions retain the same marginal distribution over complete compatible readings; full-key summaries require the explicit unused-row completion law. Unused entries cannot become identified merely because a sampler fills them.

q_full(K,X) differs from q_key(K)=Σ_X q_full(K,X). A single path density, the best path, or a sum restricted to visited proposals is not that key marginal. The finite checker independently enumerates every full dictionary and compatible source tuple, including duplicate units and unused completions, rather than trusting the cancellation formula alone.

## Independence MH, including failures

Starting from one supported complete state, draw ONE fresh stochastic action path under the SAME frozen observation-conditioned policy. If it dies, retain the old state. If it completes at y, accept with

    α(x,y) = min(1, w(y)/w(x)).

For distinct successful states,

    P(x,y) = q(y) min(1, t(y)q(x)/(t(x)q(y))),
    t(x)P(x,y) = min(t(x)q(y), t(y)q(x)) = t(y)P(y,x).

Set P(x,x)=1−Σ_(y≠x) P(x,y). Failed proposals and rejections are included in this diagonal. Normalization and detailed balance therefore hold without knowing s or the observation evidence Z=Σ_z t(z). One does not need the expensive full-key likelihood L(K)=Σ_X Q(X)1[encode_K(X)=C]. Direct full-path Q scoring is sufficient; its original source/EOS implementation still needs separate admission and actual cost measurement.

Initialization is separate: a valid state can be found by a bounded literal constructive procedure, but its initial distribution is not posterior. A finite chain started there is not automatically a posterior sample. No successful proposal may be silently refilled. Retrying until success changes compute and proposal timing; if success conditioning is state-independent with this same frozen policy, q/s has a common normalizer that cancels in MH, so it is a mathematically valid DIFFERENT kernel with fewer failure self-loops. Its cost, initialization and denominators must be explicitly registered. State-dependent repair, different adaptive policies or bounded refill mechanisms generally do not enjoy that cancellation. The implementation here requires one draw and preserves failures.

Greedy decoding is different: a probability product attached to its chosen actions is not the law of the greedy output. A deterministic greedy proposal generally has insufficient support. Beams, top-k truncation, temperature changes, length penalties or posterior-guided action distributions likewise require their actual sampling densities. Arithmetic underflow can destroy theoretical softmax support; finite floating full-support assertions are not established by this rational checker.

## Support is necessary; practical coverage is the real problem

In exact arithmetic, a positive policy on every legal action and a positive source on every compatible path give q(z)>0 for all t(z)>0. Nonempty emissions bound the number of actions by total observed glyph length, so the state space is finite. This supplies mathematical reachability, not useful recovery speed.

Let π=t/Z and ε=min_z q(z)/π(z). For finite positive support ε>0 and ε≤s. Since

    P(x,y) = min(q(y), π(y)q(x)/π(x)) ≥ ε π(y)

off diagonal, and the diagonal contains at least the proposal's same-state mass q(x), the same lower bound holds there. The full kernel decomposes as εΠ+(1−ε)R, yielding the directly derived total-variation upper bound (1−ε)^n. The rational checker verifies every entry of this minorization. A vanishing ε makes the bound useless; it cannot be estimated honestly by looking only at the largest weights in a finite proposal sample.

This explains why lower teacher-forced loss or one nice greedy reading is insufficient. We need stochastic success rates, truth-path density, importance-weight concentration, actual accepted inventory changes and recovery per unit time. A neural proposal can be bad as a greedy decoder yet useful after correction, or look good greedily while missing a narrow posterior basin. These are NEW empirical questions, not permission to relabel a failed existing qualification gate.

The same accounting supplies an unbiased observation-evidence estimator: assign failed draws weight0 and successful draws w=t/q, so E_q[w]=Σ_z t(z)=Z. Its second moment is Σ_z t(z)^2/q(z). The population importance-efficiency fraction Z²/E[w²] equals 1/(1+χ²(π||q)), using a zero-target failure atom, and is at most s by Cauchy–Schwarz. The rational checker verifies these moments and the success-mass bound. This is NOT a finite-sample confidence interval: a handful of small observed weights cannot exclude an unseen very large weight or certify good evidence estimates. The source likelihood remains a model probability, not a historical-language proof.

There is a further distribution mismatch to measure before trusting neural coverage. The current trainer draws empirical-corpus windows of64–224 source letters per record, whereas the original inference source uses its own contextual law and geometric EOS. For fixed observed records, compatible source lengths can range outside that training interval; two448-glyph records can require896 total actions. The existing training objective therefore fits the conditional training-generator distribution, not automatically the original source-model posterior used in correction. This can be a large variance/mixing penalty even when supervised loss falls. Full support in a mathematical softmax does not remove the mismatch. Future source-matched simulations or explicit defensive mixtures need separately registered length, source, proposal-law and resource checks; the ongoing training inputs/gates are unchanged.

## Tempering, source priors and circuits

This helper supports ONLY terminal β=1 and iid-uniform row prior. The existing key-tempering target π_key(K)L(K)^β augmented with its conditional reading has density π_key(K)L(K)^(β−1)Q(X). At intermediate β, a key-likelihood factor remains. Tempering a point reading instead would yield a different key marginal Σ_X Q(X)^β, with path-count bias at β0. A nonuniform row prior also requires its correct visited-entry factors and unused completion distribution. Both unsupported uses are refused rather than guessed.

Once proposals show measurable behavior, causal interpretation can ask WHERE true trajectories lose probability: before first binding, at reused binding, at segmentation or at long-context continuation. Intervene on the explicit row-unit memory with matched legal-state controls and compare BOTH independently trained seeds. Legal masks must remain fixed for a neural-input intervention; changing bindings can instead change the symbolic problem itself. Measure whole-path probability, failures and accepted moves, with source-only/binding-off controls. These are proposed analyses, not observed circuits, and they require another fixed protocol after the unchanged training comparison closes.

## Bounded qualification and next admission

[JOINT-READING-MH-THEORY-001](../experiments/JOINT-READING-MH-THEORY-001.md) fixes exhaustive rational checks on36 binary two-record panels, two source laws and two positive action policies. Check all successful/failed probability mass, literal unique paths, full-key joint/collapsed target and proposal laws, unused-row cancellation, every balance equation, stationarity and minorization. Explicit negative witnesses omit the once-per-row prior and misuse key marginals; helpers refuse dead ends, nonfinite probabilities and unsupported priors/temperatures. Full inventory changes are admitted by the proposal support, unlike the old label swap. This is finite arithmetic qualification, with no Latin archives, trained policy, original source probabilities, recovery panel or manuscript access.

After a finite PASS, a separately frozen admission should measure sampled-policy density replay, unchanged-source direct path probabilities including EOS, full-size proposal cost and success/failure/weight accounting before ANY inference campaign. Known-answer recovery/null controls and equal-time comparison with existing search remain necessary. There is no unbounded paid loop, no guaranteed speedup, and no historical decipherment claim.
