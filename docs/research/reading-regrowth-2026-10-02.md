# Revise a shared dictionary by cutting and regrowing a joint reading

Own derivation and implementation; finite and empirical admissions are separately registered. This is a search redesign following actual failures, not a historical reading or a recovered circuit. The current 96M paired training campaign and its original 73 dependencies stay unchanged.

## Evidence motivating the change

The [closed recovery diagnosis](../experiments/TEMPERED-RECOVERY-DIAG-001-results.md) found the generating used-unit inventory absent from every final bank on its four exposed known-answer cases. Row-label swaps preserve that inventory. The [cipher/history diagnostic](../experiments/SOURCE-ACTION-CIPHER-001-results.md) then found every original-source greedy path used only single-glyph actions, while trained guided decisions scarcely reacted to a major permutation of the observation encoder's input. All seven tested policies still recovered zero exact records. These are distinct observed failures; neither demonstrates that a larger model cannot help.

The next question is whether inference can release a bad segmentation/binding and reconstruct all its later consequences. Reusing a complete old dictionary when proposing a supposedly new prefix would leak the old future and prevent meaningful inventory changes. Here only bindings actually introduced before the cut survive. Both observed records remain coupled through a shared dictionary, and the resulting whole reading is scored before acceptance.

This concerns the same fixed, non-erasing, duplicate-permitting one/two-glyph channel. Applying it to manuscript data would require a separately falsified channel family, transcription uncertainty, physical-folio holdouts and external predictions. EXPOSED Pliny is development material. No reserved author or manuscript is opened.

## Primary method review

[Wingate, Stuhlmüller and Goodman, author Revision3 updated February8,2014, §§2–2.1 and Algorithms2–3](https://web.stanford.edu/~ngoodman/papers/lightweight-mcmc-aistats2011.pdf) describe trace proposals whose later random choices and parameters can change after an earlier choice changes. The revised Algorithm2 explicitly accounts for changed trace size and fresh/stale choices. We use this as a warning to retain selection and suffix-density corrections; our suffix-block derivation below is our own. The [2011 proceedings copy](https://proceedings.mlr.press/v15/wingate11a/wingate11a.pdf) and revised author copy differ in Algorithm2's displayed correction, so the version matters. This is not a claim that their probabilistic-language implementation solves a shared cipher dictionary.

[Lindsten, Jordan and Schön2014, §§3.1–3.3 and §6.2, especially Eq22–23](https://jmlr.org/papers/volume15/lindsten14a/lindsten14a.pdf) develop particle Gibbs with ancestor sampling. For non-Markovian histories, future-weight evaluation can make the kernel quadratic in trajectory length; their truncation discussion assumes decay of past influence. A persistent cipher dictionary can couple distant occurrences exactly, so that assumption is not supplied here. Arbitrarily splicing an old suffix after a new binding can be literally invalid. We do not implement PGAS or claim its empirical speedups; explicit regrowth makes the whole suffix compatible and its proposal law observable.

[Reddy and Knight2011, §2.2 and §§8–9](https://aclanthology.org/W11-1511.pdf) document transcription ambiguity and limits of broad generation/decoding accounts. This constrains what a synthetic success could imply. The earlier [joint-reading review](joint-reading-independence-2026-10-02.md) separately reviews blocked decipherment proposals, joint MH and Voynich substitution/anagram precedents. None establishes a historical Latin assumption or a primary precedent for this exact kernel on Voynich.

## State, target and prefix invariance

A complete state consists of the visited dictionary and the source texts for every whole observed record. The existing deterministic scheduler selects the least-consumed *fraction* of a record, resolving ties by record order. Raw consumed offsets give a different path when lengths differ; that draft error was caught and corrected before any scientific invocation. Emissions are nonempty, so action length n is finite and bounded by total observed glyph length.

Let U be the unit-pool size and m(z) the number of visited source rows. Retain the existing target:

    t(z) = Q*(X(z)) U^(-m(z)).

Q* is the product of the unchanged dense binary64 source coefficients, rational geometric continuation224/225 per source letter and stop1/225 once per record, resetting source context at each record. We interpret each stored binary64 coefficient as its exact rational value. Dense rows pass their existing floating normalization tolerance, but their exact binary sums need not equal1. We therefore call this a fixed computational positive potential, not a certificate that its infinite-string law is exactly normalized. For the finite observation-compatible state space, normalizing t defines a proper target. We never secretly renormalize rows or change the source model.

The uniform unvisited-row prior cancels when those rows are integrated out, as already independently checked in [the earlier finite qualification](../experiments/JOINT-READING-MH-THEORY-001-results.md). New finite checks independently enumerate all full dictionaries again, including duplicate units and unused completions. A sampled action-path probability remains different from a whole-key marginal.

Fix one past-only proposal policy. At each legal next action, its unnormalized weight is the source coefficient, times1/U for a first binding, times a fixed bias6 for a newly introduced two-glyph unit, and times stop probability if this action closes its record. Common continuation factors cancel across current actions. The bias is an engineering heuristic motivated by six possible two-glyph units starting with a given first glyph in the six-glyph pool. It is not an exact future-likelihood factor or a new target prior. The root-only and regrowth comparison share precisely the same policy.

Actual integer sampling matters. Convert positive binary coefficients to aligned integer weights; retain rational bias/stop/prior factors as integers. Give each legal choice one count, then distribute the remaining 2^32-minus-number-of-choices counts by exact largest remainders, with action order breaking ties. Draw an integer from the grid using PCG64. Thus actual action probability is count/2^32, every legal choice has positive probability, and tiny weights do not disappear through floating cumulative sums. The reference path log probability is computed from those actual counts. Quantization changes the proposal; the correction still targets unchanged t.

## Correcting a variable-length suffix move

For a current complete action path x of length n_x, choose a pre-action cut k in0,…,n_x−1:

    c_x(k) = eta * 1[k=0] + (1-eta)/n_x.

The regrowth arm fixes eta=1/8; the independent-rebuild control fixes eta=1, always cutting at the root. Uniform cuts eta=0 are additionally checked in finite qualification. Reconstruct the literal prefix from an empty dictionary, discard later bindings, and draw one suffix under the same past-only policy. A dead suffix leaves the current state unchanged, without refill. A complete candidate y shares precisely that prefix and can have a different number of actions, source letters and used units.

For this auxiliary cut, accept with

    min(1, t(y) c_y(k) q(x_tail | prefix) /
              [t(x) c_x(k) q(y_tail | prefix)]).

The preserved prefix is identical, so its policy probability cancels. It does not cancel if an adaptive policy or old future bindings alter the suffix law. Selection probabilities are marginal cut probabilities, not the chosen mixture-component probability. Under uniform cuts their ratio is n_x/n_y; with root bias it is the displayed c_y/c_x. Omitting it changes the invariant law.

For fixed k, the accepted unnormalized flux equals

    min(t(x)c_x(k)q(y_tail|prefix), t(y)c_y(k)q(x_tail|prefix)),

which is symmetric. Summing over every common cut retains detailed balance, including when several cuts can produce the same candidate. Rejections, same-state proposals and failures sit on the diagonal. Root proposals have positive mathematical support over every compatible complete path; this supplies finite reachability, not a practical mixing bound.

The implementation evaluates target and acceptance ratios as positive integers, including exact grid exponents. It compares a rational acceptance probability to lazy uniform64-bit blocks. A ratio smaller than2^(-64) can still be accepted; a floating log comparison with a finite-resolution uniform would need separate rounding analysis. Equality to a block threshold triggers another block; a fixed16-block cap aborts rather than substitutes an approximate decision. PCG64 is required because not every NumPy bit generator returns full64-bit random blocks. Stationarity statements refer to the ideal uniform primitive law; a fixed pseudorandom seed produces a deterministic trace, not a convergence certificate.

## Qualification, empirical admission and limits

[READING-REGROWTH-THEORY-001](../experiments/READING-REGROWTH-THEORY-001.md) exhaustively compares the actual implementation with independent literal action enumeration, rational quantization, full-key collapse, every acceptance ratio, transition normalization, detailed balance and stationarity. Negative kernels omit cut selection or visited-row prior; witnesses must exist. Unequal record lengths and failed tails are included. Unit fixtures also test extremely tiny acceptance, bit-cap aborts, invalid RNGs, law-instance mismatch, future-inventory release and JSON trace replay. Preparation failures are retained in the notebook.

Only after a published finite PASS, [READING-REGROWTH-ADMIT-001](../experiments/READING-REGROWTH-ADMIT-001.md) fixes all97 existing exposed positive/null cases, both policies, eight attempts each, one paired seed per case, original dense source, and known-answer density checks. Initial states are the already archived source-increment greedy readings, explicitly converted from their old schema without reusing their greedy reference density. Save every candidate and rejection, reference proposal counts, acceptance bit-work, retained target and final RNG state. One deterministic audit replays exactly those eight moves and independently checks literal encoding and dense source potential; no chain is extended.

This is a density/cost admission, not a recovery or posterior calibration gate. Eight steps cannot certify mixing. Changing inventory, source-score gains or completion on nulls cannot establish semantics. An empirical PASS permits estimating a separately registered longer known-answer campaign; failure does not authorize retrying this namespace. There is no adaptive budget, paid loop, optimizer/GPU action, or modification of live training. Mechanistic interpretation can later connect proposal failures to first-binding, segmentation or history representations only with causal interventions and qualified behavior.
