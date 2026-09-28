# Escaping unknown-unit search barriers

2026-09-28. **Results-blind theory review; proposals, not experiments.** Read
[the inference review](blind-channel-next-inference-review.md) and
[recovery design](blind-channel-recovery-design.md). No empirical pilot,
corpus, fitted-channel, or answer files were opened. This changes
no registered solver or experiment. The deductions below concern the one-state
deterministic family, with duplicate units permitted and fixed source/stop law.

**Conditional recommendation:** after a demonstrated search gap at a completed
swap/replacement local optimum, test coordinated two-row replacements. They can
cross a support barrier that no amount of completing the existing neighborhood
removes. Dense relaxation and partial-key search remain credible alternatives,
but introduce separate projection and bounding problems.

## What the primary sources actually supply

These exact primary PDFs were opened; the cited sections, rather than search
snippets, support these summaries.

| Primary source and sections read | Relevant result and boundary |
| --- | --- |
| [Chiang et al. 2010, *Bayesian Inference for Finite-State Transducers*](https://aclanthology.org/N10-1068.pdf), §§2–3 | Forward-backward estimates normalized competing transitions; blocked path proposals can change whole derivations. The initial cascade supplies the parameter inventory. Their approximate sampler omits a discussed MH correction. This supports an auxiliary finite-inventory workspace, not unrestricted structural discovery or an exact sampler claim. |
| [Berg-Kirkpatrick & Klein 2013, *Decipherment with a Million Random Restarts*](https://aclanthology.org/D13-1087.pdf), §§2–3.1 | HMM decipherment is sensitive to initialization and restart budget. Cipher symbols are supplied units; the model uses a fixed trigram source and tuned count smoothing. Its empirical restart counts do not predict unknown-unit performance or justify unbounded compute here. |
| [Nuhn, Schamper & Ney 2013, *Beam Search for Solving Substitution Ciphers*](https://aclanthology.org/P13-1154.pdf), §§3–4, abstract | Partial substitution hypotheses, extension order, score estimates, and pruning separate search design from source quality. Reported stronger-source gains over weaker-source exact optimization illustrate objective/search separation. Their supplied cipher tokens are not unknown variable emissions; beam pruning does not certify optimality. |
| [Corlett & Penn 2010, *An Exact A* Method for Deciphering Letter-Substitution Ciphers*](https://aclanthology.org/P10-1106.pdf), §§2–3 | Relaxing consistency at unassigned symbol occurrences gives an admissible bound for their fixed-unit permutation problem. The bound below transfers that principle with a different sum/max recurrence; it is our deduction, not their published variable-unit algorithm. |
| [Luo et al. 2021, *Deciphering Undersegmented Ancient Scripts Using Phonetic Prior*](https://aclanthology.org/2021.tacl-1.5.pdf), §§1–3.2 | Joint word segmentation/cognate alignment is possible under explicit linguistic constraints. Their inputs include a known-language vocabulary and IPA information; matched cognate spans differ from discovering a reusable arbitrary character-to-codeword dictionary. Unknown word boundaries are not evidence that our unknown emission units are solved. |

## Separate a search gap from a wrong target

Fix `J(u)=C(u)-log2 p(Y|u)` before comparison. A legal reference channel with
smaller `J` than the returned channel proves that this run missed a better
candidate. It does not prove that reference is globally best. Conversely, a
wrong decoder beating the generating channel reveals an objective preference;
more effective optimization may worsen recovery. Inspect model and data bits
separately and compare decoding behavior, including equivalent descriptions.

Known-channel decoding does not establish unknown-key rankings. Teacher-matched
sources remove one misspecification mechanism, not finite-sample ambiguity.
Evaluator-only reference inspection diagnoses a run; any subsequent redesign
needs fresh cases. A completed neighborhood certificate concerns only its
declared moves and tolerance.

## A strict barrier, with positive rational probabilities

**Deduction; hand-specified counterexample, no fitted data.** Let source alphabet
be `(a,b)`, glyph alphabet `(x,y)`, maximum unit length two, and complete pool
`(x,y,xx,xy,yx,yy)`. Set `rho=1/2` and:

| Context | `P(a)` | `P(b)` |
| --- | ---: | ---: |
| start | `99/100` | `1/100` |
| a | `1/100` | `99/100` |
| b | `99/100` | `1/100` |

Observe two independent records, both `xyxy`. For incumbent `u=(x,y)`, each
record has the unique plaintext `abab` and probability
`p0=(1/2)*(1/2)^4*(99/100)^4`.

Every nonidentity pair-swap/single-row replacement is worse or impossible:

- Swap to `(y,x)`: the probability ratio to the incumbent is `1/99`, with
  unchanged model bits.
- Replace `a` by `xy`: only plaintext `aa` works; the ratio is
  `4*(1/100)/(99/100)^3 < 1`, and the model costs one extra bit.
- Replace `b` by `xy`: only `bb` works; the ratio is
  `4*(1/100)^2/(99/100)^4 < 1`, again with one extra bit.
- All other eight single-row replacements make `xyxy` impossible.

The coordinated replacement `u'=(xy,xy)` supports every two-letter source
string, whose probabilities sum to one. Thus each record has probability
`p1=(1/2)*(1/2)^2`, giving ratio `p1/p0=4/(99/100)^4`.

Use the existing literal code with one state, one alternative per row,
`max_emission_length=2`, binary glyph indices, one source, and any positive
weight-grid denominator. With `max_states=max_alternatives=source_count=1`,
the fixed-width encoder gives **four model bits** for `(x,y)` and **six** for
`(xy,xy)`: two length bits plus respectively two/four glyph bits; singleton
state/weight fields cost zero. Larger shared bounds add the same constants.
Therefore

```text
J(u') - J(u) = 2 - 2*log2[4/(99/100)^4]
             = -2 + 8*log2(99/100) < -2 bits.
```

This is a strict local optimum with an improving two-row replacement.
**Duplicate units matter:** the better channel deliberately maps both letters
to `xy`, which our family permits. It loses plaintext information. This proves
a search barrier and simultaneously warns that an MDL improvement is not
decipherment evidence. It does not establish the same example for injective keys.

## Three alternatives and their failure modes

**Dense soft support, then joint projection.** A positive distribution over all
units allows probability to move before hard deletions destroy parses. EM can
optimize likelihood within that inventory; independent row projection need not
preserve it. Dense rows exceed the current sparse/grid grammar and are only a
proposal workspace. Charge their compute; rescore actual projected legal models
with literal code bits. A fitted relaxed likelihood is neither the final score
nor an upper bound on the best hard key: local EM has not maximized the relaxation.

**Strict-argmax counterexample.** Use order-zero `P(a)=P(b)=1/2`, `rho=1/2`,
units `{x,y}`, and records `(x,x,x,y,y)`. A relaxed optimum has both rows
`q(x)=3/5,q(y)=2/5`, yielding record probabilities `3/20` and `1/10`.
Each row strictly prefers `x`; there is no argmax tie. Independent projection
produces `(x,x)` and zero probability for both `y` records. Yet hard key `(x,y)`
gives probability `1/8` to every record. Thus even a relaxed global likelihood
optimum can project catastrophically. A bounded joint projection must retain
alternative assignments and check complete-record support; entropy sharpening
alone does not fix this example.

**Partial keys with a valid bound.** For a partial key `K`, let `S_a` contain its
assigned unit, or the complete pool if row `a` is unassigned. Define `h+a` as the
updated source context. For record `y` of length `n`, set `B_n(h)=rho` and:

```text
B_i(h) = (1-rho) * sum_a P(a|h)
         * max_{v in S_a} [1{y starts with v at i} * B_(i+|v|)(h+a)].
```

Invalid/overrunning matches contribute zero. Backward induction proves
`p(y|u) <= B_0(start)` for every completion `u`: each fixed emission is among the
maximized choices, and all coefficients are nonnegative. At a complete key,
this becomes the exact marginal, retaining the source-letter **sum** rather than
substituting Viterbi. Consequently
`C_min(K)-sum_y log2 B_0(start)` is a lower bound on attainable `J`, where
`C_min` charges assigned units and shortest legal units for unresolved rows.

The relaxation allows different emissions for the same unresolved letter at
different occurrences and records. With one source letter, pool `{x,y}`, and
record `xy`, its bound is positive although every hard key is impossible. With
pool `{x,xx}` and records `(x,x^(2m))`, only key `x` supports both; the bound can
use `xx` on the long record, overestimating the best joint likelihood by
`(1-rho)^(-m)`. Thus it can be exponentially loose. A capped beam offers proposals,
not a certificate; exhaustive branch-and-bound needs safe numerical bounds and
an unpruned frontier. These examples should falsify an assumed tightness claim
before implementation.

**Latent parse/key alternation.** Sampling whole posterior parses can coordinate
many boundaries; rebuilding keys from several parses can suggest macro edits.
One best parse can freeze the wrong segmentation. A deterministic incumbent's
posterior cannot contain absent units, so alternation without expanded support
cannot invent them. Complete replacements require exact marginal rescoring.
Neither hard alternation nor approximate blocked sampling inherits ordinary EM
monotonicity or an exact posterior guarantee.

## One bounded conditional next experiment

If the separately frozen systematic search completes a local certificate,
known-channel decoding is adequate, and the evaluator identifies a legal
better-scoring reference, register a **paired two-row replacement audit** on
four fresh teacher-consistent variable-unit keys. Do not alter DEV-003. If the
wrong recovered channel already beats the reference, diagnose the objective
before spending this budget on unchanged-score optimization.

Both arms start from the same ciphertext-only local incumbent. The macro arm
chooses four distinct row pairs using a frozen RNG, evaluates every double
replacement for those pairs, accepts only exact `J` improvements, and applies
the unchanged local polish. Allow two such rounds. With 23 source letters and
42 units this is at most `2*4*41^2=13,448` macro candidates, before polishing.
The comparison arm spends its budget on further seeded starts plus the same
local search. Include the analytic barrier above as a correctness fixture,
not an empirical recovery success.

Freeze eight fit and four transfer records per key under the disclosed proper
source/stop law, without outcome-based replacement of cases. Charge all
initialization, scoring, polishing, and validation. Planning caps: 120 CPU
seconds and 20,000 exact candidate scores per arm/key, one worker, 2 GiB memory,
20 CPU-minutes total including validation, no paid compute. Record timeouts and
retain completed candidates; these are limits, not measured throughput.

Report objective improvement, completed versus attempted blocks, support
failures, cost, and evaluator-only transfer edit error separately. An accepted
macro proves escape from the tested local neighborhood. Advancement toward
recovery requires improved fresh-text decoding across keys, not merely a
shorter code. If random row pairs miss useful interactions, preserve that
negative result before considering soft or posterior-guided pair selection.
