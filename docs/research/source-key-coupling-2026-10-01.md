# Diagnose the objective's interactions before learning a repair policy

2026-10-01, before SOURCE-COUPLING-001. Gold-assisted model intervention,
not a decipherment method or newly recovered neural circuit.

SOURCE-REVISE-001 visited131072keys but reduced errors only5.12%; every
generating key still scored above the selected key. Supplying the key to the
reader nearly recovered the artificial text. Those observations locate a
search deficit, but do not tell us whether focused single-row improvements,
coordinated changes, a source-context effect or support discontinuities matter.

## Review and transfer limitations

The direct Voynich/decipherment review remains `PRIOR_WORK.md`. Revisited
[Hauer & Kondrak2016](https://aclanthology.org/Q16-1006.pdf), especially§5:
their language/anagram/substitution constraints and synthetic results do not
qualify our variable-length duplicated units or establish Hebrew for Voynich.

[Berg-Kirkpatrick & Klein2013](https://aclanthology.org/D13-1087.pdf),§§2–3,
contrasts found and gold-initialized likelihoods in HMM/EM decipherment.
Their results motivate separating a missed solution from inadequate model
scoring and checking initialization. Their trigram HMM, learned homophonic
emissions and restart counts do not predict our hard-key search cost. We do
not launch a million restarts, import their same-author language assistance or
repeat their then-current Zodiac340 status as current history.

[Sundararajan, Dhamdhere & Agarwal2020](https://proceedings.mlr.press/v119/sundararajan20a/sundararajan20a.pdf),
§1.3 equations1–3, defines anchored discrete derivatives of set functions.
We use this elementary inclusion-exclusion operation on legal key replacements.
We do NOT compute their Shapley-Taylor index, which additionally aggregates
over contexts with specified weights and attribution axioms. These finite
differences depend on our selected key and target units; they are not unique
neuronal functions, historical causes or average population interactions.

One attempted auxiliary NeurIPS-page retrieval returned an internal error;
no design choice, claim or citation relies on it. Primary texts above and
the existing exact-search/probability notes supply the actual rationale.

## Fixed interventions on the original objective

For the existing selected key K, retain J(K)=-23log42+sum_records logL(K,C)
and its exact all-source-path scorer unchanged. Enumerate EVERY one-row
replacement:23rows×41other legal units=943alternatives, not random proposals.
This tells us whether a selected bank maximum is actually a single-row local
optimum within a declared0.1nat tolerance. It is not a claim about pair/whole
key optimality, and scores still embody supplied source/channel assumptions.

Let M be the gold-used rows that K gets wrong. For a subset S⊆M, let K_S
replace those rows with their generating units. Evaluate every S of sizes1,
2,3. Also evaluate the full M endpoint and generating FULL key. Rows outside
S are fixed, including unused rows. This is a deliberately answer-assisted
counterfactual diagnosis, not an algorithm permitted gold at manuscript time.

Choose one reproducible wrong target per row, with the SAME length as its
gold target, excluding both incumbent and gold units. Use the same S sets
with these targets. This matches changed rows and emission-length changes,
but not all statistical properties or semantic quality; one draw per row
does not establish a null distribution or significance. Preserve all outcomes.

F(S)=J(K_S) defines a binary replacement face. Where every subface has
positive support, compute:

    d_ij = F(ij)-F(i)-F(j)+F(empty)
    d_ijk = F(ijk)-F(ij)-F(ik)-F(jk)+F(i)+F(j)+F(k)-F(empty).

These log-score derivatives quantify nonadditivity around this anchor. They
need not sum to the full M endpoint improvement: higher orders and other
backgrounds are unmeasured. If any subface has zero support, the derivative
is UNDEFINED, not zero or an arbitrarily clipped large number.

Separately report a joint-only improving face when its gain exceeds1nat but
every proper nonempty subface improves by at most0.1nat or has zero support.
A support bridge has positive full-face likelihood with ALL proper nonempty
subfaces unsupported. These certify statements ONLY within this binary face.
Alternative target units or paths outside it may connect the same keys. No
global energy barrier, disconnected23-row state graph or minimal required
block size follows. Temperature can cross finite valleys; it cannot restore
a literally zero-probability intermediate state under the unchanged target.

## Source-context intervention

Construct a separate validated order0source from the ORIGINAL root counts,
same0.5smoothing, same23letters, rho and key prior. Its sole probability row
must equal the original root row exactly, transitions stay0 and arrays are
immutable. The original source arrays/count archive are never edited. Score
EVERY identical candidate under this IID model as well as the context model.

All original and IID source-letter probabilities are strictly positive, so
a fixed dictionary has positive observation likelihood exactly when at least
one literal source path emits the ciphertext. Consequently their support
masks must coincide. This is a mathematical claim about these models; enforce
it at runtime. An all-zero subface is a structural emission-compatibility
issue, not evidence of language meaning. Finite interactions can persist
under IID because segmentation and mapping choices remain coupled.

Compare signs/magnitudes and joint witnesses under both models, including
cases where context creates or removes a joint-only gain. This intervention
changes the scoring objective in a CONTROL; it does not silently substitute
IID for the final contextual target or demonstrate that context is unnecessary.

## Implications for a later solver

Large missed single-row gains would justify systematic or learned local repair
before stronger global machinery. Gold-specific joint gains with weak single
gains would justify candidate-conditioned coupled proposals, whose test-time
inputs must be ciphertext/current key/source constraints rather than gold.
Comparable decoy gains or IID effects would narrow that interpretation.

One can imagine learning a repair policy or an explicit factor-graph module
from these diagnostics, with a fixed verifier evaluating its moves. None is
trained here. A useful objective intervention is not mechanistic interpretation
of the still-unsuccessful neural inverse, and its geometry does not identify
Voynich's linguistic structure.
