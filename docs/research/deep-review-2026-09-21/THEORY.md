# What can the text identify? Theory and inference design

Reviewed 2026-09-21. Research/design only. The [18 source records](theory.sources.json) distinguish 16 selected-section reads from two abstract-only discoveries. DreamCoder is also D22 in the decipherment review and is counted once in the unique-work total. No proofs were independently verified in full and no method here has been replicated in this project.

## The inverse problem comes before the architecture

We observe a finite string. We do not observe its intended language, plaintext, encoding procedure, omissions or authorial state. A predictor estimates what tends to follow a prefix. A decoder must additionally explain why particular observations correspond to particular meanings.

Our target should therefore be staged:

| Object | What would count as evidence | What it does not establish |
| --- | --- | --- |
| Predictive model | Better probabilities on genuinely unseen data | A historical mechanism |
| Predictive state | Compact joint predictions with consistent updates | Plaintext meanings |
| Model mechanism | Selective interventions explain the trained model's behavior | The author used that mechanism |
| Historical decoder | Constrained rules, accounted exceptions, independent readings | Certainty from one plausible passage |

This is our problem formulation. It is not a claim that these stages are universally necessary or sufficient.

## Identifiability is conditional, not all-or-nothing

Identifiability asks whether different hidden explanations can produce exactly the same observable distribution. If they can, unlimited samples from that distribution cannot choose between them without extra assumptions. Finite data introduces additional uncertainty even when a model is identifiable in principle.

Locatello and colleagues give a precise nonidentifiability construction for unsupervised disentanglement under particular continuous latent-prior assumptions. It warns against assigning unique meaning to learned coordinates; it does **not** prove that structured ciphertext recovery is impossible. [T01](https://proceedings.mlr.press/v97/locatello19a/locatello19a.pdf)

Khemakhem and colleagues show how auxiliary variables can restore specified forms of identifiability, requiring injective mixing, an appropriate conditional latent family and sufficiently rich full-rank modulation. Manuscript section or scribe labels might be useful auxiliary information, but they are neither interventions nor evidence that those mathematical conditions hold. [T02](https://proceedings.mlr.press/v108/khemakhem20a/khemakhem20a.pdf)

For ordinary finite-state HMMs, Gassiat and colleagues provide identification up to state relabeling with known state count, full-rank transitions, independent emission measures and positive stationary support. Their result uses the **distribution** of observation triples, not three observed symbols. Our edge-emitting synthetic processes need their own assumptions or a valid conversion, potentially with more states. [T04](https://alice.cleynen.fr/wp-content/uploads/2015/03/GCR_2015.pdf)

Project consequence: every recovery benchmark needs an observability audit. Include cases with identifiable dynamics, cases with unresolved labels, and distinct hidden descriptions that are observationally equivalent. An honest inference system should preserve the ambiguity in the last case.

## Joint futures and recursive updates

Our own elementary example illustrates a blind spot in separate prediction heads:

| Process | Possible next two symbols | First-position probability of 1 | Second-position probability of 1 |
| --- | --- | --- | --- |
| Same pair | 00 or 11, equally likely | 0.5 | 0.5 |
| Opposite pair | 01 or 10, equally likely | 0.5 | 0.5 |

The individual predictions match exactly; the pair distributions are disjoint. This is a mathematical illustration, **not a new experiment**. A normal autoregressive predictor can distinguish them by updating after the first symbol. The problem is evaluating or clustering only separate future marginals, which discards that dependence.

Computational mechanics defines states through equivalence of entire future distributions. Such states are predictive summaries, not necessarily the physical machine's hidden variables. CSSR makes the additional requirement operational: a state must update consistently after a new symbol. Its finite-state, stationarity and synchronization conditions matter; it is not merely another clustering routine. [T05](https://arxiv.org/pdf/cond-mat/9907176v2), [T06](https://arxiv.org/pdf/cs/0406011v1)

Predictive state representations similarly encode state through probabilities of multi-step tests. Their finite construction from known POMDP structure is not a ready-made learner for unknown manuscript text. We can borrow the test representation while evaluating the learning step independently. [T07](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf)

Project proposal: score normalized joint continuation distributions and update closure. Keep one-step prediction as a baseline. Report horizon-specific errors, so a large first-symbol gain cannot conceal failure on later symbols.

## Explicit probabilistic models are serious competitors

Spectral HMM learning estimates observable operators from moments under rank and conditioning assumptions. It supplies an alternative to neural optimization, but poorly conditioned moments and invalid finite-sample probabilities need diagnostics. [T03](https://www.learningtheory.org/colt2009/papers/011.pdf)

Bayesian structural inference integrates parameters within restricted unifilar graph classes. The unique path property enables the calculation; arbitrary graph enumeration is not free. Its posterior ranks the supplied candidates, including when all candidates are wrong. [T10](https://arxiv.org/pdf/1309.1392)

Our proposed edge-emitting comparator has one nonnegative matrix per observed symbol, with entries
`M_a[i,j] = P(next state=j, symbol=a | current state=i)`.
Rows sum to one across both symbols and next states. Given a row belief vector b:

`P(a | history) = b M_a 1`

`b_after_a = b M_a / (b M_a 1)`

`P(a1...ah | history) = b M_a1 ... M_ah 1`

These equations specify a model family; they do not assert that Voynich follows it. Smoothing and zero-probability handling must be explicit. State count is selected on development data with a fixed complexity penalty, never supplied by hidden labels. Compare multiple restarts and a neural reference; fit failures and model-class failures are different.

## Teacher access must not be confused with evidence

Angluin's regular-language result assumes membership and equivalence queries. Conditional-sample HMM work also distinguishes exact probability queries from sampled access and introduces additional conditions. A saved historical manuscript supplies neither oracle. [T17, abstract only](https://www.sciencedirect.com/science/article/pii/0890540187900526), [T08](https://proceedings.mlr.press/v195/mahajan23a/mahajan23a.pdf)

We can query our trained predictor or search for counterexamples to an extracted rule. That tests agreement with a fallible teacher. The teacher's own discrepancy from synthetic ground truth must be measured before extraction quality is interpreted. On Voynich, teacher fidelity to the historical source remains unknown.

## Learn an inference procedure across tasks

Prior-data fitted networks train on complete tasks sampled from a chosen prior. Their objective targets posterior prediction under that prior. Our proposed adaptation is to sample a fresh generator and emission key per sequence episode, observe a prefix, then predict the suffix. This visible-only sequence version is our extension, not a replication of the original supervised task setup. [T12](https://arxiv.org/pdf/2112.10510v7)

The ambitious possibility is to learn reusable inference about hidden processes instead of memorizing a few keys. The risk is equally concrete: a narrow simulator prior can make the learner confidently wrong on another family. Vary noise, state count, key structure and mechanism; hold out entire mechanisms as well as parameter settings.

DreamCoder suggests bounded program proposals and reusable libraries. It starts with a supplied grammar and task specification; many demonstrations use input/output examples. A typed, auditable decoder grammar is a reasonable future comparison, but unconstrained program search over guessed translations would lack an objective truth signal. [T13 / D22, same work](https://arxiv.org/pdf/2006.08381)

## Model the observation channel and pay for flexibility

Unknown missing observations can confound inferred transitions; explicit omission models help under the right specifications. That result concerns deletions, not an unrestricted license to call arbitrary glyphs filler. [T09](https://proceedings.mlr.press/v202/perets23a/perets23a.pdf)

A candidate decoder must account for the cost of its key, segmentation, omission/null choices and exceptions. MDL provides coding-based ways to compare such flexibility; the coding convention itself must be frozen and disclosed. Shorter descriptions do not uniquely establish historical truth. [T11](https://arxiv.org/pdf/1908.08484)

Project proposal: separate clean encoding, observation noise, and transcription uncertainty. Start with no deletions; introduce one bounded channel mechanism at a time and require gains over a matched channel-free model. Preserve the undecoded observations and alignment.

## Calibrate uncertainty and choose informative tests

Simulation-based inference is inference under a simulator. Simulation-based calibration checks whether the inference procedure behaves consistently with that chosen model; it cannot establish that the simulator describes the manuscript. Assess predictive mismatch, uncertainty and informativeness separately, including intentionally wrong simulator families. [T15](https://arxiv.org/pdf/1911.01429), [T16](https://arxiv.org/html/1804.06788v2)

Experimental design offers a useful decision rule: prefer observations or diagnostics expected to distinguish currently plausible hypotheses. We cannot manipulate the medieval author. We can select independent manuscript annotations, controlled synthetic counterexamples and analysis measurements. Enumerating a small candidate set is enough initially; amortized neural design is deferred. [T14](https://proceedings.mlr.press/v139/foster21a/foster21a.pdf)

Repeatedly adapting to final results consumes their evidential independence. Formal reusable-holdout methods require mechanisms we have not implemented. Generate fresh synthetic final keys/families and leave the manuscript final test untouched during this design phase. [T18, abstract only](https://research.ibm.com/publications/the-reusable-holdout-preserving-validity-in-adaptive-data-analysis)

## Decisions

Adopt the **evaluation targets** now: observability, explicit update rules, joint futures, fresh-task generalization, uncertainty and complexity accounting. Compare a small explicit probabilistic model with a compact neural inference model. Defer large adaptive-memory stacks and broad program synthesis until these comparisons work.

The mathematical literature gives both reasons for optimism and precise boundaries. Some restricted hidden processes are recoverable. Neither their theorems nor a sophisticated neural representation supplies missing semantic anchors automatically. The next [design and experiments](NEXT_DESIGN.md) turn these distinctions into falsifiable implementation requirements.
