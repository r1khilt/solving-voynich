# Deferred research candidates

**Status: active research.** Source review, corpus preparation, 18-run architecture comparison, two rounds of context tests and supervised synthetic causal calibration are complete. The table preserves the original candidate inventory; current priorities follow it.

| Candidate | Concrete output | Prerequisite |
| --- | --- | --- |
| Ground the project in primary sources | A cited source map covering manuscript/transcription access, well-supported observations, major competing explanations, and prior reproducible work | User starts research |
| Establish data provenance | A corpus manifest with source/version, rights, checksums, uncertainty, folio/line/layout preservation, and comparison of transcription choices | Source review and permitted acquisition |
| Design leakage-resistant evaluation | Frozen split policy, controls, and diagnostic definitions | Understand corpus structure and duplication |
| Inventory available resources | Verified API/compute access, cost estimates, spending bounds, and secure credential setup if needed | Concrete experiment requirements; user assistance as needed |
| Build a minimal synthetic benchmark | Reproducible signal/filler, homophony, stateful, and nonsemantic generators with known latent truth and explicit identifiability assumptions | Chosen question and evaluation protocol |
| Establish compact baseline models | Statistical/compression/copy-mutate and small neural comparisons with held-out metrics | Validated data and splits |
| Compare learned mechanisms | Causal tests across known synthetic processes and manuscript models | Validated predictive baselines and controlled synthetic tasks |
| Investigate latent families or segmentation | Frozen rules and held-out comparisons against matched controls | A result justifying this branch |
| Introduce visual constraints | Traceable image/text alignments and independently justified weak constraints | Verified images, annotations, and a question benefiting from them |

Choose the next step by expected information gain, reproducibility, cost, and ability to rule out competing explanations. There is no reason to train a model or build a retrieval stack before it answers a defined question.

## Priorities suggested by the first experimental round (historical record)

1. **Separate distant content from order.** Hold prefix length and the last 16 units fixed; compare frequency-matched prefix replacements within and across manuscript sections, with explicit histogram/adaptive-frequency predictors. This tests the EXP-0003 suggestion without assuming linguistic meaning.
2. **Learn a causally useful predictive state.** The EXP-0004 supervised state readout worked, but patching its weight directions did not. Review causal representation-alignment methods before attempting optimized low-rank interventions or state-machine extraction. Use a fresh synthetic holdout; this round's synthetic test is now exposed.
3. **Stress-test ambitious generator theories.** Extend calibration to state-dependent alphabets, homophonic classes, insertion/null processes, and copy/mutate systems matched for superficial statistics. Include generator-family holdouts and impossible-label controls. Eventually ask whether a compact discovered mechanism predicts unseen outcomes, not merely whether activations look similar.
4. **Develop label-free candidates.** Voynich lacks the synthetic state/role labels used by these probes. Test unsupervised candidate state/equivalence classes and complexity penalties on synthetic tasks before assigning manuscript glyphs to signal or filler. Keep exact historical semantic claims downstream of independently testable rules.

Every priority needs a new bounded registration and method-specific source review. No final manuscript test scoring or paid API spending is scheduled by this backlog.

## Updated priorities after EXP-0005/0006/0007

The first two priorities above have now been attempted. EXP-0005 finds both distant content and order sensitivity; EXP-0006 achieves late output steering with known-rule supervision, but its output-weight control matches that effect. EXP-0007's restricted group-order/form contrast remains small and uncertain. These outcomes motivate the following **unexecuted** proposals:

1. **Line-aware production versus ordinary recency.** Hypothesis: lines act as processing units, reset points or formatting templates, rather than distant text acting as one undifferentiated stream. Selectively move definite spaces versus line boundaries while preserving nonseparator order and matching changed positions. Compare simple distance-weighted histogram and line-position predictors; balance category and frequency effects. Success would identify model sensitivity to layout under controls, not prove that the author used an encoding reset.
2. **Discover predictive states without labels.** Hypothesis: contexts that imply the same distribution over future strings belong to reusable latent states, even if their surface spelling differs. On fresh synthetic families, infer states from future-prediction signatures, penalize the number of states, and freeze a transition model before evaluating independent sequences. Compare against visible-symbol clusters, n-gram states and copy/mutate baselines. Evaluate known labels only afterward; do not use them for selecting clusters or intervention directions. This is the missing bridge from our supervised toy analyses toward unknown text.
3. **A stateful alphabet versus copy/mutate machinery.** Build several explicit competing generators: state-dependent symbol maps, homophonic symbols with ambiguous fillers, and edits of recently produced forms. Match corpus length and selected simple statistics, then reserve different diagnostics, alphabets/keys and generator families for testing. Ask whether the inference procedure identifies which *observable* mechanisms can be distinguished. Preserve ambiguous cases rather than forcing a cipher/hoax verdict.
4. **Intervene on an earlier computation and test downstream consequences.** Late output steering is insufficient. Use new synthetic evaluation pools and interventions on an inferred state that must affect multiple future decisions or transitions consistently. Require shuffled-supervision, random-model, readout-direction and distribution-preservation controls. Compare final-position and prefix-wide effects explicitly; the EXP-0006 early failure does not establish absence of state information elsewhere.

Review primary method and Voynich/cipher precedents before implementing each proposal. Define finite compute/RAM budgets and what result would change our mind. Do not treat the user's available RAM or credit estimate as a reason to scale a model without an information-gaining test. These are hypotheses and candidate experiments, not promised future background jobs.
