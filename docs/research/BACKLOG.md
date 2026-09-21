# Deferred research candidates

**Status: active research.** Source review, corpus preparation, 18-run architecture comparison, first causal context tests and initial synthetic calibration are complete. The table preserves the original candidate inventory; current priorities follow it.

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

## Priorities suggested by the first experimental round (not yet executed)

1. **Separate distant content from order.** Hold prefix length and the last 16 units fixed; compare frequency-matched prefix replacements within and across manuscript sections, with explicit histogram/adaptive-frequency predictors. This tests the EXP-0003 suggestion without assuming linguistic meaning.
2. **Learn a causally useful predictive state.** The EXP-0004 supervised state readout worked, but patching its weight directions did not. Review causal representation-alignment methods before attempting optimized low-rank interventions or state-machine extraction. Use a fresh synthetic holdout; this round's synthetic test is now exposed.
3. **Stress-test ambitious generator theories.** Extend calibration to state-dependent alphabets, homophonic classes, insertion/null processes, and copy/mutate systems matched for superficial statistics. Include generator-family holdouts and impossible-label controls. Eventually ask whether a compact discovered mechanism predicts unseen outcomes, not merely whether activations look similar.
4. **Develop label-free candidates.** Voynich lacks the synthetic state/role labels used by these probes. Test unsupervised candidate state/equivalence classes and complexity penalties on synthetic tasks before assigning manuscript glyphs to signal or filler. Keep exact historical semantic claims downstream of independently testable rules.

Every priority needs a new bounded registration and method-specific source review. No final manuscript test scoring or paid API spending is scheduled by this backlog.
