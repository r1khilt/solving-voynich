# JSPACE-0001 execution details frozen before development scoring

This operational supplement implements the previously registered six-condition experiment without changing its task splits, layer grid, strength, scoring, or success criteria. It is written while independent corpus calibration runs. Development and final have not been scored at writing time.

## Execution equivalence and auditability

The original adapter repeats the entire prefix computation for every generated token, editing only the same earlier prefix location each time. The evaluation adapter builds a fresh key/value cache during the edited prefix computation and then decodes future tokens through the original model. At source block l, its own cached keys/values were computed before the post-block edit; blocks above l cache the consequences of that edit. This matches the original causal semantics and does not repeatedly inject edits into newly generated tokens.

Before any development scores, compare both adapters on three generic prompts, clean and patched at block15, across four continuation steps. Require identical argmax tokens and maximum logit error below .002 at every check. The independently expanded neuron recorder must also agree with the original forward within that bound. If this qualification fails, stop; it cannot be dismissed because answers look reasonable. Qualification prompts contain no development/final examples.

Freeze a separate evaluation manifest with source, lens, input, fit, qualification, and this supplement's hashes. Every clean baseline stores the actual rendered token IDs, generated tokens, and a digest of its residual captures. Record every intervention's full output, accepted strings, baseline correctness, matched donor correctness, relative perturbation norms, before/after country coordinates, and truncation. Six conditions are evaluated on the same records. Invalid geometric swaps count as failures and are retained; any identity generation mismatch aborts scoring.

## Denominators and selection

The primary desired-change denominator is **all semantic records with a correct clean recipient answer**, fixed across intervention conditions. Donor-correct conditioning is a separately reported subset. Also report all semantic records. Copy preservation uses **all copy records**, a stricter rule than conditioning on clean copy correctness; copy examples never enter desired-change denominators. No filters use a successful perturbed answer or the sign of the initial country readout.

Development selection and final criteria are exactly those in the original registration. Every primary relation must be represented among eligible final examples; absence fails the corresponding criterion. Final evaluation requires an unchanged development summary and a saved selected-layer artifact. Completed splits refuse repeat scoring. Interrupted work resumes the same observation grid and cumulative two-hour evaluation budget. The budget covers development plus final computation; independent calibration has its own four-hour cap.

No confidence claim treats examples as independent countries. Report grouped counts by family, relation, country pair, and template/paraphrase grouping. Final contains only three unordered country pairs. Descriptive per-pair/leave-one-pair-out results are more appropriate than a spurious tiny p-value from hundreds of dependent prompt rows.

## Geometric interpretation

For unit country rows u and v, the registered coordinate exchange is exactly a reflection along the normalized contrast n=(u-v)/||u-v||: h' = h - 2(n.h)n. It therefore preserves the Euclidean norm of h and is its own inverse. This is an algebraic consequence of this two-row intervention, not evidence that the model uses a one-dimensional semantic variable. A negative or absent initial country contrast can make the requested exchange behaviorally unhelpful; report that failure rather than selecting only favorable examples.

Random and orthogonal controls match the swap displacement norm. The raw-unembedding control performs its own coordinate swap and can have a different displacement norm; record that difference and do not attribute an advantage uniquely to geometry when it could arise from intervention strength. The full-donor control is much broader and tests whether the chosen site can carry an alternative answer-producing state. Success there is not selective concept identification.

The two-token-form average, normalization, and layer choice are frozen. The small dictionary was selected around known countries, so neither successful readout nor successful swapping establishes a comprehensive J-space. Further neuron/dose/position analyses are exploratory, must retain failures, and cannot replace the original final decision.
