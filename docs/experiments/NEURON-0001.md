# NEURON-0001: transferring a country change between different questions

Registered during JSPACE-0001 calibration, before neuron selection or any causal outcome. This is a bounded companion investigation on the same frozen Qwen model, not a new manuscript result. It does not depend on JSPACE-0001 succeeding and cannot replace its final decision.

## Question

Can a sparse set of MLP neurons carry a country change across query functions? For example, can a change measured between two *currency* questions make a *capital* question return the donor country's capital, while preserving unrelated copying? This is stronger than transplanting a donor answer to another instance of the same question.

The reasoning, scaling symmetry, and source review are in `docs/research/latent-mechanisms-2026-09-21/CAUSAL_GEOMETRY_NOTES.md`. Arithmetic cross-format studies motivate the design but do not establish the result in this model or domain. Unit responses are not assumed monosemantic.

## Frozen design

- Same cached model files, numerical precision policy, task seed51021, and chat rendering as JSPACE-0001. No weight updates or paid services.
- Use **development tasks only**. Candidate neurons are selected from the first paraphrase of both semantic families, using one direction of each unordered country pair: 24 source/donor comparisons, four relations, three country pairs, two families. No output-correctness filter during selection. The second paraphrase supplies 48 semantic evaluation records. All24 development copy records supply nuisance-preservation controls. This is a paraphrase holdout, not unseen countries, facts, or languages.
- MLP layers `[7,15,23,31]`, zero based. Candidate counts `[16,64,256]`. **Primary layer23/count64**, chosen before scores; other settings are exploratory and cannot rescue the primary decision.
- At every transformer block, record the last rendered prefix position's residual, attention write, MLP write, and all12288 SwiGLU activations. Keep these arrays in ignored outputs. Validate the explicit trace against the ordinary model.
- Let Δa be the natural donor-minus-source MLP activation difference and w_j the down-projection column. Rank units using contribution energies e_j=(Δa_j)^2||w_j||², normalized to sum1 within each comparison. Average within each relation; rank by the **minimum across the four relation means** to favor units that change across query functions. Stable descending sort with lower unit index as tie break. This statistic is invariant to reciprocal scaling of an up-projection row and down-projection column. It is descriptive candidate selection, not proof of a circuit.

## Interventions

Use a cyclic alternate query: capital←currency, currency←capital_continent, capital_continent←language, language←capital. In the alternate query, compute the donor-minus-source neuron difference. Add only selected coordinates of that difference to the recipient's original MLP activation at the same layer/last prefix position. The desired answer uses the **recipient query** and **donor country**.

Conditions: selected neurons/same-query difference; selected neurons/cross-query difference; random neuron set of equal size, excluding the selected set, with Gaussian coefficients rescaled to match the cross-query edit's **residual displacement norm**. Also retain identity and full-MLP cross-query difference as controls. Copy cross-query changes are measured from the corresponding indirect-fact capital questions; same-query copy changes use the matched copy donor. Output scoring and truncation rules match JSPACE-0001; full strings and both all-case/clean-correct denominators are retained.

An MLP activation increment η is implemented as an equivalent post-block residual increment W_down η, since Qwen has no nonlinear transformation between the down projection and that residual addition. Require numerical agreement with an explicit neuron increment before scoring. Future tokens receive the effects through the prefix cache; newly generated neurons are not repeatedly edited.

## Primary decision and limits

At layer23/count64, require cross-query donor-answer success >=50% among initially correct semantic examples, advantage over matched random neurons >=20 percentage points, >=30% in every relation, and >=95% correct on all copy controls. Report same-query and full-MLP success, original-answer retention, all-case denominators, wrong-query-answer intrusions, and per-pair results. No significance claim from treating reverse directions or shared country facts as independent samples.

Failure can indicate query-dependent representation, poor candidate selection, an unsuitable site, or insufficient competence; it does not prove absence of country knowledge. Successful sparse cross-query transfer would support a portable country-change mechanism within this limited panel, but would still not establish necessity, an exhaustive circuit, or decipherment. Neuron deletion/rescue studies would require their own registration.

## Resources and audit

Local additional cap45minutes and45GB MLX allocations; no concurrent second dense8B model while JSPACE-0001 occupies memory. Trace and score sequentially with the same loaded model. Expected scale: hundreds of complete36-layer traces and roughly3200 short generations. Retain source/input/selection/trace digests, exact seeds, counts, and wall time. No paid compute, final Voynich holdout, or JSPACE-0001 final examples. Store compact summary/figures in `results/NEURON-0001`, bulk tensors and row logs in `outputs/NEURON-0001`. Abort on numerical/identity failure; incomplete work is labeled incomplete.
