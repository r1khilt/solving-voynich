# NEURON-0001: cross-query neuron transfer failed

**Registered decision: FAIL.** At the preselected block24 (zero-based23), changing64 selected MLP neuron coordinates using a country difference measured in another query produced0/31 desired donor answers among initially correct semantic cases. The matched random-neuron control also produced0/31; copying remained24/24.

## Completed study

The registered24 training comparisons selected units without filtering for correctness. Evaluation used48 semantic records from a distinct paraphrase plus24 copy records. Four layers, three sparse counts and same-query/cross-query/random/full-MLP/identity conditions produced3168 generations. The run retained97 complete36-layer input traces, including attention writes, residual states, MLP writes, SwiGLU gates, values and activations. Runtime1249.30s (20m49s), peak MLX allocation28.798GB; no cap reached.

The actual model's cache/trace qualification passed. Explicit neuron increments and equivalent residual writes agreed to relative error at most0.0001076, below0.002. Every identity output matches its baseline. Independent exact-label rescoring reproduced all archived correctness flags, and hashes/source checks passed.

No tested layer/count produced a desired donor answer among initially correct cases, including the full12288-neuron cross-query change. This is a stronger negative observation than the primary64-neuron failure, but all these settings share the one-position intervention limitation. They are not independent replications.

## The edits were real, but usually insufficient

At block24, the median64-neuron cross-query edit was3.94% of the recipient residual norm; the full-MLP edit was10.84%. At block32 these rose to16.17% and31.88%. These are nonzero changes, and some generated strings changed. For example, block32's64-neuron same-query condition changed7/48 semantic strings, while its cross-query condition changed4/48. None counted as the intended donor answer among the31 initially correct cases. Avoid interpreting unchanged top-choice answers as zero changes to every output probability; probabilities were not recorded for this study.

Baseline competence is a material limit:31/48 semantic records were initially correct, comprising18/24 indirect facts and13/24 anchored aliases. The eligibility filter was frozen and applied consistently, but leaves a small and dependent panel.

## Selection coverage audit

A post-hoc descriptive check finds that the selected64 units at block24 carry a median20.01% of the summed, column-norm-weighted neuron-change energy on selection examples and14.97% on held-out wording. This is not explained residual variance: cross terms can reinforce or cancel. The decline limits selection transfer, but does not alone explain the failure, since the full-MLP cross-query condition also failed. Exact summaries for all registered layers/counts are in `selection-energy-audit.json`.

## Natural response analysis

The symmetric gate/value decomposition reconstructs measured SwiGLU changes with maximum relative error2.67e-6. Country-change directions vary with the queried fact. The descriptive residual interaction statistic averages0.410 at block24 and0.811 at block36, where zero would mean identical changes across all four queries. MLP-write interaction averages0.499 and0.879 respectively. These values quantify query dependence on this panel; there is no universal threshold for a concept representation.

Units were ranked by down-column-norm-weighted change energy, which is invariant to reciprocal up/down scaling. That avoids one arbitrary activation-scale confound, but does not identify an exhaustive circuit, establish necessity, or handle all forms of superposition. The failure could reflect missing attention paths, distributed/redundant coding, query-dependent representations, candidate selection, or a poor intervention site. It does not show that neurons are unimportant or that the model lacks country knowledge.

Artifacts: `results/NEURON-0001/{inputs,selection,qualification,neuron-edit-qualification,results,decision,natural-response-analysis}.json`, `query-dependence.png`, and lossless `audit/observations.jsonl.gz`. Raw traces and down-projection norms are ignored and hashed. No model retraining, paid API, new language holdout or Voynich interpretation.
