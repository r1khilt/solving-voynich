# TEACH-0011 results: the key acts through Q to switch between G rows

**Registered outcome after a transparent numerical repair: `G-EDGE-SUPPORTED`. Descriptive factorial label: `Q-only sufficient`. Independent audit: PASS.** The transplanted query key changes the layer-2 query vector, which redirects all four heads from the G row keyed by the original base key to the row keyed by the transplanted donor key. The recipient G rows already contain both object values, so changing K or V is unnecessary.

## What ran and numerical qualification

The study generated 256 new cross-G groups from seed `71111`, split 128 discovery/128 confirmation. It reproduced TEACH-0010's upstream edit: the same donor-G0 cut-2 query key was inserted into each base recipient G0/G1/G2 state. Layer 2's four fixed heads were decomposed into Q, K, V, query-to-source weights and six additive source contributions. All eight base/edited Q×K×V cells and all 63 nonempty source masks were evaluated.

The first execution was invalid only at the projected full-source identity: float32 output projection amplified a sub-threshold head-space discrepancy to `1.1920929e-6`. Its artifacts are preserved. Applying the same frozen linear projection in float64 and casting the delta back reduced the discrepancy to `4.77e-7`; no model weight, example, intervention, selection rule or threshold changed. The valid run is therefore a non-blind numerical repair. Other valid-run numerical maxima were `8.05e-7` for query-head reconstruction and `2.39e-7` for source summation. Runtime was 1.133 seconds on CPU.

## The transplanted key acts through Q alone

On confirmation, the factorial was effectively binary. Every cell with edited Q succeeded; every cell with base Q failed:

| Q/K/V sources | Seed 0 items / groups | Seed 1 items / groups |
| --- | ---: | ---: |
| BBB | 0/384 / 0/128 | 0/384 / 0/128 |
| **DBB — edited Q only** | **369/384 / 114/128** | **373/384 / 119/128** |
| BDB — edited K only | 0/384 / 0/128 | 0/384 / 0/128 |
| BBD — edited V only | 0/384 / 0/128 | 0/384 / 0/128 |
| **DDB — edited QK** | **369/384 / 114/128** | **373/384 / 119/128** |
| DBD — edited QV | 369/384 / 114/128 | 373/384 / 119/128 |
| BDD — edited KV | 0/384 / 0/128 | 0/384 / 0/128 |
| DDD — full edited QKV | 369/384 / 114/128 | 373/384 / 119/128 |

All successful cells passed 256/256 cross-G non-injection. The key representation therefore functions as a query/address: substituting its Q projection is sufficient to retrieve the recipient-specific donor object from an otherwise clean base table state.

This complements TEACH-0009 rather than contradicting it. In the first lookup, the name/query address remains fixed while F-row assignments change, so the new key arrives through V. In the second lookup, the G table remains fixed while the intermediate key changes, so the new object is selected by changing Q.

## The causal source is the two-row G contrast

Discovery selected physical slots G0+G1 as the smallest shared source subset, with minimum-both-seed 97.14% items and 91.41% exact groups. Confirmation reproduced the all-head effect:

| Source contribution delta | Seed 0 items / groups | Seed 1 items / groups |
| --- | ---: | ---: |
| **Both G rows** | **369/384 / 114/128** | **373/384 / 119/128** |
| All six sources | 369/384 / 114/128 | 373/384 / 119/128 |
| Donor-key G row alone | 5/384 / 0/128 | 6/384 / 0/128 |
| Base-key G row alone | 0/384 / 0/128 | 0/384 / 0/128 |
| All non-G sources | 0/384 / 0/128 | 0/384 / 0/128 |
| Norm-matched random delta | 0/384 / 0/128 | 0/384 / 0/128 |
| Cyclic wrong-recipient delta | 0/384 / 0/128 | 0/384 / 0/128 |

Why are both rows needed if one row contains the desired object? The intervention is a **difference** between two attention computations. It must simultaneously remove the old base-key-row contribution and add the new donor-key-row contribution. Adding only the positive donor-row change generally leaves the old object evidence in place; removing only the base row does not supply the new object. The pair is the sufficient causal contrast.

## Every head changes its address

Attention mass makes the routing switch visually obvious, although the causal factorial and source interventions establish it. Examples:

| Model/head | Base run: base-key G | Base run: donor-key G | Edited run: donor-key G |
| --- | ---: | ---: | ---: |
| Seed 0, head 0 | 99.27% | 0.38% | 99.06% |
| Seed 0, head 1 | 96.64% | 2.40% | 96.08% |
| Seed 0, head 2 | 97.85% | 1.03% | 97.67% |
| Seed 0, head 3 | 86.47% | 8.95% | 85.46% |
| Seed 1, head 0 | 98.80% | 0.43% | 99.00% |
| Seed 1, head 1 | 58.52% | 20.54% | 57.31% |
| Seed 1, head 2 | 96.45% | 2.01% | 96.88% |
| Seed 1, head 3 | 86.84% | 8.32% | 87.79% |

For the edited columns, the mass shown is on the donor-key row; base-key mass collapses correspondingly. Head 1 in seed 1 is less sharply specialized but flips in the same direction. This shared routing behavior explains why all four heads together are robust despite different smaller-subset effects across seeds.

## Verification, completed synthetic circuit and limits

`scripts/teacher0011_analyze.py` independently regenerated the suite and 63-mask selection, verified frozen source/checkpoint/row hashes and all labels, rescored 20 confirmation conditions, checked attention-mass normalization and reproduced the registered decisions. It reports `audit: pass`; it does not rerun inference.

The full qualified synthetic circuit is now:

1. **Name-to-key lookup:** the query uses stable QK routing to address the matching F row; that row's V content flows through layer-1 heads 1+3 and writes a distributed reusable key.
2. **Key-to-object lookup:** the key changes layer-2 Q, redirecting all four heads from the base-key G row to the donor-key G row; the two-row contribution contrast writes the recipient-specific object.
3. **Output:** the edited attention residual is already sufficient; the layer-2 MLP has no independent donor-answer route on these counterfactuals.

This is a causally validated two-hop retrieval algorithm, not merely an attention pattern or linearly decoded feature. It remains heavily scaffolded: records are parsed, roles and row boundaries are explicit, the query has its own slot, and supervised answers are abundant. No Voynich token, plaintext, language or cipher mechanism was tested. The next serious bridge is to remove this parser by training on raw serialized rows with variable order, distractors and boundary ambiguity, then ask whether the same intermediate-key causal abstraction re-emerges.
