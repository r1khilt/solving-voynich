# TEACH-0009 results: the two heads read key content from the queried F row through V

**Registered outcome after a transparent numerical repair: `SOURCE-EDGE-SUPPORTED`. Descriptive QKV label: `V-only sufficient`. Independent audit: PASS.** The two TEACH-0008 heads already point at the row matching the query. Changing only their V inputs transfers the counterfactual key, while changing Q, K or QK without V does nothing. Source decomposition localizes the effective change to the queried F row.

## What ran and why the result is non-blind

The study generated 256 new cross-G groups from seed `69111`, split 128 discovery/128 confirmation. The layer, query slot and heads `[1,3]` were frozen from TEACH-0008. For each head, it decomposed the query output into Q, K, V, six attention weights and six additive source contributions. It then ran all eight base/donor Q×K×V combinations and screened all 63 nonempty source-slot subsets on discovery.

The first execution was invalid: explicit matrix-multiply/softmax/value outputs differed from TEACH-0008's scaled-dot-product head outputs by `1.19e-6` and `1.43e-6`, above the frozen `<1e-6` gate. Its artifacts are preserved. The corrected implementation used PyTorch's math scaled-dot-product path, which returns both its attended output and the exact weights it used. No examples, interventions, controls, selection rules or thresholds changed. The valid run is therefore a non-blind numerical repair.

In the valid run, maximum head-reconstruction errors were `9.54e-7`; six returned-weight×V source contributions summed to the backend output within `2.39e-7`; full source deltas matched full hybrid deltas within `9.54e-7`. Runtime was 1.533 seconds on CPU.

## Q and K provide stable addressing; V carries the changed key

On each confirmation seed's 384 recipient-table items:

| Q/K/V sources | Meaning | Seed 0 items / groups | Seed 1 items / groups |
| --- | --- | ---: | ---: |
| BBB | base identity | 0/384 / 0/128 | 0/384 / 0/128 |
| DBB | donor Q only | 0/384 / 0/128 | 0/384 / 0/128 |
| BDB | donor K only | 0/384 / 0/128 | 0/384 / 0/128 |
| **BBD** | **donor V only** | **375/384 / 120/128** | **382/384 / 126/128** |
| DDB | donor Q+K, base V | 0/384 / 0/128 | 0/384 / 0/128 |
| DBD | donor Q+V | 373/384 / 120/128 | 382/384 / 126/128 |
| BDD | donor K+V | 379/384 / 125/128 | 383/384 / 127/128 |
| DDD | full donor QKV | 377/384 / 124/128 | 383/384 / 127/128 |

Every cell preserved 256/256 cross-G non-injection. Donor V therefore carries a reusable key rather than a fixed donor-G0 answer. Donor Q or K is unnecessary here and occasionally makes the finite intervention slightly worse, which is consistent with small routing perturbations around an already-correct address.

This result has a task-specific reason. Base and donor examples keep the same query name and F-row names; they swap the keys assigned to those names. The correct row address should stay fixed while the selected row's content changes. The experiment shows the model implements that factorization internally. It does not establish that arbitrary linguistic meaning always lives in V or that QK is generally unimportant.

## One dynamically selected F row supplies essentially the whole key update

Discovery's smallest **fixed physical-slot** subset was both F rows, with minimum-both-seed 98.96% items, 97.66% groups and 100% non-injection. Both are needed in a fixed mask because the queried name appears in row 0 for some examples and row 1 for others.

Confirmation aligned rows by which one actually matched the query:

| Source contribution inserted | Seed 0 items / groups | Seed 1 items / groups |
| --- | ---: | ---: |
| Both F rows, frozen discovery subset | 376/384 / 123/128 | 383/384 / 127/128 |
| All six sources | 377/384 / 124/128 | 383/384 / 127/128 |
| **Queried F row only** | **379/384 / 124/128** | **383/384 / 127/128** |
| Other F row only | 0/384 / 0/128 | 0/384 / 0/128 |
| All four non-F sources | 0/384 / 0/128 | 0/384 / 0/128 |
| Norm-matched random write | 1/384 / 0/128 | 0/384 / 0/128 |
| Cyclic mismatched-group write | 13/384 / 2/128 | 12/384 / 1/128 |

The queried row alone is as good as or slightly better than all sources. The other row and all G/marker/query sources together are exactly inert under this counterfactual. This is a source-edge result: the causal information flow is from the matching F-row value, through heads 1 and 3, into the query residual.

## Attention weights agree with the causal decomposition, but are not the evidence by themselves

The query attention weights were already sharply concentrated on the matching F row:

| Model/head | Base queried row | Donor queried row | Base other row |
| --- | ---: | ---: | ---: |
| Seed 0, head 1 | 92.16% | 92.85% | 2.94% |
| Seed 0, head 3 | 88.90% | 90.19% | 1.74% |
| Seed 1, head 1 | 87.91% | 88.91% | 1.06% |
| Seed 1, head 3 | 95.39% | 95.57% | 1.53% |

The small base-to-donor changes in attention mass are not what carries the key: donor QK with base V produces 0/384. The causal evidence comes from the factorial interventions and source contribution controls; the weights only make the mechanism legible.

## Verification, implications and limits

`scripts/teacher0009_analyze.py` independently regenerated the deterministic suite, reconstructed the 63-mask selection, verified source/checkpoint/row hashes and every label, rescored all 19 confirmation conditions, checked attention-mass normalization and reproduced both decisions. It reports `audit: pass`. It does not rerun neural inference.

Together TEACH-0007/8/9 now support a concrete synthetic circuit:

1. the query/name establishes a stable address to the matching F row;
2. heads 1 and 3 read that row's value through V;
3. their joint output writes a distributed low-rank key into the query slot;
4. later layers use the recipient G table to convert that key into the answer.

This is substantially stronger than token cosine similarity, a probe or an attention heatmap: changing the proposed content channel changes behavior across new downstream mappings, while every wrong source and matched control fails.

The architecture still receives explicit typed rows, clean role boundaries, a dedicated query token and supervised synthetic answers. The counterfactual preserves row names and swaps only row values, making a stable-QK/changing-V solution especially natural. No Voynich glyph, latent, word meaning or cipher mechanism was tested. The next mechanistic step is to decompose how later layers read the query-key state together with G rows, including whether another attention edge performs key-to-object lookup and whether the same two-stage circuit survives unparsed sequences with distractors.
