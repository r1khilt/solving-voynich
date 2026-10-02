# Valid encoding removes about seven bits, not the letter-assignment problem

Own read-only post-outcome analysis of the four frequency-order profiles in the closed [003 engineering panel](../experiments/COMPILED-INVENTORY-003-results.md). [Receipt](../../results/INVENTORY-ENTROPY-001/result.json), [exact-count/entropy checker](../../scripts/summarize_inventory_entropy001.py). No source scoring, compiler call, sampler, Gold, fitted bank, translation, training or historical evidence is involved. All input manifests/archive hashes and original frozen dependencies are checked.

The original uniform labelled dictionary prior has42^23 possible keys, entropy124.023300724bits. A valid full-record encoding event has H equally likely supported dictionaries, so its conditional prior entropy is log2(H). The reduction is−log2 π(B), the information in the observed Boolean support event alone. It is not the information in the full language likelihood.

| Exposed case | Remaining supported-key entropy, bits | Support-event reduction, bits | Conditional label-assignment entropy, bits |
| --- | --- | --- | --- |
| 0 | 117.024683102 | 6.998617622 | 82.566712881 |
| 1 | 116.397030056 | 7.626270668 | 82.575835259 |
| 2 | 116.431671343 | 7.591629380 | 82.613498968 |
| 3 | 116.508553272 | 7.514747452 | 82.595311967 |

For each k occupied unknown codes, the exact sampler uses probability b_kT23(k;k)/H, where b_k counts satisfying subsets and T counts onto assignments of23 labelled rows to the chosen k codes. The chain rule decomposes supported-key entropy into cardinality entropy + expected log2(b_k) + expected log2(T). All four exact integer products/sums reproduce H; the floating chain rule agrees within1e−10bits. The source-letter assignment term dominates, roughly82.6bits. Mean occupied codes conditional on support is roughly18.9; this is a dependent occupancy law, not42 independent coin flips.

These figures explain why zero-free initialization is useful but insufficient: it removes impossible encodings, yet random supported keys retain enormous assignment uncertainty. They do **not** imply82.6bits of irreducible posterior uncertainty, a cryptographic lower bound, or required brute-force runtime. Language context can constrain those assignments strongly; equivalence from unused rows can make full-key recovery harder than message recovery. These are whole-key **prior** entropies, not posterior or used-key entropies. A future blind message-recovery test must measure what the language-weighted search actually resolves.

This diagnostic was made while the fixed tempering engineering registration was being validated, before any of its original-source calls; it changes no predefined parameters, budgets, controls or success criteria. It supports prioritizing label/inventory exploration and calibrated language evidence over further claims based on successful support compilation alone.
