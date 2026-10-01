# CENSORED-CONTEXT-001 results

The fixed contextual correctness/cost gate **PASS**. All960 native joint-prefix/closure attempts, all60 independent Python prefix references, and all192 legacy closed-scorer comparisons completed with positive values. Maximum discrepancy6.59383658785e−12nats is below the frozen1e−7 tolerance; all192 closed graph-node/edge/call counts agree exactly. [Registration](CENSORED-CONTEXT-001.md), [result](../../results/CENSORED-CONTEXT-001/result.json), [audit](../../results/CENSORED-CONTEXT-001/audit.json).

Registration15dea609b69e03456cbb791fb1c79a386bb3fb77 was committed, pushed and exactremote verified before ONErun65369 and ONEfullaudit71562, both terminal0. No retries, source changes, compilation, training, GPU, paid services or manuscript holdout were used. All186 frozen paths remain unchanged. Audit replays every native/Python/legacy value, graph count, selected key, source/array/library identity and decision. This is same-author replay; closed-reference checks share the native backend. Computational independence comes from Python prefixes and the preparation's5184 exact-rational tiny-source values, not independent expert review.

| Measurement | Run | Full audit |
| --- | ---: | ---: |
| Wall seconds | 3.513783 | 3.429922 |
| CPU seconds | 3.480161 | 3.411481 |
| Peak RSS bytes | 1,240,498,176 | 1,240,727,552 |
| Paid spend | $0 | $0 |

All1800wall/1600absoluteCPU/2GiB host/16MiB compact caps held. The conservative3–20minute stage estimate overestimated this workload. Source loading/admission overhead outside the reported stage interval is not part of the3.51seconds. The source has1,447,724 contexts and399,571,824 dense array bytes; the native wrapper pins those arrays. Python references own an additional copy.

Native calls consumed0.952275 seconds total for960 joint attempts. Whole-unclosed-prefix median2.173791ms/p953.862025ms; full-closed median2.129209ms/p954.258927ms/max10.624833ms. Earlier cuts are cheaper. Total native bridge edges3,608,495/nodes1,776,028; maximum per-attempt nodes11,780/edges31,430. Maximum conservative owned-work envelope78,143,588bytes excludes pinned source. Both wrapper validations scan the dense source, so construction costs are separate: native0.451862s/Python0.490279s across12queries. These are observed-bank costs; arbitrary proposed mappings can create different graphs, and full contextual sampling is not automatically qualified.

## Read-only post-outcome diagnostics

[Static diagnostic utility](../../scripts/summarize_censored_context001.py) and [receipt](../../results/CENSORED-CONTEXT-001/post-outcome.json) read only already-closed likelihoods and hash-bound parent banks; they perform no new source/scorer/search calls. This analysis is exploratory and does not change the gate.

The contextual and IID final scores disagree in ranking on552of1422 comparable key pairs; ties within1e−10 are excluded. Context-minus-IID log-correction ranges span13.0455 to187.4397nats across the twelve16-particle banks. Normalizing those correction factors gives maximum weight above.99 in4of12banks; inverse-squared-weight sums range1.0–2.58544. This is a fixed-bank concentration statistic, **not** an effective number of independent posterior samples or a valid global evidence/coverage estimate. Particles were the fixed first16 correlated IID-search outputs, not fresh contextual posterior draws. All192 contextual likelihoods are lower than the paired IID scores; this absolute difference alone does not establish a language or historical mechanism.

Ending the observation is a separate conditioning event. The full unclosed prefix can finish inside a two-glyph emission; a closed record cannot. Exact closure log-increment ranges span.0109362–10.2170nats within these banks, so treating EOS as just the same key-independent constant would be incorrect. In case0-root its normalized closure increment assigns.831467 to one of16particles; ten other banks are much less concentrated. End conditions need explicit monitoring in any future sampler.

## Implication

We can now cheaply evaluate the examined mappings with genuine source context, rather than discarding letter ordering. Context strongly changes candidate preferences, which argues against merely enlarging or reweighting the failed IID banks. The next experiment should initialize and update mappings under the contextual target, with prospective computation limits and a control that separates proposal quality from extra scoring work. Exact42-code row Gibbs updates have a clear target-invariance argument, but may still suffer coupled support barriers. Neither this benchmark nor its static analysis recovers a reading, qualifies posterior accuracy, identifies a historical language, or deciphers Voynich. Goal ACTIVE.

Preparation validation:2724tests+23subtestsPASS/13skip,167.78seconds; scoped new code/test lintPASS, five unchanged older full-tree findings retained. Publication validates all JSON/hash/resource/frozen-path/local-link bindings without rerunning the experiment.
