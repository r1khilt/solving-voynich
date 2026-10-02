# SOURCE-ACTION-BENCH-001 — Full-size training shape fits the Mac

ONErun51404/exit0 after registration29a8ecae5c777170c0857f8a574e6f527e0b7980
was pushed and exact origin verified. Nine frozen dependencies/input hashes
unchanged. [Prospective design](SOURCE-ACTION-BENCH-001.md),
[actual result](../../results/SOURCE-ACTION-BENCH-001/result.json),
[closed hash/resource/arithmetic check](../../results/SOURCE-ACTION-BENCH-001/closed-check001.json).
No retry, Latin/panel/holdout access or substantive language fit.

| Scale | Parameters | Mean optimizer update,4timed | Trace packing | Maximum batch/shape |
| --- | ---: | ---: | ---: | --- |
| Small | 5,775,918 | 0.085975s | 0.228840s | 4episodes ×2records ×448glyphs;448actions/episode |
| Large | 96,039,982 | 0.581279s | 0.232404s | Same |

Five static artificial-batch optimizer steps per scale, firstwarmup excluded
from timing. Falling loss on this repeated artificial batch is not an inverse
competence result. Every required gradient was present and finite. Initial and
final short SAMEweight MPS/CPUfloat64 legal logits agree within3.593e-6,
whole-path NLL within3.597e-6, both<=.002. Only twelve-source-letter traces
received these independent numeric checks, not the complete long shape.

Entire stage6.378919wall/2.939506CPU seconds, hostRSS1,644,249,088bytes and
sampled MPSdriver6,249,480,192bytes; all600wall500absoluteCPU4GiBhost8GiBdriver
bounds held. Admission excluded from stage timing but included in absolute CPU
limit. Torch2.14.0/macOS26.6.2/Python3.12.13,CPUthreads2/OMP1/OPENBLAS1,
allocator fraction.1543209531of55,662,788,608-byte recommended working set.
Memory checks are sampled; no continuous driver peak certificate.0USD.

Maximum-shape four20k-fit/twoseed/twoscale planning cost, including measured
trace packing per update, is45,139.951719seconds≈12.54hours. This EXCLUDES source
generation, inference, audit, preparation and thermal/runtime variability;
actual shorter Latin batches may differ. It is not an automatic training
launch or an estimate of time to decipher Voynich. Trace packing is already
slower than the small model update and deserves engineering attention.

ONE read-only closed check verifies input/freeze hashes, parameter/step counts,
timing/projection arithmetic, references and resource bounds. Its first attempt
used systemPython withoutTorch and failed before receipt/source/model execution;
using the project interpreter completed the same read-only check. No optimizer
or inference rerun occurred. This is not a full optimizer replay audit.

Separate [cached inference implementation](../research/source-action-cached-inference-2026-10-02.md)
now matches the reference on tiny CPUdouble prefix/rollout controls. Its GPU
and long-reading cost/equivalence remain unmeasured. Register that admission
panel, then an explicit fresh-key/source-controlled objective comparison before
substantive training. The prototype is untrained for actual Latin recovery;
original recovery remains FAIL and Voynich UNSOLVED.
