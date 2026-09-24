# TEACH-0014 prospective local-compute amendment

**Status:** answer/parser scientific launch is admitted only after a clean,
source-matched new benchmark passes. This amendment follows the
source-matched benchmark stop recorded in NOTEBOOK NB-TEACH-61. It changes the
local wall-time ceiling from 8 to **12 hours** without changing the 11 arms,
two seeds, 6,000 updates per arm and seed, model architecture, answer labels,
optimizer, confirmation suite or outcome thresholds. The original stopped
benchmark at `results/TEACH-0014/benchmark.json` remains untouched. No
confirmation model prediction was inspected before this amendment.

The first benchmark from committed HEAD `acc7967e6051e5ce93842e7b2b003d3da8b31356`
measured 264 MPS optimizer updates. Its median-based training projection was
22,070.7485 seconds (6.13076 hours); the registered 1.5 multiplier and
1,800-second evaluation reserve produced 34,906.1228 seconds (9.69615 hours),
above the original 28,800-second cap. The peak sampled MPS allocation was
450,110,464 bytes, below the unchanged 12 GiB cap. A 12-hour ceiling leaves
about 2.30 hours beyond that conservative projection, but neither a short
benchmark nor sampled allocation proves sustained thermal throughput or peak
physical-memory safety. A 4 GiB combined artifact cap remains unchanged.

Each amended benchmark must run from a clean committed version of *all* source
and audit files in `SOURCE_PATHS`, including this amendment, in a fresh result
and checkpoint directory. The original stopped result and first amended v2
benchmark are read-only historical evidence. The final 24-update-per-arm MPS
backward benchmark under `results/TEACH-0014-v3` must pass its own
12-hour projection gate before scientific training. Source admission is fixed
in `LAUNCH_ADMITTED=True` after parser and artifact checks passed independent
fixtures; `check_benchmark` still rejects changed source, a failed projection
or a missing benchmark. No paid API, cloud GPU
or external spend is authorized by this amendment.

The first amended benchmark from source `a03d0fb` passed the independent
resource audit in45.696 seconds: 21,115.2893s raw training projection,
33,472.9339s (9.29804h) conservative, and456,139,776 sampled MPS bytes.
Its SHA-256 is `b7231e4b8ae226860b82702c55085fa5c6ab1cbafebd31573b271d360d1f6394`.
It is preserved in the v2 directory. The final launch-admitted source differs,
so a second source-matched amended benchmark must run in the new **v3** result
and output directories before training. Source matching forbids reusing v2.

The parser archive freezes the **zero-logit threshold** before model outcomes.
Candidate slots are known adjacent ordinary-symbol occurrence pairs, so each
slot corresponds directly to a visible occurrence pair; no bipartite matching
ambiguity exists in this architecture. Precision is measured over all proposed
rows, recall over true signal-path rows. A complete-table hit requires the
selected candidate set to equal all true visible rows. The independent symbolic
solver rejects duplicate left-side mappings or missing requested edges. It
scores two-hop composed answers and exact four-item factorial groups. Marked
means at least one visible `EDGE` marker in the body; fully marker-free means
none. The scorer requires every saved gate vector, both seeds, all 19 panels,
and the frozen suite hash. These measures diagnose the synthetic public grammar;
they cannot by themselves establish unknown-language parsing or a manuscript
decipherment. Sampled checkpoint replay must also verify gate logits, not only
answer logits, before a parser result is trusted.

The existing diagnostic decision order remains: qualified oracle first, then
parser, then routing and causal tests. Parsing-bottleneck, routing-bottleneck,
interface-mismatch and reusable J-space claims additionally require the
specific frozen interventions in `TEACH-0014-design.md`. If those are not
implemented and audited after training, the answer/parser records stand alone
and the mechanism labels remain withheld.
