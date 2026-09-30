# DENSE-SOURCE-001: identical source probabilities, compiled lookup tables

Prospective computational benchmark, no new statistical model, key selection or
transfer reading. Original active KEY-BANK-FIT-001 source and budgets remain
unchanged. Prior decipherment/Voynich scope remains in
[KEY-SOURCE-DIAG-001](KEY-SOURCE-DIAG-001.md). This cannot improve semantics by itself.

## Construction and review

[Kanda, Akabe and Oda, section2.3](https://arxiv.org/pdf/2207.13870)
describe longest-suffix failure transitions in Aho–Corasick automata. Their
double-array representation differs from the dense tables proposed here; no
reported speedup is transferred to this workload. Attempts to fetch the original
1975paper from ACM and an academic mirror failed; do not claim a fresh full-text
review of it. Our construction follows the standard recurrence and the audited
source's stronger closure property.

The retained context set is closed under dropping first or last characters.
For context c and letter a, use ca when retained; otherwise use the precomputed
transition from c[1:] on a. The failure context is shorter, so construction by
depth is well-founded. Root transitions fall back to root when no child exists.
This implements the same longest-retained-suffix lookup as the lazy adapter.

Compute each float64 probability row with the unchanged count interpolation:
root half-count smoothing, or `(counts(c,a)+tau*q(a|c[1:]))/(total(c)+tau)`.
No quantization, threshold, tau, order or count changes. The dense23-column
float64 probability and uint32 transition arrays require399,571,824bytes for
1,447,724states; count archives and temporary arrays add memory. Allocate only
after checking a768MiB table cap. The retained source is the audited large
compact order12/minimum4/tau64model; no fitting or new corpus access.

## Fixed benchmark and checks

Before full-size use,23artificial tests compare every row/transition against
the lazy representation across orders0/1/3/8/12 and three support thresholds,
including all-empty higher levels. Exact per-key and k-best readings/scores are
identical on exhaustive short observations; caps, leading-zero contexts and
invalid settings are checked.

After code/protocol publication, construct the large tables once. Compare up to
32evenly spaced retained states per depth against lazy rows/all23transitions.
Require probability delta≤1e-15and identical transitions; all rows must remain
finite, positive and normalized. This samples numerical equivalence and does
not independently re-evaluate all33millionentries.

Use only four fitting records and the original learned parent from old positive
B-key1and its glyph-shuffle control. Select8equally spaced candidate indices from
each complete one-move neighborhood, including endpoints. Score allfourrecords
under both source representations with unchanged exact decoder/500kstatecap.
Positive order: lazy then dense; shuffle order: dense then lazy. Lazy caches
start empty for each case. Require identical complete outputs, including strings,
scores and graph counts. Report construction cost and each backend's time.
These are exposed hardware measurements, not accuracy evaluation.

Finally profile one separate cold-lazy decode of the shuffle parent's first fit
record; exclude it from timing comparisons. Keep the20largest cumulative entries.
This can identify lookup overhead but cannot prove all runtime is removable.

OneCPU process/thread, no GPU/paid services.420wall/CPU seconds,480outer timeout,
sampled2GiB RSS cap; expected1–5minutes, no rerun/cap extension. It may coexist
with the two original fitting workers within a10GiB planning allowance. A failure
retains partial files and process return code. Any adoption in empirical fitting
or transfer needs its own prospective freeze; never replace an active campaign's
module or relabel earlier failures.

```
PYTHONPATH=.:src .venv/bin/python scripts/benchmark_dense_source001.py --freeze COMMIT
```
