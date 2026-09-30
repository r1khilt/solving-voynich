# DENSE-SOURCE-001 results

**The compiled lookup tables preserve the source and make this tested workload
faster.** All128readings across both representations are identical, including
plaintexts, marginal/MAPscores, reachable states and edges. No key was selected
or transfer passage opened.

| Fitting workload,8keys×4records | Lazy seconds | Dense seconds | Observed ratio |
| --- | ---: | ---: | ---: |
| Original positive B-key1 | 0.636426 | 0.224992 | 2.83× |
| Original glyph-shuffle B-key1 | 27.287209 | 4.297465 | 6.35× |

Tables for1,447,724states took0.263501seconds to construct and occupy399,571,824
bytes, plus the original counts/runtime. No source training, smoothing, support
threshold, probability precision or decoding rule changed. This is not a universal
speedup estimate; candidate ambiguity and cached-context reuse affect timing.

Sampled376source rows and8,648transitions match the lazy adapter exactly, with
maximum probability delta0. All dense rows pass positivity/normalization checks.
The post-run audit verifies78frozen source/input bindings, all reading-archive
hashes,128complete output comparisons and graph-count/timing arithmetic. It does
not independently recompute all33millionprobability or transition entries.

A separate cold-lazy parent profile took2.440seconds: `_step` has1.229cumulative
seconds and probability lookup0.870seconds. Other cumulative entries overlap
these calls and must not be added together. Lookup work is a measured bottleneck,
not just a conjecture about why null cases are slow.

Source16fd41dce73c672c0bb2473161809cc7033ca52awas published/remoteverified before
one invocation. Total35.587wall/35.538CPU seconds, peakRSS1,021,526,016bytes,0paid;
outer37.417seconds/exit0. No restart or cap increase.23newartificialtests and
full2,118tests+23subtestsPASS,10skips; later reader integration reaches2,129.

The active original key-bank fitting campaign still uses its original frozen
source and budgets. Dense lookup is admitted only through a separate new pipeline
freeze. Earlier timeouts or incomplete banks remain failures.
