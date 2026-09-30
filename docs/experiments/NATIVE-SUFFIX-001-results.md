# NATIVE-SUFFIX-001: exact scoring benchmark passes on all sixteen cases

The native scorer preserves the tested likelihoods, supports and graph counts,
while reducing total measured scoring time from23.6634to1.5423seconds: **15.34×**
relative to the existing decoder with dense source tables. It is also12.04×faster
than the new Python marginal-only control. This is a hardware/workload result,
not an improvement in statistical accuracy or a universal speed guarantee.

| Fitting workloads | Full Python, seconds | Marginal-only Python | Native marginal | Full-Python/native ratio |
|---|---:|---:|---:|---:|
| Eight positives |1.392403|1.075582|0.192420|7.24×|
| Eight glyph-shuffles |22.271027|17.497594|1.349873|16.50×|
| All sixteen |23.663430|18.573176|1.542294|15.34×|

Each case uses four original fit records under8equally spaced keys from its
original one-move neighborhood. All512distinct workloads complete under all
three backends,1,536scores total. No slow/null case is removed. Maximum absolute
likelihood discrepancy1.3642421e-12nats, versus frozen1e-7tolerance. Every support,
node count and edge count is identical. All four engineering clauses pass.

Separating Python marginal-only scoring shows that removing Viterbi work helps
modestly; compiled execution provides most of the measured acceleration. The
source probabilities, smoothing, counts, dictionary code, stopping law and
candidate family are unchanged. No probability pruning, approximate inference,
new training, key fitting or transfer/answer access.

Source/protocol`c7c850e28d71c9a88e8485af21230a1a75fdba1d`was pushed/remoteverified
before the sole invocation. Source tables constructed in0.239119seconds;
explicit local native build/load in0.502486seconds. Compiler/build flags,
source/library SHA256and ABI are retained. Total45.359708wall/44.818842hostCPU
seconds; outer46.926062seconds, exit0, peakRSS899,022,848bytes. HostCPU excludes
compiler-child CPU. OneCPU worker, no GPU or paid service; $0.

Independent post-run artifact audit rechecks87frozenbindings, all16compressed
archives, all1,536score records, identical graph/support counts and timing/gate
arithmetic. It does not run an additional inference campaign. Before launch,
full2,162tests plus23subtests passed,10skipped;26native-specific tests exercised
the installed compiler. An additional address/undefined-behavior sanitizer
harness passed1,014normal/error/cap calls with no findings. Its first direct
compiler attempt lacked standard-header search configuration; the installed
working wrapper subsequently compiled it, with both attempt logs retained.

The practical next step is broader fitting search with explicit resource limits.
Sampling-based projections for a complete native one-move bank are3–5seconds
per positive and17–33seconds per shuffle. They do not guarantee timings for
unseen dictionaries, and the prior44-minute incomplete fitting campaign remains
an unchanged failure record. A new experiment namespace/protocol must precede
new fits; faster execution cannot turn a wrong model into a decipherment.
