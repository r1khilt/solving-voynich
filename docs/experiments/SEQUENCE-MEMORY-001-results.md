# SEQUENCE-MEMORY-001: fixed-shape engineering gate passes

All three artificial-input arms completed both untrained full-size models. This
qualifies the fixed8 implementation for a separately registered attempt under
the original memory limit; it does not establish neural decoding accuracy.

Source `852e355b9725045628ff1a1fafaa5dd3fe1895ce` was published and remote-verified
before the single sequential campaign. Each arm scored 8,192 records /1,843,110
letters from exactly matching model initializations and input strings. All96
sampled CPU-float64 reference comparisons passed; maximum delta2.565e-7nats.
Comparing every saved score across arms gives maximum delta3.411e-13nats.

| Arm | Sampled driver peak, bytes | Peak host RSS, bytes | Scoring seconds | Letters/second |
| --- | ---: | ---: | ---: | ---: |
| Variable64 | 2,003,582,976 | 534,265,856 | 7.8380 | 235,152 |
| Variable8 | 1,312,014,336 | 544,636,928 | 22.1893 | 83,063 |
| Fixed8×270 | 1,259,061,248 | 528,318,464 | 22.6943 | 81,215 |

Live tensor allocation peaked at29,620,480bytes in each sampled trace, showing
why it cannot substitute for total driver allocation. Fixed8 uses about37.16%
less sampled driver memory than variable64 and4.04%less than variable8, at about
one third variable64 throughput. Smaller batching has the larger observed effect.
No causal attribution to a particular allocator or cache mechanism is established.

The artificial variable64 workload also passed; **it did not reproduce the real
candidate failure**. Its shapes/data differ from the candidate workload. Do not
claim the earlier failure was spurious or that success here guarantees the real
run. Fixed8 is used prospectively for its unchanged shape and lower measured
driver allocation, as registered before the benchmark.

Complete campaign67.343081wall seconds; summed measured worker work56.553128wall /
12.722130hostCPU seconds; zero paid use. Model numerical checks and startup explain
the difference from scoring-only times. The fixed8 scoring rate projects about
456seconds for37,032,100candidate letters, before preparation/audits and without
a trained-model or thermal performance guarantee.

Independent postrun accounting checked all source bindings, process codes, data
and initial-weight identities, trace/score hashes, every batch-progress index and
fixed call shape, memory maxima, finite scores, and all cross-arm score deltas.
An initial accounting invocation rejected relative paths before saving any result;
using absolute repository paths fixed that accounting error, with no benchmark
rerun. Compact records and `audit.json` are in `results/SEQUENCE-MEMORY-001/`;
complete traces and score arrays remain ignored under checksummed output paths.

Next: KEY-BANK-NEURAL-002 applies fixed8×270 to all original candidate readings
and both unchanged trained models. NEURAL-001 remains an engineering failure;
CONFIRM-002's fresh pipeline remains unchanged.
