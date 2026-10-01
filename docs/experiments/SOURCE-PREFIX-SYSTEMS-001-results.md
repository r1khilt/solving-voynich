# Cipher-only source-prefix search: correct accounting, inadequate exploration

2026-10-01. Registration was published at
`5387bf51f0c16d40d45aff5b7ccf43d1ebbff550` and verified against the remote
before one invocation. The fixed engineering gate passed. This does **not**
qualify plaintext recovery: six of eight full-source searches found no complete
candidate, and none separated a best reading from its unresolved search mass.

The engine hypothesizes source letters and binds their one/two-glyph emissions
as it goes, enforcing one shared dictionary across both records. It receives
no generation source, key, length, boundary or used-row mask. Unused dictionary
rows integrate out; distinct compatible used dictionaries contribute to a
reading's marginal mass. Search heuristics do not change those probabilities.

## Fixed outcomes

All 288 tiny calls agreed with a separate rational full-key/source enumerator
on the required support, score, incomplete evidence interval and best-reading
checks. All eight large calls used the same selected, unchanged Latin source
with 399,571,824 bytes of dense probability/transition arrays.

| Artificial fixture | Progress bonus | Expanded states | Complete candidates | Stop |
| --- | ---: | ---: | ---: | --- |
| Repetitive, 64 letters per record | 0 | 5,000 | 0 | Expansion cap |
| Repetitive, 64 letters per record | 4 | 1,261 | 512 | Terminal cap |
| Repetitive, 224 letters per record | 0 | 5,000 | 0 | Expansion cap |
| Repetitive, 224 letters per record | 4 | 1,574 | 512 | Terminal cap |
| Markov-sampled, 64 letters per record | 0 | 5,000 | 0 | Expansion cap |
| Markov-sampled, 64 letters per record | 4 | 5,000 | 0 | Expansion cap |
| Markov-sampled, 224 letters per record | 0 | 5,000 | 0 | Expansion cap |
| Markov-sampled, 224 letters per record | 4 | 5,000 | 0 | Expansion cap |

The two calls with complete candidates each returned 512 distinct readings.
Their literal re-encoding, independent scalar source/prior scores and reading
aggregations passed the registered 1e-9 tolerance. No gold-reading accuracy
was computed: these are artificial forced-length workloads, not recovery tests.
All searches were incomplete. All eight reading-separation flags were false.

The conservative priority generated 203,907–218,415 states per call and pruned
194,811–209,319. The progress heuristic greatly reduced branching work on the
Markov fixtures (8,911 and 6,582 generated states, zero queue evictions), but
still reached no complete reading. Therefore absence of completions in those
two calls cannot be attributed to queue eviction alone; the expansion budget
and exploration order remain limitations. This does not isolate which search
change would solve them.

## Validation, resources and reproducibility

One benchmark, session95645 terminal0: 2.988600 stage wall seconds,
2.970559 CPU seconds, 866,975,744 peak resident bytes. One auditor,
session68972 terminal0: 3.034002 wall seconds, 3.025530 CPU seconds,
875,331,584 peak resident bytes. Both stayed within the registered
600-wall/500-absolute-CPU-second/2GiB bounds. Zero paid spend, neural/GPU
inference, additional training, new corpus or historical/held-out panel use.
The statistical source was used for prediction. No retry or cap extension.

The auditor regenerated all fixture RNG/source/key inputs, replayed all 296
calls and work/accounting outputs, checked source and trace hashes, and repeated
the alternate scalar/literal checks. It uses the same large-search algorithm
and author; it is not independent scientific confirmation. Exact tiny rational
enumeration provides an independent algorithm only at toy scale. Floating
bounds are not interval-arithmetic certificates.

Prelaunch full suite: 2,536 tests plus 23 subtests passed, 13 skipped, 151.63s;
56 new focused tests passed. Scoped lint and whitespace checks passed. The five
previous unrelated full-tree lint findings remain unchanged.

Compact records are in `results/SOURCE-PREFIX-SYSTEMS-001/`. The ignored
2,770,291-byte full trace has SHA256
`204915ac76a23d2ad6856196f06cd15cdd46848e4aeeaf1c60f44d0ef874319e`;
the 50,289-byte fixture file has SHA256
`9e9198f19a748086f09a4ab7e129b1907bc5dadc067a52b071eb478a8effb905`.
The result's 7,372 bytes have SHA256
`7a2137229e5da30d11a371a2a36c2efd33ef3a80a01201297c49f58a6e7c8e8b`.

## Implication and next question

We now have a tested probability-accounting core that can search directly for
readings without guessing never-observed key entries. We do not yet have a
practical reader. The current schedule completes the first record before using
the second record's detailed constraints. A deterministic interleaved schedule
could let both records reject bad shared-key choices sooner while preserving
each reading/key leaf's probability and avoiding duplicate action orderings.
That is a working hypothesis requiring exact enumeration and a fresh bounded
comparison, not a demonstrated remedy. Raising the budget also needs measurement;
this very small workload used only about three seconds, not hours of search.

The separate four-fit neural campaign continues unchanged. Its final training
audit and fresh unknown-key qualification remain pending. No Voynich reading or
historical generator has been identified.
