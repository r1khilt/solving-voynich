# Systematic substitution search closes the exposed A recovery gap

2026-09-27. The separately frozen [DEV-002 diagnostic](BLIND-CHANNEL-DEV-002.md)
recovered both simple substitution messages exactly, including all 896 held-out
letters across four records. Independent scalar replay agrees on every evaluated
record. These are reused development cases, with a narrower candidate family;
this is neither fresh confirmation nor recovery of unknown variable units.

| Case | Earlier transfer edits /448 | New transfer edits /448 | New exact transfer records | Earlier search seconds | New search seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| A key 1 | 71 | 0 | 2/2 | 57.699 | 0.645 |
| A key 2 | 188 | 0 | 2/2 | 59.652 | 0.698 |

The selected mappings also decode all 1,792 fitting letters exactly. Every letter
of each ciphertext is accounted for by the unchanged mapping. No answer file or
earlier fitted channel initialized the new fits. The source probabilities,
fitting/transfer inputs, stop law and coding context are unchanged.

## Why this is useful, and what changed

The earlier search had already missed demonstrably better models. This follow-up
shows that a systematic search can actually find good mappings without those
answers. The new optimizer considers all 253 pair swaps in each complete sweep;
precomputed start/bigram counts make those scores independent of record length.
Sixteen starts took less than 0.7 seconds of measured core search per case.

The new optimizer also assumes a deterministic one-state bijection. The earlier
search allowed variable units, multiple alternatives and two states. Both better
coverage and the narrower family changed, so the time/accuracy difference is not
a controlled measurement of speed alone. Pair-local certificates are not global
optimality certificates. A known-key bijection's perfect oracle reading is
automatic; the meaningful observation here is finding the reading without its key.

This does not establish that a first-order source model suffices for unknown-unit
recovery. [DEV-001's B cases](BLIND-CHANNEL-DEV-001-results.md) still failed, and
the new method cannot even represent their variable-length units. Their better
gold-assisted witnesses remain diagnosis, not blind recovery. No Voynich text,
stateful C case, final author, or historical Borg decoding was processed.

## Coverage, controls and independent checks

| Case | Complete sweeps | Neighbors scored | Starts at selected likelihood /16 | Fit total bits | Transfer log likelihood |
| --- | ---: | ---: | ---: | ---: | ---: |
| A key 1 | 361 | 91,333 | 14 | 3356.300044 | -1076.173364 |
| A key 1 shuffle | 287 | 72,611 | 1 | 4102.391250 | -1400.682562 |
| A key 2 | 361 | 91,333 | 15 | 3397.777781 | -1084.274687 |
| A key 2 shuffle | 289 | 73,117 | 1 | 4160.178673 | -1424.816768 |

All 16 starts in every case terminated at pair-local optima; all four selected
keys have complete neighborhood certificates at the registered tolerance. The
remaining positive starts found poorer local optima. Positive endpoint keys are
not all identical: there are eight distinct key-1 endpoint mappings and seven
key-2 mappings. Equal fit likelihood need not resolve unobserved letters or
guarantee equal future accuracy. Only each selected mapping received the registered
recovery evaluation; no endpoint-accuracy claim is inferred from score ties.
The post-evaluation [restart summary](../../results/BLIND-CHANNEL-DEV-002/restart_summary.json)
counts trace events with `stop_reason=pair_local_optimum` and scores within
`1e-8` natural-log units of the selected score, after checking archive hashes.

Shuffles preserve each record's glyph counts and have no plaintext target, so
they receive no accuracy score. Their poorer likelihoods are descriptive here;
no new rejection threshold or claim of general nonlanguage discrimination is
introduced. Each model costs 185 actual code bits in the fixed context (a common
family-selector bit, when included, affects all cases equally).

The evaluator verifies frozen source/input/model/full-trace identities and
replays the selected fit likelihood. It uses the independently authored DEV-001
scalar forward, max-path and edit-distance implementation to cross-check all
24 record evaluations, including literal predictions. Maximum difference is
zero in this run. This is independent numerical replay, not a new independent
audit of every neighbor in the empirical search trace. Before empirical fitting,
32 production/independent tests covered the exact objective, swap deltas, all
tiny permutations, record resets, ties, partial-sweep certificates and local/global
counterexamples. The full repository suite passed 1,417 tests plus 23 subtests,
with eight skips, before these four fits.

## Provenance and resources

The following checkpoints were pushed and remotely verified before the next stage:

- Source/protocol: `8cce2359f352155fe6ad489db9f45aafb1453570`.
- Four selected mappings: `05b561391c821a2aaf263859178128db8ce68930`.
- Unchanged source SHA-256: `7de1c233b3ea69eb632247062f97db86040666daa13ba73bba28bbb51b3493b0`.
- Unchanged data manifest SHA-256: `8587ae6010de94996bdd57d74ea2e1eee3f71940cdb13c1cc3125adfdfc9acea`.
- [Evaluation](../../results/BLIND-CHANNEL-DEV-002/evaluation.json) SHA-256: `706892df3afd4289383ee88320baf83c9957f204d592eb8fd5f7657792631734`.
- Ignored full predictions SHA-256: `42d19fef0c8303bee6ff4839df239a6906b68d3c5ee9b8297f12e8abb372b3bc` (15,193 bytes).

Four sequential core searches used 2.516 seconds in total. Recorded process CPU
through archive creation totals 3.497 seconds; these are different measurement
scopes. No paid compute or API calls. Full traces/predictions remain ignored and
hashed; compact mappings, metrics and provenance are tracked.

```sh
PYTHONPATH=.:src .venv/bin/python scripts/run_blind_channel_dev002.py fit --freeze 8cce2359f352155fe6ad489db9f45aafb1453570
# Freeze, push and verify the selected mappings before evaluation.
PYTHONPATH=.:src .venv/bin/python scripts/run_blind_channel_dev002.py evaluate --freeze 05b561391c821a2aaf263859178128db8ce68930
```

Next: independently validate batched exact marginal scoring for variable-length
units, then separately freeze a bounded full-unit-neighborhood B diagnostic.
That must test actual held-out reading, not merely better fitting likelihood.
