# SOURCE-PRUNE-DIAG-002 — three failures discard truth in the first six letters

Registration7efe79367b6a8c72ef9d62ca5932afb4981bd207 was pushed and exactremote verified before ONE sixteen-cell run. ONE complete replay audit passed. Original001 admission failure remains preserved separately; it had zero experimental calls.

The successful wide case preserves the literal true path through all128source letters and returns its correct used-key group at rank1. The other three wide cases discard the literal true path after6,6,5source letters, respectively, across the two records combined. None of their ALL visited terminal groups agrees with every true used mapping; the512output limit is not the cause of these failures.

| Case | Arm | Source letters consumed at first loss | Total source letters | Loss stage | Geometric rank | Final rank | Compatible used-key group rank |
|---|---|---:|---:|---|---:|---:|---:|
| 0 | unmerged | 3 | 128 | final_beam | 446 | 446 | absent |
| 0 | merged | 3 | 128 | final_beam | 446 | 446 | absent |
| 0 | guided | 2 | 128 | final_beam | 17 | 136 | absent |
| 0 | wide_guided | survived | 128 | none | — | — | 1 |
| 1 | unmerged | 3 | 128 | final_beam | 244 | 244 | absent |
| 1 | merged | 3 | 128 | final_beam | 244 | 244 | absent |
| 1 | guided | 6 | 128 | geometric_prebeam | 1294 | — | absent |
| 1 | wide_guided | 6 | 128 | final_beam | 8794 | 9071 | absent |
| 2 | unmerged | 3 | 448 | final_beam | 474 | 474 | absent |
| 2 | merged | 3 | 448 | final_beam | 474 | 474 | absent |
| 2 | guided | 1 | 448 | final_beam | 11 | 295 | absent |
| 2 | wide_guided | 6 | 448 | final_beam | 6964 | 6561 | absent |
| 3 | unmerged | 2 | 448 | final_beam | 163 | 163 | absent |
| 3 | merged | 2 | 448 | final_beam | 163 | 163 | absent |
| 3 | guided | 2 | 448 | final_beam | 81 | 148 | absent |
| 3 | wide_guided | 5 | 448 | final_beam | 1035 | 10957 | absent |

At width4096, failed cases1/2/3 pass preliminary width16384 but rank9071/6561/10957 after iid guidance. Their geometric ranks are8794/6964/1035. Case3 is particularly diagnostic: it would pass a4096geometric cutoff on that same incoming frontier, then guidance demotes it beyond the final cutoff. This is a conditional ordering fact, not an experiment proving that an unguided4096search would recover it: changing earlier pruning changes the frontier.

Narrow guided case1 is removed by preliminary geometric rank1294>512. The other narrow guided cases survive preliminary pruning but fail final width128, ranks136/295/148, after2/1/2letters. Every unmerged/merged unguided narrow path dies after2–3letters. Fifteen of sixteen literal trajectories die; only case0-wide_guided survives. The first losses each have one structural alias and aggregate state mass equal to raw literal prefix mass; merging has supplied no extra gold-state mass at those early loss points. All post-loss gold-state alias counts in these realistic runs are zero. Tiny controls separately cover alias reappearance, so this is an observed absence rather than assuming aliases impossible.

Raw-prefix ghost rankings after loss are conditional on the actual already-pruned frontier. They are not counterfactual search ranks. A rank9071at this frontier does not establish a universal9071beam-width requirement or predict eventual recovery from wider search. No proof that the source is false, Latin is the historical language, gold is globally optimal, or all other coherent search methods fail follows from these results.

The next method should specifically test stronger future shared-key/source constraints or controlled hypothesis diversity in the first dozen source decisions. Enlarging the returned-key list cannot recover branches already eliminated. Merely increasing a weak neural inverse is also unsupported: the separately completed95Mmodels failed their capacity preference gate. [Prospective next design](../research/after-pruning-2026-10-01.md) distinguishes cheaper constraint lookahead from a learned critic and gives competence requirements before neural mechanistic analysis.

All16ordinary returned JSON objects—including complete layer traces, masses, terminal lists, counters and original guide/cache work—exactly equal the old frozen SOURCE-STATE-SYSTEMS-001 outputs. The single complete observer replay matches all new traces. Independent rescheduling, per-prefix source/EOS/once-binding-prior reconstruction, direct geometric sums and vectorized iid-guide arithmetic cover12,322checks; maximumdelta9.09494701772928e-13nats versus1e-7. These are same-author realistic search replays with alternate arithmetic/tiny-law evidence, not independent-agent or independent full-search certification.

Run115.542201209wall/115.137738CPU/880066560RSSbytes; audit114.784763792wall/114.664374CPU/882245632RSSbytes. Bulk513035bytes, no paid use/GPU/new training/holdout. All1800/1600/2GiB/128MiBcaps held. Expected2–10minutes/stage conservative; actual1.93/1.91minutes. Final preregistration2662tests+23subtestsPASS/13skip153.79s, scopedlintPASS/same5oldfindings. Original130frozen bindings unchanged. Post-outcome read-only publication checker validates all closed hashes/values/monotone survival/source-step counts/resource arithmetic and freezes without new search/scoring/model calls.

Known-answer exposed synthetic diagnosis, no new recovery accuracy or causal neural circuit claim. Voynich remains unsolved. Exact evidence: [result.json](../../results/SOURCE-PRUNE-DIAG-002/result.json), [audit.json](../../results/SOURCE-PRUNE-DIAG-002/audit.json), [publication-audit.json](../../results/SOURCE-PRUNE-DIAG-002/publication-audit.json).
