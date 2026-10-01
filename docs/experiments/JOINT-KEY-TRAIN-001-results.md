# JOINT-KEY-TRAIN-001 — all four fits completed, capacity preference failed

Original registration/freeze9053abf667fb296ef41e7eb5d9b7eedd1da94533. All four original scheduled fits completed20,000updates; no restarts/extensions/additional fits. The single registered complete CPU audit passed. This is an engineering and density-learning result, not a qualified decoder.

| Arm | Parameters | Initial NLL/row | Selected NLL/row | Used rows correct | Whole keys exact |
|---|---:|---:|---:|---:|---:|
| small-72203 | 5,423,146 | 3.900868541 | 3.407782929 | 49/1248 | 0/64 |
| large-72203 | 94,981,674 | 3.879034721 | 3.500445738 | 54/1248 | 0/64 |
| small-72209 | 5,423,146 | 3.883487515 | 3.402742869 | 50/1248 | 0/64 |
| large-72209 | 94,981,674 | 3.890978509 | 3.477961037 | 59/1248 | 0/64 |

Each arm improves proper full23-row key density by at least0.1nats/row from initialization, passing that development gate. Selected snapshot is step20,000 in all four. Both large models have worse NLL than their seed-matched small controls, failing the required0.05nats capacity advantage. Used-row greedy accuracy rises slightly with capacity but stays3.93–4.73%; no arm exactly recovers any of the64complete validation keys. Used masks are diagnostic only. The neural inverse is not yet competent enough to support decipherment-circuit claims.

Each fit sees80,000distinct literal and canonical training dictionaries and1,840,000assigned key-row targets. Matched capacities see the same stream in each seed:23,037,420original source letters for72203 and23,069,848for72209. Four fits total320,000episodes/7,360,000row targets; repeated streams across capacity are not additional independent data. Validation is fixed Pliny development material reused for selection, not new qualification. Training uses actual corpus windows; the separate state-search fixtures are Markov-generated. Do not conflate their failure mechanisms.

Campaign wall32,097.939641seconds (8.916hours), all-child CPU2139.561040seconds; all4returncode0, no outer timeout. Per-arm peak host RSS0.782/1.961/0.781/2.060GB. Original finite GPU training envelope held; sampled Metal/host guards are not total-machine RAM guarantees. No paid API/cloud spend. Bulk checkpoints, complete ledgers/source texts and validation arrays stay outside Git, bound by existing manifests.

The single full audit returned PASS: every episode/source window/key/rejection/seed/exclusion/optimizer ledger, all24checkpoint hashes/configs/finite tensors, selected arithmetic and inventory checked. Initial AND selected checkpoints replay all64full-input true and greedy key log densities with CPU float64; largest discrepancy8.870381e-6nats versus0.002bound, maximum greedy CPUargmax deficit0. Audit290.343525wall/380.224817CPU seconds, peakRSS2.3875584GB, within registered3600/3000/8GiB. Other snapshots have finite/hash/accounting checks, not full density replay. Same-author alternate inference path, no independent agent review or full retraining reproduction.

The exact artifact evidence is in [audit.json](../../results/JOINT-KEY-TRAIN-001/audit.json), [campaign.json](../../results/JOINT-KEY-TRAIN-001/campaign.json) and each arm result. No new training campaign is launched by this completion. Future proposals need competence/whole-key/null controls; activation pictures or token similarity do not substitute for successful recovery. Voynich remains unsolved.
