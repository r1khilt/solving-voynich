# TEACH-0012 results: raw serialized relation binding failed

**Registered verdict: `not_qualified`. Independent artifact audit: `pass`.** The positive parsed-memory control passed and the shuffled-label null stayed below its leakage ceilings. The 41,965,568-parameter raw Transformer failed every registered relation-binding gate in both seeds. TEACH-0013's contingent raw-model mechanism experiment therefore cannot enter; its implemented confirmation machinery is prospective only.

## What ran

The frozen training source was `49e2a84bf1b543718d8a6b063b734eda96aa249f`. Two seeds each trained five arms for 8,000 AdamW updates at batch 64: parsed-memory oracle, four-layer raw, repeated four-layer raw, twelve-layer raw, and shuffled-label raw null. The generator drew episode-local symbols from one shared 2,048-symbol pool, globally shuffled 8–16 generic edge rows, varied row formats and boundary markers, and inserted valid distractor chains. The final checkpoints, not selected milestones, supplied all confirmation predictions. Every arm saw the frozen task stream for its seed. The independent CPU audit regenerated the suite and decisions, verified all 10 prediction and loss archives, all 10 final checkpoints and all 10 stored milestones, and reported `pass`.

The MPS campaign finished in **14,104.11 seconds (3 h 55 min 4 s)**. Sampled peak MPS allocation was **686,391,296 bytes** and retained local artifacts occupied **2,477,476,788 bytes**, below the registered six-hour, 24-GiB and 4-GiB ceilings. PyTorch reported version 2.14.0. No paid API, manuscript training or manuscript final-holdout scoring was involved.

## Exact confirmation behavior

All single-item cells below contain 128 items. A factorial group passes only when all four linked F-by-G counterfactuals are right; each arm has 128 such groups. The fully marker-free slice and long-context slice each contain 128 items. The table shows counts rather than rounded percentages.

| Seed | Arm | First read | Direct read | Copy | Both families held out | Factorial items / exact groups | Marker-free | Long with six distractors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | Parsed memory | 128 | 128 | 128 | 127 | 510 / 127 | 128 | 128 |
| 1 | Parsed memory | 128 | 128 | 128 | 128 | 508 / 127 | 128 | 127 |
| 0 | Raw deep | 3 | 1 | 115 | 3 | 4 / 0 | 2 | 1 |
| 1 | Raw deep | 4 | 5 | 127 | 11 | 35 / 0 | 8 | 3 |
| 0 | Raw shallow | 9 | 2 | 128 | 9 | 20 / 0 | 3 | 1 |
| 1 | Raw shallow | 11 | 5 | 128 | 2 | 14 / 0 | 7 | 5 |
| 0 | Raw looped | 0 | 0 | 14 | 0 | 4 / 0 | 1 | 0 |
| 1 | Raw looped | 0 | 1 | 7 | 1 | 0 / 0 | 0 | 0 |
| 0 | Raw null | 12 | 6 | 126 | 1 | 22 / 0 | 2 | 7 |
| 1 | Raw null | 8 | 7 | 128 | 3 | 25 / 0 | 7 | 5 |

Both raw-deep seeds also scored **0/128 exact groups** on each four-query F and four-query G panel. Their train/train composition was only 1/128 and 8/128, so the failure cannot be assigned solely to the held-out family split. The parsed control's 127/128 factorial groups in both seeds show that the logical task, labels, optimizer and two-read answer path can work when relational records are supplied. The raw null's jointly held-out composition was 1/128 and 3/128, with 0/128 exact factorial groups, satisfying the preregistered negative control.

## Interpretation and limits

The registered labels are `RAW-SEQUENCE-BINDING: not_qualified`, `DEPTH-HELPFUL: false`, and `LOOPING-COMPETITIVE: false`. The primary 42M model did not consistently improve on the 14.7M shallow arm; repeating a shared four-layer block was substantially worse. Increasing untied depth or reusing blocks under this training objective did not solve the input-to-relation problem.

The failure occurs before reliable two-hop composition: even first-hop and direct lookup collapse while copy can be near perfect. It does **not** isolate a single cause. The raw models may fail to segment edges, bind a queried left side to its right side, address a stored row, preserve the intermediate key, or combine these operations. Candidate-membership and prediction-change diagnostics indicate some surface sensitivity, but neither proves a correct row assignment. A future study needs explicit, separately measured parser, memory-address and recurrent-read interfaces, including a parsed-row upper bound and interventions at each boundary.

The independent audit checks the saved suite, labels, scores, decisions, source ancestry and archive hashes. It does not rerun neural inference from the checkpoints or prove every optimizer update independently. This is a controlled synthetic result. It neither identifies the Voynich script's language nor supplies a reading of the manuscript.

## Reproduction and retained evidence

- Registered protocol: [`TEACH-0012.md`](TEACH-0012.md).
- Compact score and provenance: [`report.json`](../../results/TEACH-0012/report.json), [`audit.json`](../../results/TEACH-0012/audit.json), [`benchmark.json`](../../results/TEACH-0012/benchmark.json), and [`status.json`](../../results/TEACH-0012/status.json).
- Independent audit command: `PYTHONPATH=.:src .venv/bin/python scripts/teacher0012_analyze.py`.
- The ten compressed prediction archives, ten loss archives and local checkpoint files remain reproducibility evidence at their recorded paths. The checkpoint files and bulk archives are intentionally excluded from the compact Git checkpoint; their SHA-256 values are retained in the report/audit.
