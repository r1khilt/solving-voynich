# EXP-0030 results — corrected copy-mutate recovery calibration

Registration: [EXP-0030](EXP-0030.md), frozen and verified on `origin/main` at `834f83e43c1c00f6835e592d98831816a2cff448` **before** the single run. Generator correction was separately frozen at `d4fedd1700043b9393f90596830876b2e5dee4b9`. This is exploratory synthetic calibration on a previously reused Finnish source, not a Voynich decipherment.

## Integrity gate: PASS

The separately frozen [source-oracle auditor](../../scripts/audit_exp0030_source_oracle.py) regenerated and exactly matched **all 4,700 persisted rows** (4,000 train, 400 validation, 300 Finnish) and their manifest SHA-256 hashes. It captured the *pre-null* `C(L)` before each channel call, independently of the stored target derived from the mask. Across all **2,951 WORLD_C rows**, it found zero text/mask/tag length mismatches, zero failed full inverses, and zero stored-window targets inconsistent with a prefix of the original `C(L)`. The copy-mutate counts were 770 train, 75 validation and 99 Finnish. Raw corpus digests matched CHANNEL-ALIGNMENT-0001. This validates the synthetic labels in this run; it does not validate the real manuscript model.

Run: `PYTHONPATH=.:src .venv/bin/python -m voynich.copy_mutate_transfer --root . --device cpu --experiment-id EXP-0030 --data-seed 4030 --finnish-seed 4031 --search-seed 4032 --max-wall-seconds 600`. Oracle audit: `PYTHONPATH=.:src .venv/bin/python scripts/audit_exp0030_source_oracle.py --root . --experiment-id EXP-0030`. No paid service or ZL3b score. Python 3.12.13, torch 2.14.0, six CPU threads; training loop 75.24 seconds within the 600-second complete-job cap. Model: 70,301 parameters, 3,000 updates, CTC plus null supervision. Checkpoint SHA-256 `b1fe842a1320a65b5923446910efec2d273ef62dac97bff44a7f6a8825bebda2`; generated split hashes and source hashes are in `data/manifests/exp0030_data.json`.

## Frozen decision: **FAIL**

Validation selected `exact_count_neural` among 20/200 null-feasible programs; its selection score **0.1926** exceeded the mean of 20 random same-length programs **0.1304**. No Finnish score was used to select the winner.

| Finnish holdout condition | Observed | Registered criterion | Result |
| --- | ---: | ---: | --- |
| All-family reconstruction | **0.1884** | > frozen 0.2043 | FAIL |
| All-family matched-random reconstruction | 0.1977 | transparency control | Model below random |
| Null precision / recall | 0.5225 / 0.5239 | each ≥ 0.50 | Pass |
| Predicted null rate | 0.2969 | 0.15–0.45 | Pass |
| Vocab-filter mask F1 | 0.4045 | ≤ 0.55 | Pass |
| Copy-only reconstruction (99 rows) | **0.1665** | > copy matched random + 0.05 = **0.2575** | FAIL |
| Copy-only matched-random reconstruction | 0.2075 | 20-seed control | Model below random |
| Independent pre-null source replay | 0 errors | 0 errors | Pass |

The frozen easy-filler EXP-0013 checkpoint, applied without retraining to this same corrected mixed holdout, reconstructed **0.2020**, also below 0.2043 but above the newly retrained model's 0.1884. This is a control on the corrected split, not a direct comparison with any malformed EXP-0023 score. A separate arithmetic/hash audit confirmed the saved decision matches all registered gates, the result/decision files agree, code/checkpoint/derived SHA-256 values match the manifest, and the source-oracle report passes.

## Exploratory post-result family diagnosis

After the frozen FAIL, a separate [family diagnostic](../../scripts/exp0030_family_diagnostic.py) reran *only inference* on the saved checkpoint and holdout. It reproduced the primary all-family mean exactly as the weighted mean of the three groups. These were **not** additional gates or selection criteria.

| Filler | n | Retrained recon | Frozen easy checkpoint | Matched random | Retrained null precision / recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random character | 94 | 0.1547 | 0.1660 | 0.1941 | 0.376 / 0.387 |
| Periodic | 107 | 0.2382 | 0.2741 | 0.1894 | 0.825 / 0.839 |
| Copy/mutate | 99 | 0.1665 | 0.1583 | 0.2075 | 0.335 / 0.323 |

The mixed average clears the null precision/recall floor because the periodic rows are much easier. Under this setup, retraining on corrected copy/mutate did **not** yield useful copy/mutate deletion: its copy-only reconstruction and null localization are worse than matched-random deletion. It also underperformed the frozen easy checkpoint on the mixed sample. This is one small architecture, one model seed, one adapted Finnish source and a particular synthetic generator; it does not show that copy/mutate removal is impossible, that a larger language/world model would fail, or that the manuscript contains such a channel. Earlier malformed-copy PASS/FAIL claims remain invalid for the intended channel; this new FAIL does not retroactively repair them.

## Artifacts and next state

`results/EXP-0030/{results,decision,source_oracle_audit,family_diagnostic}.json`; `data/manifests/exp0030_data.json`. Weights and full generated rows remain ignored under `outputs/EXP-0030/` and `data/processed/exp0030/`; hashes and regeneration commands are recorded. No decipherment, frozen Voynich decoder or manuscript null-layer evidence follows. The next useful experiment needs a genuinely disjoint language/source and an explicit stronger context/channel model with a true-source oracle and an unrecoverable-copy control; preserve this Finnish result as an exposed calibration set.
