# EXP-0031 result — synthetic cipher glyph names change the frozen model's decisions

**Exploratory diagnostic: valid, material symmetry violation; group averaging did not clear its benefit rule.** No Voynich decoding, hidden plaintext, language identification or fresh-language transfer was tested. The Finnish source was already exposed in prior experiments.

## What was changed and why it is a control

The corrected EXP-0030 WORLD_C generator chooses an arbitrary ordered cipher alphabet from a 90-character pool. Replacing every glyph consistently by a fresh bijection of that full pool leaves the synthetic data distribution and gold keep mask unchanged; the pre-result coupled null-generator tests verify the operation for each selected family. A post-result check coupled the **full plaintext→cipher→null pipeline** under common random seeds for 20 seeds per family and reproduced transformed text, pre-null target and mask exactly in 60/60 cases. Spaces, symbol-equality patterns, token recurrence, positions and copy-aware feature arrays are unchanged. The frozen neural decoder does receive learned raw-glyph embeddings, so its response can change. This is a causal **input renaming** test of that architecture, not an identification of a particular neuron or a claim that real Voynich glyphs are interchangeable. See the [identifiability note](../research/SYMBOL_RENAMING_IDENTIFIABILITY.md) and [pre-result registration](EXP-0031.md).

Source freeze `6fc567606916de271ba22acb9ea93903d5a8cd3b` was pushed and independently verified on `origin/main` before the first intervention run. The runner used the unchanged EXP-0030 model (`b1fe842a1320a65b5923446910efec2d273ef62dac97bff44a7f6a8825bebda2`) and corrected Finnish JSONL (`138f158484ff5e3859fad113f96910d61be224dc44d6b9ab16084d0edce014d7`), 300 rows, 31 seeded full-pool renamings per row plus identity, seed `310031 + 1000×row_index + replicate`. Python 3.12.13, PyTorch 2.14.0, six CPU threads, no gradients/training, 3.14 seconds observed versus 300-second cap, no paid service or download. Exact command:

```sh
PYTHONPATH=.:src .venv/bin/python -m voynich.symbol_renaming_assay --root .
PYTHONPATH=.:src .venv/bin/python scripts/audit_exp0031.py --root .
```

## Observations

Identity re-inference differed by **0.0** at maximum probability error, and copy-aware feature arrays were exactly invariant for all 9,300 nonidentity windows. On the 99 copy-mutate rows, **24.71%** of exact-count keep/delete decisions differed from the original after a random glyph renaming, averaged by row and 31 renamings. The registered row-bootstrap 95% interval was **[23.25%, 26.15%]** (2,000 draws, seed 310131), well above the registered material-violation thresholds (>5% mean and >2% lower bound). This measures 392,832 position-level comparisons, with uncertainty resampled at the row level. The three families behaved differently:

| Null family | Rows | Mean keep-mask flip | Mean absolute keep-probability drift | Original reconstruction | 32-way averaged reconstruction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random character | 94 | 23.29% | 0.1169 | 0.1547 | 0.1661 |
| Periodic | 107 | 10.39% | 0.1237 | 0.2382 | 0.2441 |
| Copy-mutate | 99 | 24.71% | 0.1528 | 0.1665 | 0.1735 |
| All | 300 | — | — | 0.1884 | 0.1964 |

The original mixed and copy reconstruction scores exactly reproduce EXP-0030's **0.18838171635900422** and **0.16651924193353831**. Group-averaging probability vectors lifted copy reconstruction by **0.0070** to **0.1735**, short of its registered >0.02-gain condition and still below the frozen copy-only matched-random **0.2075**. Mixed reconstruction rose to **0.1964**, also below its local matched-random **0.1977**. Original copy null precision/recall **0.3347/0.3231** became **0.3503/0.3382**. Thus the score did not become useful through this parameter-free symmetry repair.

The glyph embedding changed by mean relative L2 **1.23** and the BiLSTM hidden state by **0.62** on copy rows. Their numerical sizes depend on layer scaling and do not prove that an individual dimension or neuron implements the failed decision. The controlled output flips are the behavioral evidence.

## Audit, interpretation and next state

The independent auditor passed. It verified checkpoint/holdout/runner hashes, decoded all 600 compact original/averaged masks, independently recomputed every row's reconstruction and null counts, checked family/mixed aggregation and the bootstrap/decision rules, and directly replayed 20 frozen raw model probability vectors. Maximum direct-replay absolute error was **4.77×10⁻⁷**. `results.json` SHA-256 `cd2fc6a6b8c7957484bdc5e06313f05c54b39b3b121053dbd28aeb00ff419c83`; `audit.json` SHA-256 `f9a6ec4682c7ea843faa1ba683ae89885bb13a11a3483fa96c755375154b5358`. Source/auditor SHA-256 values are embedded in those reports. Focused coupled-generator tests passed 13/13; full repository suite passed 1,121 tests, eight skipped and 23 subtests; Ruff, Python compilation and diff hygiene passed.

The result **does** show a mismatch between the synthetic problem's arbitrary-glyph symmetry and the trained small decoder's behavior. It **does not** establish that the mismatch alone caused the EXP-0030 failure: even explicit averaging did not beat the copy matched-random baseline, and copy-mask recovery may be limited by context, the noisy channel, objective, available data or intrinsic ambiguity. Nor does it license scaling this exact model, treating a synthetic success as decipherment, or imposing full glyph-exchangeability on the manuscript. The next meaningful model comparison is an invariant/canonicalized encoder against a capacity-matched raw-embedding control, trained on corrected data and evaluated on a genuinely disjoint language/source with identifiable positive and iid-unrecoverable controls. That comparison needs a new registration and source freeze; none was run under EXP-0031.
