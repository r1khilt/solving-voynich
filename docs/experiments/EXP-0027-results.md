# EXP-0027 result: one-edit word cache on ZL3b validation

**Registered decision: `support_copy_channel_research` (exploratory only).** This is evidence that a one-edit cache improves this particular held-out-word predictor. It is **not** evidence that the historical author copied words, that the text lacks meaning, or that any Voynich word has been read. The ZL3b validation leaves had already been used by other project experiments; final test remains unscored.

## Execution and checks

- Registration and exact source: `docs/experiments/EXP-0027.md`, source commit `058d0f4399c6c707780ce63d269bbb60096cbc5e`.
- Command: `PYTHONPATH=.:src .venv/bin/python -m voynich.word_copy_channel`.
- Pinned derived train/validation SHAs: `49618c7be69cef573fe9ad8ae3627ad9f7b495601af1897aafb6082837fceca4` / `9bee4149f26fc49b49e4a826b73598dfb79a1e785731c15a1763489315a2efd3`; the runner refused drift. It did not read test.
- 4,088 scored training-calibration words on 16 leaves; 3,700 certain validation words on 24 pages / 10 physical leaf groups. Other words, including uncertain transcription events, were excluded from both targets and copy sources. All locus types pooled.
- Train-selected base smoothing `beta=1000` (the largest registered candidate); exact cache: window64, half-life16, lambda0.05; edit cache: window64, half-life16, lambda0.1, exact-operation probability0.75. Selection used train calibration only.
- Runner wall time32.55s CPU, no paid API; full repository checks before source freeze: 1,046 passed, 8 skipped, 23 subtests; changed-file Ruff and diff hygiene passed. Separate `scripts/exp0027_audit.py` recomputed the page/leaf denominators, per-word aggregates, 2,000-draw leaf bootstrap and decision from the compact result; audit **pass**. Result SHA-256 `7e8aa2f52fce84760a4ac47e614e8b989cf44c66cc059c3ace8f0ee5b49eb83f`.

## Observations

| Proper next-word model | Bits per certain EVA word (lower better) |
| --- | ---: |
| Train word-frequency + character-4gram backoff | 12.54725 |
| Base + exact recent-word cache | 12.47886 |
| Base + exact/one-edit recent-word cache | **12.40205** |

Edit over exact saves **0.07681 bits/word**; paired physical-leaf bootstrap 95% interval **[0.05462, 0.11107]**. Nine of ten validation leaf groups have positive edit-over-exact differences. Edit over base saves0.14520 bits/word. On100 within-page word-order permutations, the 95th percentile of edit-over-base gain is0.13054, below the real0.14520; the known one-edit insertion control gives edit-over-exact +7.31115 bits/word. All four preregistered gate components pass.

The *edit-specific order* evidence is weaker: the real edit-over-exact gain0.07681 is **below** the 95th percentile0.08073 of that same contrast on the page permutations. This is not a separate registered failure criterion, but it prevents interpreting the aggregate pass as proof that nearby edited pairs occur in a special sequence. Page-vocabulary composition and the manuscript's highly constrained word shapes explain much of the edit benefit. The positive control is deliberately artificial and creates very rare words under the base model; its large gain is an implementation/sensitivity check, not a model-to-manuscript match.

## Limits and next discriminating step

`beta=1000` landed on the top of the calibration grid, so the base may be under-smoothed; a stronger spelling prior or other baseline could absorb part of the edit gain. The within-page permutation moves eligible words across locus types and line positions, possibly mixing layout-specific vocabularies. A registered successor should freeze the current result, extend base strength using train-only selection, and compare within-locus and locus-type-preserving reorderings. A historically meaningful claim would further require ordinary-language, cipher, and nonsemantic controls matched in manuscript size and layout, then an explicit constrained decoder with independent external predictions.
