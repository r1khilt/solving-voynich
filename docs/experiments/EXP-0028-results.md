# EXP-0028 result: local one-edit order not supported after stronger spelling control

**Registered decision: `not_supported_under_stronger_controls`.** This adaptive follow-up overturns the practical direction suggested by EXP-0027's weaker base: under the stronger train-selected static spelling prior, adding edited local sources does **not** improve next-word prediction over an exact-repeat cache. It says nothing decisive about historical self-citation, natural language, or manuscript meaning; the model family and exposed validation set remain limited.

## Fixed execution and validation

- Registration/source commit: `docs/experiments/EXP-0028.md`, `0a3030795f66c252f8573fb601f1229118cce423`. Command: `PYTHONPATH=.:src .venv/bin/python -m voynich.word_copy_followup`.
- Same pinned ZL3b train/validation hashes, 65 fit and16 calibration physical leaves; 4,088 calibration words. Validation: 3,700 certain `[a-z']+` EVA words/24 pages/10 physical leaf groups. Final test untouched.
- Train-only selection chose base `beta=1,000,000`, global edit-convolution weight `gamma=0.1`; both local arms used window64, half-life16, lambda0.05. The edited arm used exact-operation probability0.75. All losses are proper prequential probabilities over strings on the training alphabet; corpus-wide edit and local copy densities are normalized.
- CPU wall63.17s, no paid API/download. Full repository pre-score checks: 1,048 passed, 8 skipped, 23 subtests; changed-file Ruff and diff hygiene passed. Independent `scripts/exp0028_audit.py` rechecked source/corpus hashes, exact per-page denominators, all score aggregates, 2,000-draw physical-leaf bootstrap and the registered decision: **pass**. Result SHA-256 `d4c2e4d5740ad9352f9cfc6d4b72a2b6d8e87d2ff1a18c10306289d3e10f929b`.

## Observed losses

| Next-word model | Bits per certain EVA word (lower better) |
| --- | ---: |
| Stronger static word/character + global edit prior | 11.73230 |
| Static + exact local repeat cache | **11.66096** |
| Static + exact/one-edit local cache | 11.66911 |

The registered local edit-over-exact contrast is **−0.00815 bits/word**; leaf-bootstrap95% interval **[−0.01351, 0.00213]**. It misses the predeclared≥+0.03 gate and the interval includes zero. Exact repetition still saves0.07134 bits/word over the stronger static base. The 30% known one-edit insertion positive control strongly favors edit over exact by+3.41372 bits/word, so the model can detect an obvious edited-copy signal under this setup.

The layout-preserving order controls do not rescue the historical interpretation. In100 within-locus permutations, mean edit-over-exact is−0.00682 bits/word (95th percentile−0.00549), versus real−0.00815. In100 within-locus-type permutations, mean is−0.00458 (95th percentile−0.00068). Real word order is **not** unusually favorable to local one-edit prediction under either control. The local exact-cache gain over static is nearly unchanged by within-locus shuffling (real+0.07134; shuffled mean+0.07051, derived from recorded means), indicating that much of the remaining benefit can be carried by the *set of words in a locus* rather than their exact within-locus order. These loci are transcription units and are not always physical lines.

## Interpretation and action

EXP-0027's original base loss12.54725 fell to11.73230 here after stronger train-selected smoothing and global edit-neighbor mass. The apparent local edit gain was sensitive to that baseline choice. This is an example of why local spelling resemblance or cosine proximity alone cannot identify a writing mechanism. Under the tested one-edit channel, **do not prioritize a local edit-copy decoder**. More plausible next questions are what conditions the distribution of words *within a locus* (section, scribe, layout, line position, topic, or latent process) and whether a model predicts those held-out relationships better than a frequency-/form-matched null. An alternative historical copy algorithm could still differ from this simple uniform one-edit transducer; a failure here does not refute the entire hypothesis class.

No plaintext, historical rule, or null-symbol assignment was recovered; the final manuscript test remains unscored. Both results used the already-exposed validation split and cannot be recast as independent confirmation.
