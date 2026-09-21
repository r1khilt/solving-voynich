# EXP-0022 results — Language-ID / simple-substitution with null controls (R5)

**Decision: FAIL** mode `hebrew_latin_null_preferred` (method invalid for language ID).

Registered before scores in `docs/experiments/EXP-0022.md`. One scored run. Thresholds not retuned. **Not a decipherment.**

## Command

```sh
.venv/bin/python -m voynich.language_id_attack --root .
```

Elapsed ≈ **119.5 s** local CPU. Seeds / letter budget as registered (`4022`/`4023`/`4024`; 12,000 letters; margin 0.020 bits/char).

## Preferred languages (mono, margin ≥ 0.020)

| Corpus | Preferred | Top-3 mono bits/char |
| --- | --- | --- |
| `true_eva` | **eng** | eng 3.585 · tur 3.630 · heb 3.648 |
| `scrambled_glyph` | **heb** | heb 3.427 · hun 3.470 · arb 3.529 |
| `section_shuffled` | *(none)* | heb 3.623 · gle 3.627 · eus 3.660 (margin 0.004 < 0.020) |
| `zipf_length_null` | **heb** | heb 4.237 · arb 4.288 · hin 4.793 |

## Gate outcomes

1. Panel files present; letter budget met; ZL3b test not opened for scoring.
2. **Hebrew/Latin null gate:** `heb` preferred on `zipf_length_null` → **FAIL** `hebrew_latin_null_preferred`. EVA rankings are **not** interpreted as language identification.
3. Secondary (non-deciding under invalid method): `eng` preferred on true EVA is **not** preferred on scrambled (heb) or section-shuffled (empty). That pattern is consistent with artifact risk but is superseded by the null-gate failure.

## Interpretation

Under the frozen monoalphabetic attack, Hebrew wins on length- and unigram-matched null text. The attack family therefore cannot be used to claim a Voynich language ID (matches the Hauer–Kondrak artifact warning spirit without replaying anagram+abjad Hebrew ranking). Simple language-ID / simple-substitution claims from this attack class are **rejected as untrustworthy** on this panel and ciphertext budget.

**Not a decipherment.** No frozen reading. No held-out leaf semantic test. HYP-004/HYP-005 remain unresolved as historical explanations.

## Artifacts

- `results/EXP-0022/results.json`, `decision.json`
- `data/manifests/exp0022_data.json`
- Code: `src/voynich/language_id_attack.py`; tests: `tests/test_language_id_attack.py`
