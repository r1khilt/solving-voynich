# CHANNEL-ALIGNMENT-0001 — historical synthetic copy-mutate alignment erratum

**Status:** confirmed code/data defect; correction implemented prospectively. This is a deterministic audit of an existing benchmark, not a new Voynich result or a retroactive change to any registered threshold.

## Defect and consequence

In the historical `insert_nulls` implementation, the `copy_mutate` filler sometimes called `_mutate_token` on a single character and received a **two-character** string. It appended that whole string to the character stream but appended only **one** mask bit and family tag. Joining the stream produced more characters than mask positions. `_fit_len` then silently truncated or padded the mask to 128 positions. Consequently, for affected rows, the stored mask no longer marks the actual inserted character positions, and `sample["ciphered"]`—computed by applying that mask to the visible text—is not the pre-insertion `C(L)` that the experiment intended to recover. The nominal filler rate also no longer describes the actual visible character fraction. The later oracle deleted positions from the **misaligned stored mask** and was compared against the target created by that same mask; its 1.0 reconstruction is therefore not evidence that the true copy-mutate channel was inverted.

## Exact replay and audit

Audit source: `480930b:src/voynich/latent_recovery.py`, SHA-256 `07f39846ae1daad63019cb3baae0ccff5a99be05220c442d5913c0543bb35e50`. Python 3.12.13, NumPy 2.5.3. Recleaned the three local raw texts with the historical `clean_plaintext` and verified byte equality to their saved `.clean.txt` files. Regenerated EXP-0023 using data seed 4011, Finnish seed 4012, filler rate 0.30, families `random_char,periodic,copy_mutate`, 4,000 train / 400 validation / 300 Finnish rows. A wrapper around the **unchanged historical** `insert_nulls` recorded its input `C(L)` and output without consuming additional random numbers. Every regenerated saved field matched every persisted JSONL row: 4,000/4,000, 400/400, 300/300. The persisted files' SHA-256 hashes matched `data/manifests/exp0023_data.json`: `2c824915…`, `7d572869…`, `cd72bf0f…` respectively.

For each WORLD_C row, checked whether `len(noisy) == len(mask)` before `_fit_len`, and whether the stored target with trailing padding spaces removed is a prefix of the actual pre-insertion `C(L)`. This prefix condition is necessary because the visible window is truncated to 128 characters; padding spaces are excluded. Results:

| Split | Copy rows: raw length mismatch | Copy rows: scored target not source prefix | Random/periodic: either failure |
| --- | ---: | ---: | ---: |
| Train | 842/842 | 842/842 | 0/1,611 |
| Validation | 79/79 | 79/79 | 0/163 |
| Finnish holdout | 100/100 | 100/100 | 0/200 |

Mean extra unlabelled visible characters per copy row were 19.12, 18.86 and 18.28 in those three splits. A separate 100-seed, single-family invariant check found 100/100 copy failures and 0/100 each for random and periodic. These counts establish corruption **on this exactly replayed split**; they are not a count for every historical run.

## Scope of earlier claims

- **Affected by the same generator path:** EXP-0011's full-family benchmark, EXP-0014b, EXP-0016b, and EXP-0023. Their numerical reports remain historical observations on their stored, malformed synthetic data. Their PASS/FAIL labels cannot be used as evidence about recovery of the intended copy-mutate `C(L)` channel. EXP-0023 was exactly replayed and counted above; the other IDs were identified by their recorded filler families and code path, not individually replayed here.
- **Derived from the malformed EXP-0023 split:** EXP-0024's oracle result and EXP-0026's search-missed-inverse diagnosis. The reported 1.0 oracle score is a stored-mask consistency check, not a true-channel inverse. The claims that the typed decoder failed to find a recoverable inverse must be retested on corrected data.
- **Not implicated by this specific bug:** easy-only EXP-0012, EXP-0013, EXP-0014, EXP-0016 primary and EXP-0017. Their other limitations and historical/Voynich transfer failures remain. This audit does not establish any manuscript null layer or decipherment.

## Correction and next state

`insert_nulls` now emits and labels each actual character, including both characters from a copy mutation, while respecting the requested *character* null budget. It asserts equal final text/mask/tag lengths. `_fit_len` now refuses a pre-existing length mismatch rather than hiding it. Tests cover all six filler families, three rates, ten seeds each, preservation of the source subsequence, exact null counts, and the mismatch refusal. A corrected full-size replay with the same seeds and sample counts had zero length, full-inverse or window-target failures across all WORLD_C rows (including 813/73/90 copy rows in train/validation/Finnish); the copy counts differ because changed emissions alter later RNG draws. This changes the generated distribution and random-number trajectory; do **not** overwrite or directly compare old experiment IDs as though only a metric changed. A fresh, prospectively registered corrected-data training/holdout study is needed before renewing any copy-mutate recovery claim.
