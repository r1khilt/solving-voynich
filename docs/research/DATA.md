# Corpus and preprocessing contract

Status: acquired and prepared on 2026-09-21 UTC. This is infrastructure and data validation, not a decipherment experiment or a training result.

## Source choice and rights

The first corpus is **Zandbergen–Landini ZL version 3b**, dated 13 May 2025, downloaded directly from [the maintainer's file](https://www.voynich.nu/data/ZL3b-n.txt). ZL is the Zandbergen component of the earlier LZ project; its name acknowledges Landini. It is one stream, not stacked parallel transcriptions. The maintainer's [transliteration index](https://www.voynich.nu/transcr.html) lists 5,385 loci and records uncertainty, rare EVA forms, and drawing interruptions. We selected it over an interlinear merge to preserve those distinctions without duplicating observations.

The maintainer [expressly supplies the collected transliterations under CC0](https://www.voynich.nu/roadmap.html). This statement does not license the site's fonts or manuscript photographs. Raw and derived text remain in ignored directories; tracked manifests carry provenance, source hashes, frozen assignments, and compact statistics. No image corpus or font was downloaded.

Source SHA-256: `bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc`.

Primary format reference: [IVTFF specification, document 2.0.2, July 2025](https://voynich.nu/software/ivtt/IVTFF_format.pdf), especially §§4.2 and 6.1–6.9. The file itself declares IVTFF 2.0. It separates page headers, loci, annotations, and transliterated text. EVA represents written forms; a Latin EVA character is not necessarily one manuscript glyph or a semantic unit. Period/comma distinguish certain/uncertain spaces; brackets carry alternatives; comments can contain ordinary editorial language. These distinctions must survive preprocessing without teaching the model the editor's prose.

## Reproduce

From the repository root, with Python 3.10+ and curl:

```sh
python3 scripts/download_data.py
PYTHONPATH=src python3 -m voynich.data
PYTHONPATH=src python3 -m unittest discover -s tests -p test_data.py -v
```

The downloader verifies an existing cached file or downloads exactly the pinned source. A checksum mismatch stops processing. The parser uses only the Python standard library. Rebuilding preserves the checked-in split; changing it requires a deliberately new manifest version/directory. The first retrieval needed the environment's network escalation after sandbox DNS resolution failed; the approved retry succeeded.

## Our representation decisions

These are modeling choices, not claims about the historical writing system:

| Input feature | Model-facing text | Preserved evidence |
| --- | --- | --- |
| Ordinary EVA letters, case, apostrophe | Unchanged codepoint | Raw locus |
| Definite word space `.` | ASCII space | Raw locus |
| Uncertain space `,` | One `U+E003` event | Raw annotation and offsets |
| Alternative `[a:o]` | One `U+E000` event | All candidate readings, including empty or unreadable options |
| Unreadable `?` / `??` | One / two `U+E001` events | Original marks |
| Unreadable span `???` | One `U+E002` event | Unknown span length remains unknown |
| Rare EVA `@NNN;` | One codepoint `U+E100 + NNN` | Original numeric EVA code |
| Ligature `{...}` | Unwrapped components | Ligature span and original notation |
| `<->` / `<~>` drawing interruption | Space | Interruption type and offset |
| Paragraph markers, inline prose, text tags | No text emitted | Typed annotations and inherited text tags |
| Locus boundary | Newline | Locus ID/type/locator and page offset |
| File/page headers, full-line comments | No text emitted | Raw source and structured page variables |

No `<...>` markup, page labels, English commentary, Latin plant guesses, or editorial identifiers enter model-facing text. The parser rejects malformed/unsupported records instead of filtering arbitrary characters until they look plausible. It supports the pinned EVA subset and explicit single-stream selection, not every historical IVTFF dialect.

`Lx` extraneous-writing loci and `!` invalid locators are excluded. In this source the excluded `Lx` loci are `f17r.13`, `f66r.82`, and `f116v.1`; removing the last makes `f116v` empty. This intentionally excludes a few Voynich-looking marginal tokens as well as an associated non-Voynich editorial reading. Exclusion is recorded and reversible from the raw source.

The resulting corpus has **226 nonempty pages and 5,382 loci**, from 227 source page records. A newline separates *loci*, which are not always physical lines. Circular text, labels, and titles follow the source's order; this is not a proven reading order. Paragraph boundaries and drawing geometry remain available in metadata for later controlled representations.

## Frozen splits and contamination checks

`data/manifests/zl3b_split.json` freezes **section-aware version 2** before any model fitting. Recto, verso, and all foldout panels share a leaf. The connected Rosettes sheet combines `f85`, `f86`, and `fRos` in one group. The seed is `voynich-section-leaf-split-20260921-v2`; group ordering comes from SHA-256 rather than a platform-specific RNG.

Exact full-page duplicates and any shared contiguous span of 128 projected EVA units join groups before assignment. The projection removes whitespace and uncertainty events, conservatively detecting duplicates across spacing/uncertainty differences. The nominal 80/10/10 allocation is by independent groups, not token counts. Variable page lengths produce unequal token fractions.

Multi-label stratification uses only the source's illustration-type metadata. A deterministic bounded search reserves one group in each split for every section supported by at least three independent groups; a deficit-balancing completion fills the exact group quotas of 81/10/10. Whole mixed-section leaves remain indivisible. Astronomy has only two independent leaf groups (`f67`, `f68`), making three-way coverage impossible; the declared rare-section policy allocates them to train and validation. Single-group strata, if encountered, remain in training. An infeasible set of coverage requirements stops with an error instead of silently claiming stratification.

All eight illustration types occur in training and validation. Seven occur in test, which has no independent astronomical (`I=A`) group. Other grouping and approximate-copy limitations remain. This is still an initial within-manuscript evaluation, not a held-out-section, quire, scribe, or independent-manuscript test.

The earlier uniform assignment was an **uncommitted provisional design** and was replaced before any training or model evaluation because metadata inspection showed entire sections missing from validation. Its exact assignment remains in `zl3b_split_provisional_v1.json`, explicitly marked superseded. This was a pre-experiment design correction; no prediction outcomes were available or used to select the replacement.

| Split | Pages | Leaf groups | Loci | Tokens including page BOS/EOS |
| --- | ---: | ---: | ---: | ---: |
| Train | 177 | 81 | 4,153 | 184,930 |
| Validation | 24 | 10 | 689 | 26,371 |
| Test | 25 | 10 | 540 | 22,342 |

The audit found zero cross-split leaf, exact-page, or 128-unit span conflicts. Forty-nine short normalized locus types recur across splits; the longest is eight units. They are retained as ordinary repeated labels/words rather than erased. Approximate copying, neighboring-leaf dependence, shared hands, and subject matter can still connect splits. Split metadata includes Currier/illustration distributions; these are not model inputs.

Only training texts fit the tokenizer vocabulary. The tokenizer has 112 entries including fixed controls. Validation and test contain 6 and 5 out-of-vocabulary codepoints respectively; these map to `<unk>`. Unknown/alternative/unreadable/uncertain-space events are visible as explicit context, but `tokenizer.uncertainty_ids` must be masked from the primary prediction loss and metric. The preparation manifest reports these counts (train 2,885; validation 430; test 360, including out-of-vocabulary units). Collapsing alternatives changes sequence length and throws away candidate identities in this particular model view; their full evidence remains available for later sensitivity analyses.

Test statistics above are corpus-integrity counts only. Do not evaluate a model against the test text during architecture tuning. Vocabulary construction, normalization rules, and split selection must never be tuned to test loss.

## File and Python interfaces

Tracked records:

- `data/manifests/zl3b_source.json`: origin/version, retrieval timestamp, rights, bytes, source checksum.
- `data/manifests/zl3b_split.json`: immutable leaf assignments, seed, duplicate policy.
- `data/manifests/zl3b_preparation.json`: parser exclusions, counts, audit, derived checksums.

Ignored generated files are `data/processed/zl3b/train.jsonl`, `validation.jsonl`, `test.jsonl`, and `tokenizer.json`. Each JSONL row has `page_id`, `leaf_id`, `split`, `text`, `metadata`, and `loci`. Each locus preserves its raw string, normalized string, source line, source locator, annotations, inherited tags, and normalized page offset. **Only `text` is eligible for tokenizer input.** Load complete records separately when interpreting a model prediction.

```python
from voynich.data import load_pages
from voynich.tokenizer import EVATokenizer

pages = load_pages("data/processed/zl3b/train.jsonl")
tokenizer = EVATokenizer.load("data/processed/zl3b/tokenizer.json")
ids = tokenizer.encode(pages[0]["text"], add_bos=True, add_eos=True)
ignored_targets = tokenizer.uncertainty_ids | {tokenizer.pad_id}
```

`encode` returns integer IDs; `decode` reconstructs the normalized view; `vocab_size`, `pad_id`, `unk_id`, `bos_id`, `eos_id`, `space_id`, and `line_id` are available. Training/evaluation windows must stay inside a page and a split. Tokens in overlapping evaluation context windows must not be counted repeatedly as prediction targets.

## Validation record

Sixteen focused tests cover malformed input, editorial exclusion, ambiguity (including the source's empty alternative), rare escapes, continuation lines, single-stream selection, tag scope, Rosettes grouping, split determinism, duplicate grouping, feasible multi-section coverage, rare strata, impossible-coverage rejection, contamination rejection, train-only vocabulary, checksum pinning, split freezing, and identical rebuilds. Every generated text codepoint is audited against the permitted normalized alphabet; no editorial syntax characters remain. The first real-source pass revealed unreadable/empty alternatives and a tag-metadata argument collision; both were corrected before successful preparation. No source readings were manually edited.
