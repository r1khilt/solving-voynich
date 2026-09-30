# Borg: a key-free visual alphabet resource is now available

2026-09-30. Preparation only; no key fitting, language score or published answer
comparison. This advances the missing-resource finding in
[the earlier parser review](borg-next-executable-control.md), without changing
the running [fresh recovery experiment](../experiments/BLIND-CHANNEL-CONFIRM-001.md).

## What we found and checked

The official [DECRYPT TranscriptTool repository](https://github.com/decrypt-project/transcripttool)
contains Borg support examples for its recognition system. At verified commit
`eb05861a8e2841079fea586c985eff7a326b3892`, its
[Borg alphabet directory](https://github.com/decrypt-project/transcripttool/tree/eb05861a8e2841079fea586c985eff7a326b3892/gpu/few_shot_train/alphabet/borg)
has 24 labels: 21 one-character names and `cl`, `cm`, `dt`. We acquired the first
image in every class plus three support files: 27 files, 74,408 bytes. Their
sizes, SHA-256 digests and Git blob identities match the pinned inventory.
The loader uses class directory names to retrieve support images. These are
visual class examples, not a published cipher-to-plaintext key.

The underlying [few-shot recognition paper](https://arxiv.org/abs/2009.12577)
and [authors' implementation](https://github.com/dali92002/HTRbyMatching)
provide a route to recognizing shapes from examples. We have not downloaded
model weights, reproduced recognition performance or trained this model.

The [resource manifest](../../data/manifests/borg_glyph_resource_inventory.json)
records all retained files. A first acquisition stopped safely when `M/1.jpg`
and `m/1.jpg` collided on the Mac's case-insensitive filesystem. We reused
verified bytes and stored every resource by its Git hash, preserving both
classes; no earlier image was overwritten. This is a practical reason not to
clone and blindly consume a case-sensitive alphabet dataset on this machine.

The root code license is Apache 2.0; the nested HTR code has an MIT notice.
The repository expressly reserves Borg image rights to the Vatican Library.
Images, example crops and our contact sheets remain ignored, for private study.
Code licensing is not treated as permission to republish manuscript images.

## Local visual evidence, and its limits

We acquired one [Vatican manuscript image, folio 10r](https://digi.vatlib.it/view/MSS_Borg.lat.898/0023),
501,171 bytes, at a requested width of 1,600 pixels. This is a preparation pilot
chosen before image inspection, not a confirmation page. One reviewer compared
32 common glyph occurrences spanning 15 distinct atlas labels against the first
two lines of the pinned legacy transcription. The corresponding byte intervals
are recorded in the manifest. For example, the triangle, H-like, star-like and
cross-with-loop shapes agree with their atlas class names locally. This is
manual local correspondence evidence, not a complete legend or an independent
paleographic review. We did not infer plaintext letters from these shapes.

The whole page image also exposes original cleartext headings and abbreviations.
Those exposures are disclosed and are not cribs. This directly reinforces the
parser's earlier limitation: unmarked remaining material cannot automatically
be called ciphertext. A future historical experiment needs a separate fixed
policy for original cleartext, uncertainties and catchwords.

The three longer atlas names look like punctuation in the inspected examples.
They are **not** demonstrated multi-character tokens in the legacy ASCII file:
within preserved unresolved spans, literal `cl` and `dt` occur zero times, and
`cm` only twice. Our earlier lead that longer names might explain tokenization
was a hypothesis; the current evidence does not justify a longest-match parser.
No `cl→:`, `cm→,` or `dt→.` alias is applied.

## Whole-file coverage without silently removing exceptions

The [executable resource audit](../../scripts/audit_borg_glyph_resources.py)
checks every acquired resource against its pinned tree entry and retains the
old lossless quarantine unchanged. It reports
[these counts](../../results/BORG-GLYPH-PREP-002/coverage.json):

| Lexical category within the old unresolved view | Occurrences |
| --- | ---: |
| Characters that exactly match the 21 single-character atlas names | 97,219 |
| Literal comma, period and colon, reported separately | 3,414 |
| All other characters, individually retained in the report | 665 |
| Total | 101,298 |

This accounts arithmetically for the entire old view. It does not prove that
97,219 instances are correctly transcribed, that the three punctuation marks
are non-cipher symbols everywhere, or that 665 exceptions are dispensable.
The exceptions cover 35 types. Some may be original writing, other glyphs,
scribal variants or transcription problems. Frequency cannot decide which.
All previous annotation and uncertainty quarantines remain separate and are
not included in this denominator. The original 59-type inventory remains true.

Six new artificial tests plus 32 existing parser tests pass. The new tests
cover distinct `M/m` assets, no automatic multi-character merging, no matches
across annotation or whitespace boundaries, preserved cleartext/marker
quarantines, corrupted resources, path escape, duplicated labels and incomplete
or relabeled source-tree inventories. Changed Ruff passes. These are root
authored mechanical checks, not independent visual validation. Solver records
produced: **zero**.

## Physical page identity is a separate problem

The official [IIIF catalogue](https://digi.vatlib.it/iiif/MSS_Borg.lat.898/manifest.json)
contains 430 canvases with 410 distinct numeric recto/verso base labels. Labels
alone are not unique identifiers: eight base labels have multiple canvases,
including repeated `1r/1v/2r/2v` and suffixed views at `133r/v` and `150r/v`.
The legacy transcription has two `49v` blocks and two blocks each with base
`150r/v`. Its absent `49r` and `99r` have catalogue candidates; `205r/v` also
occur only in the catalogue. The
[candidate inventory](../../results/BORG-GLYPH-PREP-002/canvas_candidates.json)
preserves these discrepancies without assigning pages or merging suffixes.
Canvas identity, image-content verification and physical leaf identity must
be resolved before claiming page-held-out recovery.

We also inspected only parsing functions from a separate
[benchmark builder](https://github.com/matthewdgreen/cipher_benchmark/blob/729aad62d12483c549e64a2541d4f9255538c8cf/scripts/create_borg_benchmark.py).
Its page dictionary overwrites duplicate raw labels; its normalizer drops
suffixes, and its symbol conversion uses per-line bracket removal and raw
character IDs. These choices are inadequate for this project's unresolved
duplicate/multiline cases. Its curation is not independent validation of our
legacy transcription units. No benchmark plaintext, symbol-to-plaintext map,
answer comparison or external solver was used.

## Consequence for the next historical test

We now have a concrete visual resource and a small directly checked bridge to
the legacy file. The next step is a fixed image-review panel covering all atlas
classes, case contrasts, rare exceptions, uncertainty and page discrepancies.
Keep original glyph IDs and an explicit unresolved category; never use Latin
fluency to relabel a difficult mark. Freeze a review ledger and physical-leaf
assignment before fitting a key. A simpler single-glyph substitution family
may then be appropriate for a validated subset, but neither that family nor
the subset has been qualified by this preparation. No Borg or Voynich reading
has been established.
