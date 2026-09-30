# Borg exception review 004: all remaining codepoint types

2026-09-30. Preparation only. Freeze the selection and this policy before
acquiring the selected images. The original transcription and parser remain
unchanged; no cipher key, published reading, language score or solver record.

The [prior coverage audit](../../results/BORG-GLYPH-PREP-002/coverage.json)
retains 665 occurrences of 35 types outside the 21 single-character atlas
labels and three separately listed punctuation candidates. Their rarity does
not justify removing them. This panel asks what those marks look like in the
manuscript, and whether the legacy file represents original writing, a distinct
graphical class, a transcription error, or something still unresolved.

The [selection program](../../scripts/plan_borg_exception_review004.py) uses
only the pinned transcription's preserved unresolved spans. Greedily select
the page adding most uncovered types; break ties by total exception count and
then lowest original block index. Stop when all 35 types are covered. For each
type newly covered on a page, fix its first two occurrences (or sole occurrence)
as review targets. No adjustment after viewing images. The resulting
[plan](../../data/manifests/borg_exception_review004_plan.json) fixes 15 pages:
4v, 10v, 140r, 9v, 185r, 8v, 22v, 82r, 120r, 135r, 42v, 102r, 134v, 9r, 107r.
Every selected full catalogue label has one candidate; this is a catalogue
binding, not visual identity verification. All pages become preparation-exposed.

Acquire the exact IIIF images in the plan, at width 1600, at most two requests
concurrently, 45 seconds and 2.5 MB per image: 37.5 MB maximum, no paid services.
Preserve failed requests rather than selecting substitute pages. Keep raw images
and derived crops ignored, consistent with the Vatican private-study restriction
in the [resource review](borg-glyph-resource-review-2026-09-30.md).

Before accepting a target correspondence, require two distinct source-line
sequences of at least eight common atlas glyphs to establish local page identity,
then locate the target using adjoining shapes and line position. Record source
byte interval/hash, canvas/image hash, crop box/hash, observed shape/context,
confidence limitation and status. A missing, blurry or unlocatable target stays
unresolved. Do not infer plaintext values from appearance or nearby cleartext.
Original cleartext visible in the images is disclosed and never used as a crib.

This is one unblinded reviewer's preparation sample. An apparent match on one or
two examples does not establish a global alias, authorize case folding, validate
all 665 exceptions, or certify the remaining body as ciphertext. Any future
normalization policy must state its evidence, retained uncertainty and selection
loss before a separate historical experiment. No automatic normalization or
training split is produced here. The running fresh-key experiment is unaffected.
