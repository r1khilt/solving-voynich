# Borg exception review 004: the transcription mixes several kinds of marks

2026-09-30. **Local visual preparation, no historical decoding.** The fixed
15-page panel covers all 35 types left outside the single-character atlas and
the separately listed punctuation candidates. It samples 45 occurrences from
665 exceptions; it does not validate every occurrence or provide a cipher key.

The [plan](borg-exception-review-004.md) and selection program were published
and remotely verified at `7a90cd6c106c9dd886237ed7f8a74bc91463dbcb` before
acquisition. All 15 requests succeeded: 7,333,765 bytes, width 1600, two requests
at a time, no paid services. Exact URLs, hashes and request outcomes are in the
[acquisition manifest](../../data/manifests/borg_exception_review004_acquisition.json).
All images and crops remain ignored for private study.

One unblinded reviewer inspected the full pages and the fixed target locations.
Each page has two distinct eight-glyph source-line matches in the
[visual ledger](../../results/BORG-EXCEPTION-REVIEW-004/visual_ledger.json):
30 line checks, 240 common-glyph occurrences. They support local page identity,
not complete collation of the page. The 45 targets have these recorded statuses:

| Status | Target occurrences |
| --- | ---: |
| Local shape observed; meaning and global classification unresolved | 38 |
| Location supported, shape or segmentation uncertain | 2 |
| Individual transcription-character/ink correspondence unresolved | 5 |

## Concrete observations

- **Case can retain a real shape distinction.** On 4v, the W-coded double-loop
  form has a broad bar above it, while an ordinary w-coded double-loop nearby
  lacks that bar. On 185r, the X-coded marks in an XXX-like group are simple
  diagonal crosses; ordinary x-coded marks are multi-arm stars. These examples
  argue against blind case folding, without proving a global mapping rule.
- **Some entries describe cursive initials or abbreviation-like groups.** The
  fixed C/D/E/F/G/J/L/N examples show large or distinct cursive forms. H and B
  on 140r jointly correspond locally to an lb-like cursive group, unlike the
  common h-coded crossbar form. A and U on 4v are part of a joined overmarked
  abbreviation. The transcription's two characters are not independently
  certified as two cipher glyphs. No abbreviation expansion was used as a crib.
- **Some entries describe notation or layout.** The Z-coded curly signs and
  adjacent upright strokes, the XXX/VI-like group, punctuation-like marks and
  the small z-coded 2-like mark occur in visibly separate notation contexts.
  Their semantic values are not assigned. The source's hyphens on 120r sit
  beside a grouping-brace-like image stroke; parentheses on 102r describe
  curved grouping marks alongside consecutive lines. Individual hyphens and
  parentheses are not established as separate emitted cipher glyphs.
- **Some resemblance is only a lead.** The two g-coded targets on 22v resemble
  the atlas5 hooked form. No g-to5 substitution is applied. The a-coded touching
  strokes on 4v, Y-coded gutter-adjacent strokes on 10v, and sole 7-coded faint
  mark on 107r remain explicitly uncertain. No target was replaced by an easier
  occurrence, and no higher-resolution rescue changed the fixed panel.

Original cleartext headings, abbreviations and other writing were visible in
these images. All 15 pages are preparation-exposed; those strings were not
used for cipher fitting, language scoring or a key comparison. The reviewer
was already aware of the source characters and is not an independent
paleographic annotator. Descriptions such as “lb-like” and “2-like” describe
appearance, not established readings or measurements.

## Mechanical validation and consequence

The [artifact auditor](../../scripts/audit_borg_exception_review004.py)
regenerates the deterministic selection from the pinned raw source, checks
every image and source-byte binding, replays 31 unique crops pixel-for-pixel,
checks the two common-glyph lines per page, and requires all 45 original
targets exactly once. Eight artificial audit tests include omitted/duplicated
targets, missing line diversity, wrong line identity, changed crops and
unsupported certainty. With prior Borg and descriptive-supplement checks,
64 focused tests pass. Changed Ruff passes. These checks validate provenance
and arithmetic, not independent recognition of handwritten shapes.

The result narrows the input problem: **59 remaining transcription codepoint
types are not automatically 59 interchangeable kinds of cipher glyph.** Some
differences are visually meaningful; other entries may represent auxiliary
notation, multi-stroke abbreviations or layout. Frequency alone cannot decide
which. Preserve the existing raw bytes and conservative quarantines, and build
an explicit image-grounded token/uncertainty layer before historical fitting.
Do not collapse case, silently discard rare types, concatenate across excluded
marks, or infer a global alias from this sample. The original parser, cipher
search family and running fresh qualification remain unchanged. Solver records
produced: **zero**. No Borg or Voynich decipherment is claimed.
