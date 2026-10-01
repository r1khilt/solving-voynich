# Borg bridge 005: a transcription name does not always identify one shape

2026-09-30. The fixed panel reviewed **48 occurrences on seven existing page
images**, covering all 21 single-character atlas names and separately comma,
period and colon. No missing slots, downloads, model runs or plaintext fitting.
Source, program, tests, protocol and deterministic plan were published and
exactly verified on origin/main at `e336b01ebb11d7cadc537b81ecad80d465836414`
before this location review.

| Local visual judgment | Occurrences |
| --- | ---: |
| Resembles the atlas example | 41 |
| Correspondence or segmentation uncertain | 6 |
| Visibly differs from its nominal atlas example | 1 |

Separately, 41 locations lie in the main symbol runs, two in auxiliary/cursive
contexts, and five have uncertain context. These are **one unblinded reviewer's
local judgments**, not recognition accuracy, independent paleography or a
validated global legend. The assistant had already inspected these pages.

The concrete difference is the source `2` at line2047 on the image corresponding
to49v. It appears as a large open epsilon/E-like initial with a sweeping lower
return, unlike the compact 2-like baseline mark in the atlas and at the other
fixed source2 occurrence on134v. No semantic reading or relabeling follows.
The same legacy character can describe visually different forms in this panel;
an automatic literal-character-to-atlas-class rule therefore remains unproven.

The uncertain cases were retained: both fixed comma-coded line endings, a
9-coded mark on4v and an8-coded mark on8v near the fold, the m-coded group
touching a following curved cross on4v, and a faint colon-coded location on82r.
For the last case, the clearer earlier bracketed `y:` material stayed excluded;
it was not substituted for the selected occurrence. M/m-like resemblance does
not independently establish a consistent distinction between those classes.

The two dots at the fixed42v colon location resemble atlascl, and the fixed
periods resemble atlasdt. This is local shape evidence only. No `cl/cm/dt`
multicharacter parsing, punctuation alias, case folding, rare-mark deletion or
concatenation across quarantines is applied. Original cleartext was visible but
not used as a crib. Images and all19 context crops remain ignored/private.

The [ledger](../../results/BORG-GLYPH-BRIDGE-005/visual_ledger.json) preserves every
fixed target, source offset, observation, image binding and context-crop box.
The [audit](../../results/BORG-GLYPH-BRIDGE-005/audit.json) exactly regenerates
selection, rechecks input/image/atlas hashes and all19 crops pixel-for-pixel.
Mechanical PASS verifies provenance and accounting, not handwriting judgments.
All67 focused Borg tests pass, including omission, replacement, unsupported
certainty and crop-corruption checks; full-suite validation is recorded in the
notebook. No historical solver records were produced.

The practical next step is an explicit source-to-image token layer that can
retain alternative shapes, touching groups and auxiliary writing at specific
locations. Its unit/split policy must be frozen and qualified before historical
key fitting. The current six-glyph synthetic solver cannot silently consume
this larger historical alphabet. This work reduces an input ambiguity; it does
not decode Borg or Voynich.
