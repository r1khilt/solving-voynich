# Eight Borg image correspondences support two metadata repairs

2026-09-30. Single-reviewer, unblinded visual collation; no cipher key or
plaintext recovery. The [fixed plan](borg-page-identity-review-003.md) was
published and remotely verified at
`f9f3c2b88bafa9e68ddfce665d2c025ba27ed999` before image acquisition.

All eight selected [Vatican catalogue](https://digi.vatlib.it/iiif/MSS_Borg.lat.898/manifest.json)
images were acquired successfully, totaling3,866,916bytes, with no paid service.
They remain ignored for private study. The
[acquisition manifest](../../data/manifests/borg_page_review003_acquisition.json)
pins the catalogue, requests, bytes and hashes. No published key, corrected
plaintext, translation or linguistic model was consulted. Some original
cleartext is visible on these pages; that exposure is disclosed, not used as a
crib. The pages are now preparation-exposed.

## Supported local correspondences

Each row has two separate source-line comparisons of at least eight identifiable
glyph occurrences, recorded with source byte offsets and exact image crops.
The [visual ledger](../../results/BORG-PAGE-REVIEW-003/visual_ledger.json) contains
16 such comparisons and one additional page-boundary comparison:169glyph
occurrences total. This supports local page correspondence, not perfect
transcription of every mark or independent paleographic certification.

| Legacy block | Preserved legacy label | Source lines, inclusive | Supported catalogue image |
| --- | --- | --- | --- |
| 96 | 0049v | 2012–2031 | 49r, canvasp0101 |
| 97 | 0049v | 2032–2052 | 49v, canvasp0102 |
| 196, first portion | 0099v | 4097–4118 | 99r, canvasp0201 |
| 196, second portion | 0099v | 4119–4139 | 99v, canvasp0202 |
| 297 | 0150r | 6131–6151 | 150r, canvasp0305 |
| 298 | 0150v | 6152–6170 | 150v, canvasp0306 |
| 299 | 0150r.01 | 6171–6188 | 150r.[01.fx.0000], canvasp0307 |
| 300 | 0150v.01 | 6189–6206 | 150v.[01.fx.0000], canvasp0308 |

The two49v blocks are distinguishable by their image line openings and subsequent
glyph runs; assigning both to the same verso would be wrong. The four150views
have different visible texts and layouts and should not be merged by stripping
their suffixes. Pairing their recto/verso sides as physical leaves still follows
catalogue organization rather than a new codicological examination.

The oversized99v block contains an existing `#0099v` comment at source line4119.
The conservative parser had correctly preserved it as a hash comment, because
it is not a standard `#page` header. Image99r matches the earlier portion;
image99v matches the later portion. The final cipher line on99r matches4117,
followed only by a blank line and that marker; image99v starts with4120.
Thus this boundary is supported by both original metadata and adjacent image
content. It is not an inferred midpoint. The initial0099v label remains raw
evidence of a mismatch; no original bytes were corrected or erased.

## Validation and consequence

The [artifact auditor](../../scripts/audit_borg_page_review003.py) independently
reconstructs all17crops pixel-for-pixel from the acquired images; checks all
source offsets, hashes, line limits, labels and marker references; and verifies
that the proposed eight segments preserve every byte of the seven original
blocks without overlap or loss. It does **not** independently recognize the
glyphs. Nine interval checks reject gaps, overlaps, duplication, empty pieces
and omitted endings; these and existing Borg resource/parser tests give47passes.
Changed Ruff passes.

This removes two concrete obstacles to constructing defensible page splits.
The original parser remains unchanged, and no solver record or training split
has been emitted. Other repeated catalogue labels, glyph-class exceptions,
original cleartext, catchwords and uncertainty still need an explicit policy.
A future curated historical pilot can use these reviewed identities while
preserving unreviewed regions as unresolved; it cannot claim full-manuscript
transcription validation from these17local checks. The current synthetic
fresh-key campaign is unaffected.
