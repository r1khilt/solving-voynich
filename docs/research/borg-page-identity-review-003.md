# BORG-PAGE-REVIEW-003: fixed image checks for ambiguous transcription blocks

2026-09-30, before acquiring these eight page images. This is source-quality
review, not key recovery or a linguistic experiment. The existing folio10r
pilot is separate. Primary inputs are the pinned legacy transcription
SHA25679950123a2760e92f3f27287aca94d16168169651eac8a4f9c15af37e349dde3,
the acquired Vatican IIIF catalogue and the key-free visual atlas described in
[the resource review](borg-glyph-resource-review-2026-09-30.md).

## Fixed scope

Acquire only the catalogue canvases whose complete labels are `49r`, `49v`,
`99r`, `99v`, `150r`, `150v`, `150r.[01.fx.0000]`, and
`150v.[01.fx.0000]`. All must be uniquely identifiable by complete label in
this catalogue. Use its exact image-service URLs, width1600, up to2.5MB each,
45-second timeout, at most2requests concurrently, no answer/key files and no
paid service. Preserve raw bytes, source catalogue identity, URL, size and
hashes; images and crops stay ignored under Vatican's study-use terms. No
automatic alternate image, relabeling, or deletion after a retrieval failure.

Investigate these three declared possibilities:

- Legacy blocks96/97, both labeled0049v, may correspond to49r/49v. Compare both
  candidates against both blocks rather than silently correcting by order.
- Block196 labeled0099v contains43physical source lines, about twice the
  neighboring blocks. It may contain two sides with a missing page delimiter.
  Test99r/99v correspondence and any visible boundary; do not assume a split at
  the midpoint or insert invented text.
- Blocks297–300 preserve150r/v and .01 suffixes. Verify against all four
  catalogue candidates rather than merging them by numeric base label.

## Review and acceptance

The reviewer has already seen the candidate ciphertext prefixes, so this is
explicitly unblinded visual collation, not independent paleographic validation.
Use glyph shapes and layout only; do not infer plaintext or select an alignment
because it produces good Latin. Record any original-cleartext exposure without
using it as a crib. Ordinary-looking letters do not automatically mean a span
is cleartext, since this cipher's visual inventory includes letter-like marks.

For a supported candidate mapping, record at least two separate source-line
matches containing at least eight consecutive individually identifiable glyphs
each, with image crop coordinates, exact source offsets and a description of
the visible correspondence. Record competing candidates and disagreements.
Two short matches are local evidence, not full-page collation. If page boundaries
are revised, separately match the ending/starting lines around the boundary;
otherwise retain the boundary as unresolved. Do not overwrite the raw file or
the existing parser. Proposed physical identities belong in a separate ledger
with supported, contradicted or unresolved status and explicit review scope.

Output a compact ledger and report all eight images, including any failure.
No solver records or train/test split are created here. A future decoding
registration must decide whether this level of single-reviewer evidence is
adequate and must keep unresolved material out of correctness claims.
