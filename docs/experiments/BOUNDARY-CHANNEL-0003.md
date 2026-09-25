# BOUNDARY-CHANNEL-0003 — direct-ink locator line-identity challenge

**Status:** retrospective exploratory diagnostic, 2026-09-25. This is a continuation of [BOUNDARY-CHANNEL-0002](BOUNDARY-CHANNEL-0002.md), using the same six training-assigned folios and their already-exposed published pilot measurements. It is not an independent blind image study or a decipherment result.

## Question and source basis

The [source paper's Appendix A.3](https://arxiv.org/html/2608.17096v1#A3) reports ink-to-ink gaps on six 700-pixel Yale scans and describes word boxes as *locators* for adjacent token regions. It also says their coordinate provenance is incompletely documented and that the vertical-overlap-only pixel stress is weak. The [pinned source measurement program](https://github.com/lrozanova/voynich-units/blob/956a7c4fc39981f4d116fa3f4edfccce6d065571/analysis/direct_pixel/measure_direct_pixels.py) crops around the locator boxes, thresholds ink, and finds nearest ink edges around the horizontal midpoint; its algorithmic QC does not establish that the cropped strokes belong to the named neighboring words. After the exact replay in BOUNDARY-CHANNEL-0002, we asked whether each source-box line is vertically aligned with the corresponding visible word pattern, instead of an adjacent text line.

Inputs are the pinned public box archive at `956a7c4fc39981f4d116fa3f4edfccce6d065571`, the six ignored Yale IIIF JPEGs, and the same-pilot affine transforms in `results/BOUNDARY-CHANNEL-0002/pilot_replay.json`. The source-results SHA-256 is `3c15627dc3ddfca4f331a1fed1e1a290a9b8412ebf9247fe4c9b2778412aedee`; per-image and per-box hashes are in `results/BOUNDARY-CHANNEL-0003/locator_alignment.json`. No validation or final-test manuscript text or image was used.

## Diagnostic and controls

For each source line, the script projects its boxes into the scan using the archived affine and builds a smoothed horizontal pattern of occupied word positions. It builds a corresponding horizontal profile of locally dark strokes in an 11-pixel band around the predicted line center, suppressing green illustration pigment. The score is the mean centered-cosine agreement between those two x patterns over source lines spanning at least 80 pixels. The script scans vertical offsets from −30 to +35 pixels while keeping the published x placement and scale fixed. It then compares the saved pilot position with the best matching offset. Only **after** this pixel-only selection does it use the published gap values to count changes.

For a spatial control, each line's box pattern is circularly shifted by at least 20 pixels within its x span, independently, and the **maximum across the same vertical offset search** is recorded. There are 200 deterministic null draws per folio. This null tests x-pattern specificity under this score; it does not model all plausible neighboring-line coincidences, and the method was designed after looking at the six pilot pages. Manual overlays of the original and shifted boxes were inspected locally but are not redistributed with the Yale images.

| Folio | Best vertical shift from saved locator | Score at saved → best | Offset-search null draws at least as high | Same gap after shift, among finite old/new |
| --- | ---: | ---: | ---: | ---: |
| f14v | +16 px | .064 → .192 | 0/200 | 5/33 |
| f23r | +17 px | .011 → .251 | 0/200 | 13/62 |
| f39v | −1 px | .280 → .283 | 0/200 | 54/76 |
| f42v | +20 px | .081 → .499 | 0/200 | 6/35 |
| f56v | +1 px | .155 → .156 | 0/200 | 37/53 |
| f7v | −21 px | −.047 → .077 | **101/200** | 1/33 |

The most compelling page is f42v: **15/15** source lines improve at +20 pixels, and only 6/35 finite published gaps survive unchanged. f23r has 9/11 lines improve and 49/62 finite gaps change. f14v has 6/9 lines improve and 28/33 gaps change, but its selected shift moves between +7 and +16 under alternative dark-pixel maps. f39v and f56v favor the saved position within about one pixel. f7v's negative shift is *not* distinguishable from the spatial null by this test, so its changed gaps cannot be taken as an alignment correction. For f23r and f42v, alternative grayscale/high-contrast masks still favored shifts of roughly +14 to +20 pixels. The 11-pixel row band itself limits fine offset precision: a known-shift synthetic control recovered the injected line only within five pixels, not at the exact pixel.

This is evidence of a **line-registration problem on at least some pages**, strongest on f42v and f23r. It does not prove that the paper measured the wrong named words in every accepted crop. The old blind visual-QC decisions were not repeated on shifted crops. On the 283 rows with both old and shifted finite gaps that *passed the old QC*, the descriptive certain/uncertain means move from 4.985/3.143 pixels (262/21 rows) to 5.485/3.286 pixels; the gap contrast does **not** disappear. Those shifted numbers are not a corrected effect estimate: the offsets were chosen after exposure to this pilot, f7v's choice is unqualified, and new blind crop-identity/QC review is required. No word-boundary, language, key, or plaintext claim follows.

## Reproduction and next gate

With the same ignored scans and pinned external archive as BOUNDARY-CHANNEL-0002:

```bash
PYTHONPATH=.:src .venv/bin/python scripts/audit_boundary_ink_locator_alignment.py
PYTHONPATH=.:src .venv/bin/pytest -q tests/test_boundary_ink_locator_alignment.py tests/test_boundary_ink_vertical_stress.py
```

The next methodological gate is to recover scan-to-box registration **without** using this pilot's locator, crop, or threshold metadata, and to verify line and named-word identity with blinded image crops. Calibration must leave whole pilot folios out; then a frozen method can be tried on new training-assigned pages with shift/blank controls. Re-run visual QC at the newly selected locations before interpreting any label contrast. This gate remains open. The source code and compact result are tracked; all Yale scans and crop overlays remain ignored.
