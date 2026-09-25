# BOUNDARY-CHANNEL-0002 — published ink audit replay and registration stress

**Status:** retrospective exploratory measurement study, 2026-09-25. Only six *training-assigned* folios from the published 300-boundary pilot are used. The six folios and separator labels were previously inspected in BOUNDARY-CHANNEL-0001, so none of these tests is a confirmatory manuscript claim. No validation or final-test text was opened; no word, language, cipher key, or plaintext was inferred.

## Why this matters

BOUNDARY-CHANNEL-0001 found that human-drawn word-box gaps greatly exaggerate separation between ZL3b's certain (`.`) and uncertain (`,`) separator labels compared with the article's archived raw-ink gaps. To rely on even the weaker ink result, we need to know whether it can be reproduced from the actual manuscript scans, and whether its vertical and horizontal location is specific to the intended juncture. This follows the source paper's own [blind direct-pixel audit and cautions](https://arxiv.org/html/2608.17096v1#A3) and uses the [published reproduction archive](https://github.com/lrozanova/voynich-units). OpenCV's [Otsu thresholding](https://docs.opencv.org/3.4.1/d7/d1b/group__imgproc__misc.html) and [connected-component filtering](https://docs.opencv.org/3.4.6/d3/dc0/group__imgproc__shape.html) are the pixel operations in the authors' pinned program. Our stress variants are new exploratory measurements, not methods reported by those sources.

## Source and reconstruction

The external archive is checked out at `956a7c4fc39981f4d116fa3f4edfccce6d065571`. Its measurement code `analysis/direct_pixel/measure_direct_pixels.py` SHA-256 is `960ac72b7d174b639f0b3a36e5f83c664c492c3e0704f52d6b1b0768ff6e9f05`; its 300-row `results_unblinded.csv` SHA-256 is `3c15627dc3ddfca4f331a1fed1e1a290a9b8412ebf9247fe4c9b2778412aedee`. The archive does **not** redistribute the original blind locator manifest or Beinecke images. We obtained Yale's public IIIF 700-pixel derivatives for f7v, f14v, f23r, f39v, f42v, and f56v, using canvas IDs and per-image SHA-256 digests pinned in `results/BOUNDARY-CHANNEL-0002/pilot_replay.json`. The Yale IIIF manifest SHA-256 is `317d58fd9ea90392a83d9858a91eada3d0b41416a3c835857dc0154bd123a309`. The downloaded JPEGs (~700 px wide), source archive and all crop images stay Git-ignored; do not redistribute Yale scans as project assets.

The locator manifest was reconstructed only from **archived location metadata**: per-folio horizontal box-to-scan scale/shift from `locator_mid`, vertical scale from the archived crop heights, and vertical shift from grayscale threshold values on 18 evenly spaced blind IDs per folio. The calibration never used the separator labels or recorded gap values. These metadata derive from the *same* pilot, however, so the reconstruction is an exact **method replay**, not an independently selected image experiment. The registered transforms are in the result JSON. A naive word-ink-density vertical alignment initially locked onto an adjacent text line for several pages (e.g. f42v shift +50 versus recovered +31); this negative result shows why automatic extension to new folios needs line-identity validation.

Running the pinned author function on the six Yale images then reproduced **all 300/300** crop widths, crop heights, Otsu thresholds, algorithmic QC codes, ink-gap fields, and left/right ink-edge coordinates exactly (including five missing gaps; **295/295 finite gaps**). `scripts/replay_boundary_ink_pilot.py` SHA-256 `7c35234c29e1a9af0a16125d96b8669771b3272f142aa72ce2a9093483958601`; compact result SHA-256 `d46840fc5786e5ffcb30fd1046f28ceddda98a6ad3fb262b3fb3a1264fbc5992`. A first implementation run failed only while writing JSON because a NumPy integer was not serializable; casting aggregate counts to Python integers fixed output serialization without changing pixel computations.

## Six-folio sensitivity assay

On the 286 boundaries accepted by the published **blind visual QC**, we compared the source result with two new same-location metrics and four deliberate locator shifts. Every alternative uses the same pinned scan, source mask and source pair; missing algorithmic values are reported by label, and each metric is compared with the source baseline on **its exact included rows**. The full per-row derived table is ignored at `data/processed/boundary_ink_0002/vertical_stress_rows.jsonl`, SHA-256 `c83de5218906a8142681be385aa9130db7622863ceb934d416b9bc24657f59f2`. `scripts/audit_boundary_ink_vertical_stress.py` independently replays 13 complete aggregate metrics, rank AUCs, per-folio contrasts, edge-pair matches and 2,000-draw folio bootstraps from that table; max discrepancy **0**. Stress result SHA-256 `d017b2dba96e7500608c24dd419083f75ad371fe9bb8745c0bb6b1294b899e4a`; audit SHA-256 `e4ff09d8f014b7253fe4af1e4c7ca279e1cf00f85f592ce32447440bef7a7382`.

| Gap measurement | Eligible certain / uncertain | Certain minus uncertain mean (px) | AUC, chance = .5 | Positive folios / informative |
| --- | ---: | ---: | ---: | ---: |
| Published projection (raw px) | 265 / 21 | +1.842 | .631 | 5 / 5 |
| Projection inside shared box-height band | 254 / 20 | +2.625 | .716 | 5 / 5 |
| Median ink-to-ink gap on **the same scanline** | 253 / 20 | +3.304 | .756 | 4 / 5 |
| Vertical locator shifted −20 px | 232 / 18 | +.766 | .526 | 5 / 5 |
| Vertical locator shifted +20 px | 251 / 20 | +1.886 | .638 | 5 / 5 |
| Horizontal locator shifted −20 px | 260 / 19 | −.760 | .506 | 1 / 5 |
| Horizontal locator shifted +20 px | 253 / 19 | −1.342 | .397 | 2 / 4 |

The same-scanline measure has six supporting pixel rows at the median in both separator classes; it is not based solely on one isolated pixel in most cases. Its folio-bootstrap AUC interval is `[.620,.872]`, but one of five informative folios is negative, only 20 uncertain boundaries survive, and all estimator choices here are post hoc. On its exact 273 eligible rows, the published raw-pixel measure has AUC `.637`; its larger AUC is therefore not explained solely by dropping difficult rows. Seventeen uncertain boundaries have a certain comparator with the same collapsed terminal-to-initial glyph pair anywhere in the pilot, ten within the same folio and pair; the same-scanline mean contrasts are +3.20 and +3.41 px respectively. These small matched sets cannot establish an encoding mechanism.

The source archive also reports a different “vertical-overlap-only” estimator with only +0.286 px mean separation; our shared-box-band and rowwise estimators are **not identical to it**. Taken together, vertical support is estimator-sensitive. The +20 px vertical shift retains a similarly sized raw-gap contrast on 271 different eligible cases, whereas either 20 px horizontal shift destroys or reverses it. This is consistent with location-specific *horizontal* structure, but it prevents treating the current six-page contrast as uniquely localized to the intended text line. The shift controls have different QC attrition and are exploratory, so their AUCs are not a formal null distribution.

## Reproduction and next gate

Install the optional `boundary-vision` dependency group, retain the exact public archive and download the six Yale derivatives listed in `pilot_replay.json` to the ignored `data/raw/boundary_ink_0002/yale700/<folio>.jpg` paths. Then run:

```bash
PYTHONPATH=.:src .venv/bin/python scripts/replay_boundary_ink_pilot.py
PYTHONPATH=.:src .venv/bin/python scripts/boundary_ink_vertical_stress.py
PYTHONPATH=.:src .venv/bin/python scripts/audit_boundary_ink_vertical_stress.py
```

The next gate is **not** a larger language model yet. First build scan-to-box registration that does not use this pilot's locator, crop or threshold metadata; calibrate it by leaving whole pilot folios out and requiring near-correct pixel placement on all six. Use blank/shifted-image controls, raw-pixel QC and genuinely new *training-assigned* folios before testing whether physical gaps add any predictive information to a segmentation model. Freeze the geometry method and controls before looking at any repeatedly exposed validation folio. The pending blinded botanical-image review is a separate potential semantic anchor; this experiment contains no result from it.

Validation: two synthetic vertical-mixing tests passed, Ruff on all changed Python files and compilation passed, `uv lock --check` resolved, and the complete repository suite passed **1,160 tests, eight skipped, 23 subtests** in 195.08 seconds. A whole-tree Ruff check finds five unrelated pre-existing unused-variable/import findings in untouched `src/voynich/` files; this checkpoint does not alter those files. The tracked artifacts are small; raw scans, crops, the external source archive and the per-boundary derived table remain ignored.
