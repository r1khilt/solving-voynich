# BOUNDARY-CHANNEL-0005 — cohort-centered batch registration

**Pre-image registration, 2026-09-25.** The eight new Yale scans below have not been downloaded or viewed. This is a prospective test on training-assigned physical leaves after BOUNDARY-CHANNEL-0004's registered **failure**, not a reclassification of that result. The question is whether a source template can be located and distinguished from seven wrong folios when its raw correlation is calibrated against how broadly it matches other scans. This remains an image-evidence gate, not a claim about Voynich words or meaning.

## Motivation, prior art, and exposed development

In [BOUNDARY-CHANNEL-0004](BOUNDARY-CHANNEL-0004.md), 7/8 new target scans preferred their own word-box layout under OpenCV's raw normalized correlation, but f112v preferred a small f18v layout that happened to fit its first text block. This is a *comparison across templates of very different spatial footprints*, not necessarily a bad f112v affine. A post-result coverage-power penalty could fix f112v but made f18v worse when its source boxes covered only part of its own page, so it was rejected as the registered solution. [Cohort score normalization](https://www.isca-archive.org/eurospeech_2003/sivakumaran03_eurospeech.html) in recognition problems motivates comparing a template's score to its own impostor-score background. We use only the simple arithmetic idea here; this is not that paper's speaker model or a calibrated probability.

The *unchanged* image search is the BOUNDARY-CHANNEL-0004 source `scripts/register_boundary_ink_pages.py` at SHA-256 `d47269490a1c5e493585f5e6b45217753993709fa49e222939de42121de714a9`: source word-box mask against local-dark-stroke map, `TM_CCOEFF_NORMED`, `sx,sy=.60..1.00` by .02, `tx=0..260,ty=0..150`, same full scale/position search for all source-target pairs. It sees no transcription, published locator, gap, separator label, or meaning. Source-folio (s) and target-scan (t) raw scores are (R_{t,s}). The new score is

\[
C_{t,s}=R_{t,s}-\frac{1}{7}\sum_{u\ne t}R_{u,s}.
\]

Thus a source layout that fits many target pages is downweighted; a source that uniquely fits one scan is rewarded. The background uses *the other seven scans in the same fixed batch*, so this is a **batch-specific** identity diagnostic, not a standalone confidence score for one page. It also uses a target's true-page scan in the background when that source is an impostor elsewhere; this is inherent to the batch setup and must not be represented as independent per-page evidence. The formula has no fitted weight, variance floor, or learned model.

On the six exposed pilot pages and eight now-exposed BOUNDARY-CHANNEL-0004 pages, this one formula ranks every own-source first, with minimum margins `.0453` and `.0434` respectively. These are **adaptive development checks**: we examined multiple alternatives on the exposed eight-page matrix, including mild coverage penalties, leave-one-target-out z scores, source ranks and score ratios. The pre-image rule below is selected after that exploration. None of these 14 pages counts toward the new prospective gate. The small synthetic score test confirms the arithmetic can rescue a broad-template impostor, while a duplicated-target test checks that a missing page cannot produce a complete eight-page pass.

## Frozen sample, versions, and decision

The [selection manifest](../../data/manifests/boundary_registration_0005.json) pins source archive `956a7c4fc39981f4d116fa3f4edfccce6d065571`, split SHA-256 `9fc80cb4b000fdd5d952b7c2416b37b6b92f07bc56486a63309142953ed6634e`, Yale IIIF manifest SHA-256 `317d58fd9ea90392a83d9858a91eada3d0b41416a3c835857dc0154bd123a309`, exact new canvas IDs and source-box hashes. Eligibility repeats the ≥70 boxes/≥8 lines/Yale canvas/train physical leaf rule, **excluding every physical leaf** in BOUNDARY-CHANNEL-0002 and -0004. Sort by SHA-256 of `BOUNDARY-CHANNEL-0005|<folio>` and take the first eight of 98 eligible; the runner replays the selection. Fixed set: **f41r, f94r, f52v, f21r, f115r, f81r, f100r, f80v**. No selected scan has been fetched or inspected at this freeze.

**Primary gate:** all eight correct layouts must rank first under (C), and each must beat the strongest wrong layout by at least `.03` centered-score units. Any miss fails. Raw-score ranks are reported as the frozen baseline. A blank target must be rejected; replacing one target's row with an exact copy of another must prevent an eight-page pass. After scoring, inspect the fixed top/middle/bottom source lines for obvious wrong-line placement and record ambiguity. Passing batch identity is **insufficient** for direct-ink/label analysis: exact named-word and crop QC must be blinded and repeated per page, especially where source boxes cover only partial text. No image, manuscript validation text, or final-test leaf may be swapped in after seeing results.

Failure modes include source-box incompleteness, similar line layouts, artwork strokes surviving pigment filtering, search-bound clipping, and batch score dependence. The target pages are all from one manuscript; eight correct identities would validate this measurement tool on these pages, not distinguish historical languages, ciphers or generators. No word-gap comparison will be run as part of this registration test.

Expected cost is ~5–7 minutes local CPU for 8×8×441 template searches, eight 700px Yale JPEGs (~1–3 MB), no paid API or model training. Freeze code/manifest/this plan with a remote-verified commit **before downloading** the new scans. Then download only `https://collections.library.yale.edu/iiif/2/<canvas_id>/full/700,/0/default.jpg` to ignored `data/raw/boundary_ink_0005/yale700/`, hash each, and run exactly once:

```bash
PYTHONPATH=.:src .venv/bin/python scripts/register_boundary_ink_cohort_0005.py --mode prospective
```

Keep raw images/overlays ignored, retain compact score matrices and negative outcomes in Git, independently audit the rank arithmetic and controls, and record the result in `NOTEBOOK.md`. If this gate fails, any changes on these newly exposed pages become development only; a further fresh physical-leaf set would be required for another prospective claim.

## Prospective result and audit, 2026-09-25

The complete pre-image registration above, runner, manifest, toy controls, and NB-188 were committed as `2983c2124d91398f918e5016c032adfcb3da6161`, pushed, and verified by matching `origin/main` to local HEAD **before** downloading any selected scan. Only the eight fixed Yale `full/700,/0/default.jpg` scans were downloaded to the ignored directory specified above. They are 700 pixels wide, 900–1026 pixels high, and individually SHA-pinned in the [compact result](../../results/BOUNDARY-CHANNEL-0005/prospective_registration.json). The registered command ran **once**, taking 363.39 seconds on local CPU with OpenCV 5.0.0. Result SHA-256: `ae9991b7517169ce18a75b8c7665cb8aafbd11e3cb93b72ccebb7f346ff14cbb`. There was no paid API use, model training, or manuscript final-test text.

The **registered batch-identity gate passed**: eight of eight correct source layouts rank first after cohort centering, and every own-minus-best-impostor margin exceeds the fixed `.03` bar. The unchanged raw score ranks seven of eight first; f41r is fifth raw and first centered. This is a useful demonstration of why the source-specific background term matters on new pages, although the batch contains related manuscript layouts and source boxes of different completeness.

| Target | Raw own rank | Centered own rank | Centered margin | Own affine `sx,tx,sy,ty` |
| --- | ---: | ---: | ---: | --- |
| f41r | 5 | 1 | .03660 | .86,52,.82,54 |
| f94r | 1 | 1 | .04189 | .88,53,.86,102 |
| f52v | 1 | 1 | .14581 | .74,188,.74,46 |
| f21r | 1 | 1 | .10616 | .88,41,.88,144 |
| f115r | 1 | 1 | .19283 | .92,33,.94,42 |
| f81r | 1 | 1 | .11193 | .80,45,.80,66 |
| f100r | 1 | 1 | .12587 | .94,43,.94,28 |
| f80v | 1 | 1 | .16573 | .82,125,.82,42 |

The [independent audit](../../results/BOUNDARY-CHANNEL-0005/prospective_audit.json) reconstructs all 64 centered scores and ranks by column-total arithmetic, checks frozen configuration/source/scan hashes, replays **all 64** raw scores at their saved affine and verifies each saved translation is optimal at that scale. It confirms 8/8 pass, rejects a blank white target, and finds only **six** pages pass when f41r's target row is replaced by a duplicate of f94r's. The audit code is `scripts/audit_boundary_registration_0005.py`. It does **not** rerun the full 441-scale search independently. The result's own score and gate therefore have an independent arithmetic and fixed-affine replay, while the global search itself retains a shared implementation.

Full-page ignored overlays were inspected for the fixed top, middle and bottom source lines on all eight pages: each appeared to lie on its intended text lines, with no obvious whole-line offset. f41r has two separated text blocks; f94r and f52v contain short/partial blocks. f21r's best vertical translation is **144**, near the registered search ceiling of 150, so a later local stability/crop review is prudent even though the visible text lines align. This is a coarse, unblinded visual check, **not** a certified per-word glyph or boundary match. The overlays and scans remain ignored; only their source/result hashes and compact observations are tracked.

**Interpretation and stop point.** Cohort centering rescued one new raw-rank failure and passed a predeclared eight-page layout identity test. It establishes that this specific batch-dependent alignment tool can locate those eight page layouts; it does not establish whether any particular named word box encloses the intended word, whether pixel gaps reflect writer-intended word boundaries, or what any Voynich text means. Do not feed these pages into a new label/ink-effect claim until a separately specified, blinded exact-word/crop QC passes. The previously reported 0004 failure is unchanged. No decoder, key, language, plaintext, or historical generation mechanism has been identified.
