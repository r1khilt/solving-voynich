# BOUNDARY-CHANNEL-0004 — prospective image-only box registration

**Pre-image registration, 2026-09-25.** The eight new page scans named below have not been downloaded or inspected at registration time. They are from training-assigned physical leaves, not manuscript validation/final-test leaves. This is a test of whether source word-box coordinates can be located on page scans without any of the published 300-row pilot's locator midpoints, crop heights, thresholds, ink gaps, labels, or QC decisions. It is an evidence-quality gate for later decipherment work, not a plaintext or word-boundary test.

## Prior work and rationale

The [published direct-pixel audit](https://arxiv.org/html/2608.17096v1#A3) used human-drawn boxes to locate text regions on six Yale scans and cautioned that coordinate provenance is incomplete. Our BOUNDARY-CHANNEL-0002 pixel replay was exact, but BOUNDARY-CHANNEL-0003 found a plausible adjacent-line offset on at least two of those pages. OpenCV's [template-matching definition](https://docs.opencv.org/5.0/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html) gives a deterministic normalized-correlation search; [ECC registration](https://docs.opencv.org/3.4.0/dc/d6b/group__video__track.html) is designed to refine an approximately aligned image pair and is ill-suited as the first step for a sparse box layout against manuscript ink. Historical manuscript transcript mapping likewise requires both image geometry and ordered text context; these boxes supply geometry only. The method below tests *global layout fit* and cannot by itself certify which ink strokes spell any particular transcribed word.

## Frozen selection and inputs

Source archive `956a7c4fc39981f4d116fa3f4edfccce6d065571`; ZL physical-leaf split SHA-256 `9fc80cb4b000fdd5d952b7c2416b37b6b92f07bc56486a63309142953ed6634e`; Yale MS 408 IIIF manifest SHA-256 `317d58fd9ea90392a83d9858a91eada3d0b41416a3c835857dc0154bd123a309`. The tracked [selection manifest](../../data/manifests/boundary_registration_0004.json) gives exact canvas IDs and source-box hashes. The IIIF 700-pixel route was independently checked against the already pinned f42v scan: a fresh download had the same SHA-256 `6951b633415253940a073a7cd2bf6323523509261aa600f398798301d55436b4`.

Eligible source pages have a Yale canvas, belong to a *training* physical leaf, contain at least 70 boxes and eight inferred lines, and do not share a physical leaf with any of the six exposed pilot pages. Sort them by SHA-256 of `BOUNDARY-CHANNEL-0004|<folio>` and select the first eight; the runner replays this rule before any scan analysis. The fixed pages are **f1r, f112v, f108r, f47v, f93r, f20r, f18v, f77v**, in that hash order. The deliberately heterogeneous pages include dense text and short/partial blocks; failures will not be replaced by hand-picked easier pages. Download only Yale's 700px derivative for each into ignored `data/raw/boundary_ink_0004/yale700/`; record each SHA-256 and image size. Do not redistribute scans or crop sheets in Git.

## Frozen method, baselines, and decision rule

`scripts/register_boundary_ink_pages.py` reads source box positions and scan RGB pixels only. It rasterizes the word boxes and their intervening whitespace in source coordinates. Its target map is local dark contrast (Gaussian background σ8 minus gray, subtract7, clip0–40), removes pixels with green>red+5 to reduce plant pigment, then smooths σ1.2. It rejects blank/low-information targets. For each source-page template against **every** target scan, it searches `sx,sy = .60,.62,...,1.00`, positive integer translations `tx=0..260,ty=0..150`, using OpenCV `TM_CCOEFF_NORMED`; the maximum score and affine are recorded. There is no OCR, language model, semantic information, or label-guided crop choice. The wrong-page box layouts are the main impostor controls and receive exactly the same search.

On the six **exposed development** pages, this fixed implementation ranked the correct source first on 6/6 with self-minus-best-impostor margins `.0329,.0551,.0700,.1420,.0960,.1172` in folio order f14v/f23r/f39v/f42v/f56v/f7v; these values are development checks, not fresh evidence. Its image-only f42v affine was `(sx .76, tx184, sy .74, ty53)` versus same-pilot metadata `(.74,190,.75,31)`, in the direction anticipated by the independent vertical diagnostic. A known +12px x/+12px y shift of the real f42v scan moved the recovered translation **exactly** +12/+12 while keeping scale and score; a white blank scan was rejected. These checks motivated the locked thresholds below; they are not included in the prospective sample size.

**Primary prospective gate:** all eight correct-page source layouts must rank first against the other seven under the full identical search, and every self-minus-best-impostor margin must be at least `.02`. Any miss or lower margin fails the blanket registration claim. In addition, a fixed image-only manual overlay review of top, middle and bottom source-line regions on each page must find the boxes on the corresponding visible line rather than an adjacent line; ambiguous regions count against that page. This is a qualitative geometry check, so its item-level judgments will be logged, not converted into a probability claim. No page is eligible for new ink-gap/label inference until its own line and exact named-word identity receive a separate blinded crop review; the published visual-QC decisions cannot be recycled.

**Controls and limits:** the runner rejects blank targets; a synthetic known-translation test checks the matching primitive; wrong-page templates test layout specificity. After scoring, repeat a fixed +12/+12 image translation on f42v as a source-matched perturbation, and inspect whether the same transform shift is recovered. The search range may fail on radically different page scales or negative offsets; a failure is recorded, not silently widened after seeing results. This is a global axis-aligned affine and does not correct curvature, local warping, transcription mismatches or individual box mistakes. Human-drawn boxes and manuscript ink are not independent linguistic evidence. The eight pages come from one manuscript and are not independent historical manuscripts.

No random seed is used in the search. Expected local CPU time is about 4–8 minutes for 8×8 page/layout matches at 441 scale pairs; the eight 700px JPEGs should total around 1–3 MB. No paid API or model training. Stop after one full registered pass and controls; any later revised method must be explicitly exploratory on these exposed pages or use fresh training-assigned pages with a new pre-image registration.

## Execution after checkpoint freeze

First validate the registration manifest and source checksums, then commit and remotely verify the source/manifest/this plan **before downloading the eight new scans**. Download each fixed IIIF derivative from `https://collections.library.yale.edu/iiif/2/<canvas_id>/full/700,/0/default.jpg`, pin its SHA-256 and dimensions, and run:

```bash
PYTHONPATH=.:src .venv/bin/python scripts/register_boundary_ink_pages.py --group prospective_train
```

Inspect the compact cross-page score matrix, source identity ranks, margin gate, blank/shift controls, and fixed-region overlays. Record negative outcomes, image hashes, elapsed time, tests, and interpretation in a separate result section and NOTEBOOK entry before any later experiment uses the new pages.

## First prospective result after remote freeze

The pre-image code, selection, thresholds and this registration were committed as `2cef97ca7d993ea3c9a74d38f7dc898eca66840b`, pushed and verified by matching `git ls-remote origin refs/heads/main` **before** any of the eight new JPEGs was downloaded. The exact Yale route validated on f42v was then used for the fixed eight; all were 700 pixels wide, between 887 and 1004 pixels high, and their individual SHA-256 values are in `results/BOUNDARY-CHANNEL-0004/prospective_train_registration.json`. Scans and overlays remain Git-ignored. The one registered full search took **345.76 seconds** with OpenCV 5.0.0.93 and NumPy 2.5.3. Result SHA-256 `3817535474c3e7bcbe271916b96035c151115d797158445fccfa78e34288e2a2`.

| Target page | Correct-box rank | Correct score | Best wrong-box score | Correct minus wrong | Fitted `(sx,tx,sy,ty)` |
| --- | ---: | ---: | ---: | ---: | --- |
| f1r | 1 | .2525 | .1983 | +.0542 | (.76,117,.76,70) |
| f112v | **2** | .3075 | **.3268** | **−.0192** | (.90,84,.88,51) |
| f108r | 1 | .2426 | .2148 | +.0278 | (.92,40,.92,36) |
| f47v | 1 | .3825 | .2224 | +.1601 | (.80,161,.80,59) |
| f93r | 1 | .3804 | .2535 | +.1269 | (.86,73,.88,65) |
| f20r | 1 | .3584 | .2128 | +.1456 | (.90,47,.90,143) |
| f18v | 1 | .3542 | .2759 | +.0783 | (.62,61,.60,94) |
| f77v | 1 | .2757 | .2294 | +.0463 | (.82,126,.82,35) |

**Registered gate: FAIL.** Seven of eight correct layouts rank first and clear the `.02` margin; f112v does neither. The winning wrong template on f112v is f18v. It contains only **71 boxes/10 inferred lines**, and its fitted overlay covers a small upper text block on f112v; f112v's own **415-box/47-line** layout overlays most of the page. Normalized correlation is computed over each template's own footprint, so a small, coincidentally matching part can outscore a much larger correct layout. This explains a plausible *failure mode*, but it does not turn the pre-registered failure into a pass. The f18v correct transform also sits at the lower `sy=.60` search bound, and f20r's `ty=143` is near the upper translation bound; both warrant caution even though their identity margins pass.

Coarse unblinded visual checks of the selected **top, middle and bottom source lines on all eight pages** found the own-page boxes over the corresponding visible text regions rather than an obvious adjacent line. f112v's own fit spans its full text column; the f18v impostor only covers the upper part. On f18v itself, the available boxes cover a left portion of the top writing while additional visible writing lies to the right. These observations come from local overlays and are *not* blind named-word or per-boundary QC. The eight per-page overlay composites are derived Yale images and remain ignored, as does the 24-strip local contact review.

`scripts/audit_boundary_registration_0004.py` independently checks the complete 8×8 score-matrix keys, source/scan hashes, image widths, ranks and margins, and repeats the blank and +12/+12 real-image shift controls. It confirms **7/8** at the required rank/margin, blank rejection, and exactly +12/+12 recovered f42v translation with unchanged scale and score; audit SHA-256 `a26798a10255def14926b2e46e5f471ae3a4ae43455762a78fb6444df7b4ebcb`. The auditor does not rerun all 64 scale searches, so its scope is arithmetic/input/control verification. As an unregistered secondary diagnostic, each of the eight source templates had its *own* page as highest-scoring target when ranking down a source column; this is compatible with the small-footprint bias but does not satisfy the frozen target-row criterion.

No new direct-ink gap or certain/uncertain label comparison was performed, and no decoded word, language or cipher key was found. The next method must handle template-footprint comparability and preserve its negative controls. Any revised score on these now-exposed eight pages is development-only; another frozen set of distinct training physical leaves is required for a new prospective registration claim. The registered exact-word/crop QC remains outstanding.
