# BORG-GLYPH-BRIDGE-005: fixed key-free visual bridge

Registered 2026-09-30, exploratory preparation. No key fitting or decoding.
Question: do the 21 single-character atlas names consistently resemble their
legacy-source occurrences, and are those occurrences in the main symbol runs
or auxiliary/cursive contexts? Comma, period and colon are inspected separately;
the longer atlas names `cl/cm/dt` remain image-class names, never parser tokens.

Inputs are the pinned 166,508-byte Borg transcription SHA-256
`79950123a2760e92f3f27287aca94d16168169651eac8a4f9c15af37e349dde3`,
TranscriptTool atlas commit `eb05861a8e2841079fea586c985eff7a326b3892`, and the
existing preparation-exposed pilot/page-review003/exception-review004 images
and source-range ledgers. The selector checks their hashes and chooses the
first lexical unresolved-body occurrence per distinct canvas, in source-byte
order, retaining the first two canvases for each single-character class and
each of `.,:`. Missing slots are recorded. No replacement of difficult targets.
No new images, model weights, plaintext, language scores or published keys.
Existing cleartext/annotation/uncertainty quarantines are unchanged.

Freeze the generated plan and this program before reviewing selected locations.
The same assistant previously inspected these pages: this is neither blind
recognition nor independent philological validation. Selection depends on
already reviewed pages; it cannot estimate manuscript-wide error rates. A
different canvas is not automatically an independent physical leaf. No final
split or historical qualification is established here.

Each fixed target receives a source-bound context crop, a textual observation,
one of `local_atlas_resemblance`, `uncertain_correspondence`, `local_difference`,
and separately `main_symbol_run`, `auxiliary_or_cursive_context`,
`uncertain_context`. These are local appearance judgments, not assigned symbol
meanings or validated global aliases. Retain uncertain/different examples.
Mechanical PASS requires exact selection replay, all fixed targets once and in
order, input hashes, legal statuses and pixel-perfect crop replay. It does not
prove visual recognition. Report all statuses, missing slots and classes.

Method rationale: our earlier 59-type raw inventory and 45-exception image review
show why ASCII names cannot simply be assumed to define cipher units. Souibgui
et al. [2020, §III–IV](https://arxiv.org/html/2009.12577v1) detect symbols from
support images and then decode spatial detections; touching forms and class
similarity remain problems. Their Omniglot-trained recognition system is not
reproduced here. Lindemann and Bowern's [Voynich comparison](https://arxiv.org/abs/2010.14697)
studies transcription and glyph-composition effects; that descriptive analysis
does not validate our Borg classes or establish language. The present manual
panel addresses provenance and unit definitions, before ML or decipherment.

Bound: at most 48 targets, reuse existing private images, zero downloads/model
runs/paid APIs, CPU-only metadata and crop work; estimated under one hour and
under 100 MB derived images. Crops remain ignored for private study under the
previous image-rights policy. Stop after the fixed inventory; unresolved marks
stay unresolved. No historical solver records. Commands:

```
PYTHONPATH=.:src .venv/bin/python scripts/borg_glyph_bridge005.py plan
PYTHONPATH=.:src .venv/bin/python scripts/borg_glyph_bridge005.py audit
```

Publish source, tests, notebook and fixed plan; verify remote before location
review. Then publish compact observations and mechanical audit, without images.
The live JOINT-KEY-TRAIN-001 source, inputs, criteria and schedule stay frozen.
