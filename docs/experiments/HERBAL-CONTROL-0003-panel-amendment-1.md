# HERBAL-CONTROL-0003: pre-score panel amendment 1

This amendment applies the image-eligibility rule registered in
[`HERBAL-CONTROL-0003.md`](HERBAL-CONTROL-0003.md) before any feature distance or
Voynich text score. The original metadata-only panel remains unchanged at
`data/manifests/herbal_control_0003_panel.json`, SHA-256
`fc2bd19a8eb25d16d8f8387a26b551e42c1f221a048d8f8b07a3f20f02c4b725`.
The active panel is `data/manifests/herbal_control_0003_panel_amended.json`,
SHA-256 `388fe569beb08bb46bb11cad175c96846b57e7b80cd3e7c209fa86079bddc832`.

## Exclusion and replacement

The original development-unknown class *gariofilis* included BnF Latin 6823
f.071v, Commons page ID 107529796, JPEG SHA-256
`392795d579a6386ae99e7ddda13585336193b9c8c13b3cc59a70aa0a03a9242f`.
The name appears in a two-column chapter index on that page. The prominent
illustrations are associated with other entries; there is no locatable drawing
for the *gariofilis* chapter on the selected BnF page. Cropping either plant
as *gariofilis* would fabricate a visual target. The Egerton source was also
downloaded, but this whole class is excluded because one manuscript fails.
No crop or model feature was produced for this class. The two already
downloaded JPEGs remain ignored raw files, not part of the active source set.

The registered first-unused-reserve rule substitutes *branca ursina* into the
same **development-unknown** slot. Its three previously reserved Commons page
IDs are BnF 106399567, Egerton 106366904, and Casanatense 106477597. The other
71 primary classes keep their role and order. The active primary panel still
contains 24 development-known, 12 development-unknown, 24 evaluation-known and
12 evaluation-unknown classes on 216 distinct pages. Fourteen reserves remain.
The historical chapter-label task and all stage-2 method, threshold, split,
bootstrap and evaluation gates remain unchanged. The metadata near-name
unknown stratum remains 2/12 in each split after replacement.

`scripts/herbal_control_0003_amend_panel.py` replays this exact change from the
unchanged original panel and checks the resulting page count and uniqueness.
The acquisition/crop/freezing/scoring programs now require the active panel
hash. Existing reviewed crops were regenerated with the new panel provenance;
their source pages and boxes were not selected using image-model output.
Future eligibility exclusions, if any, require a separately documented panel
amendment before the complete source/crop freeze. These are chapter-metadata
labels, not botanical identifications or Voynich readings.
