# TEACH-0010 results: all four downstream heads jointly write the recipient-specific object

**Registered outcome after a transparent numerical repair: `ANSWER-WRITE-SUPPORTED`. Independent audit: PASS.** A donor key transplanted at the layer-2 query state is converted into the correct object for each recipient G table. Unlike the earlier key-writing stage, no shared one-, two-, or three-head subset was sufficient across both independently trained models; the discovery rule selected all four heads.

## What ran and numerical qualification

The study generated 256 new cross-G groups from seed `70111`, split 128 discovery/128 confirmation. For each base recipient G0/G1/G2 episode, it inserted the same donor-G0 query key after two transformer layers, then recomputed the actual recipient-conditioned downstream run. Encoder layer 2 was decomposed into four query-head outputs, attention residual and MLP.

The first execution was invalid because one generic scaled-dot-product head reconstruction differed from native attention by `1.0728836e-6`, just above the frozen `<1e-6` gate. Its artifacts are preserved. The corrected run used the per-head attention weights returned by native PyTorch MHA and explicit V projections. Projected head sums then matched native attention within `7.16e-7`; residual-plus-MLP matched the native layer exactly. No behavioral choice, example, intervention, control or threshold changed, so the valid result is explicitly non-blind. Runtime was 0.481 seconds on CPU.

## Downstream readout is broader and less seed-aligned than upstream key writing

On discovery, all four heads gave minimum-both-seed 97.14% donor items and 91.41% exact three-G groups. No smaller subset met the shared threshold.

The main triple subsets exposed different seed-specific decompositions:

| Heads | Seed 0 items / groups | Seed 1 items / groups |
| --- | ---: | ---: |
| 0+1+2 | 344/384 / 92/128 | 297/384 / 59/128 |
| 0+1+3 | 113/384 / 3/128 | 110/384 / 6/128 |
| 0+2+3 | 187/384 / 19/128 | 370/384 / 114/128 |
| 1+2+3 | 4/384 / 0/128 | 0/384 / 0/128 |
| **0+1+2+3** | **374/384 / 119/128** | **373/384 / 117/128** |

Thus the robust abstraction is not “one universal answer head.” Seed 0 relies heavily on the 0+1+2 combination, while seed 1 can use 0+2+3. Only the complete four-head write is reliably sufficient across both. This is compatible with distributed, interacting routes; finite head-subset effects are not an additive attribution.

## Fresh confirmation

On each seed's 128 untouched groups:

| Intervention | Seed 0 items / groups | Seed 1 items / groups |
| --- | ---: | ---: |
| Full upstream query-key patch | 384/384 / 128/128 | 384/384 / 128/128 |
| **All four query-head deltas** | **375/384 / 119/128** | **368/384 / 113/128** |
| Edited post-attention query state | 384/384 / 128/128 | 384/384 / 128/128 |
| Edited final layer query state | 384/384 / 128/128 | 384/384 / 128/128 |
| Edited MLP write alone | 0/384 / 0/128 | 0/384 / 0/128 |
| Norm-matched random write | 0/384 / 0/128 | 0/384 / 0/128 |
| Next-recipient cyclic write | 0/384 / 0/128 | 0/384 / 0/128 |
| G0 write repeated across G0/G1/G2 | 127/384 / 0/128 | 121/384 / 0/128 |
| Input-level two-F-row control | 384/384 / 128/128 | 384/384 / 128/128 |

Every correct selected-head transfer had 256/256 cross-G non-injection. Replaying a selected G0 write across all recipients works almost exclusively on G0 and produces no complete three-table group; on the incompatible G1/G2 contexts it generally leaves the base answer intact. Likewise, a cyclic wrong-recipient write produces zero donor answers. The downstream write is therefore tightly recipient-conditioned rather than a reusable context-free key.

The gap between the all-head delta and the perfect post-attention state is also informative. The head-only intervention inserts the edited answer write into a clean residual that still contains the original base key, creating a small conflict. Patching the complete post-attention query state carries both the transplanted key residual and the recipient-conditioned head write, restoring every answer. The MLP contributes no independently sufficient donor-answer route on this counterfactual.

## Verification, circuit update and limits

`scripts/teacher0010_analyze.py` independently regenerated the suite, reconstructed the global 15-subset selection, verified frozen source/checkpoint/row hashes and all labels, rescored 13 confirmation conditions and reproduced the registered decision. It reports `audit: pass`; it does not rerun inference.

The qualified circuit now spans both lookup stages:

1. layer 1 heads 1+3 use stable QK routing to read the queried F-row V content;
2. they write a distributed reusable key into the query state;
3. encoder layer 2 reads that key in the context of the recipient G rows;
4. all four layer-2 heads jointly write a recipient-specific object representation;
5. the residual key plus attention write is sufficient before the MLP; the MLP alone is inert.

The source of the downstream write is not yet isolated. The next prospective test should factor layer-2 Q/K/V and source-slot contributions under the actual edited recipient run, asking whether the query key addresses the matching G row and whether that row's V carries the object. That symmetry is plausible after TEACH-0009 but remains untested.

All findings remain confined to a parsed, supervised synthetic system. They do not identify a Voynich object code, word meaning, cipher operation or historical language.
