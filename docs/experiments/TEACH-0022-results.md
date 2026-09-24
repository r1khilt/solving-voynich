# TEACH-0022: fresh linked-reader factorial comparison

**Registered linked-reader hypothesis: failed. Route-supervision arm: synthetic competence passed.** The fixed design and decision gates are in [TEACH-0022-linked-reader.md](TEACH-0022-linked-reader.md). This is a controlled graph-row teacher study, not a Voynich reading or a recovery of the interrupted TEACH-0014 campaign.

## What was frozen and run

Two new 19-panel, 4,736-episode evaluation suites were drawn once at seeds84411 (development) and84511 (confirmation). The no-generator structural/exposure audit passed with zero exact graph, logical or render ID overlap between them and the exposed74111 and older reserved84311 manifests. Seed84311 was checked for IDs only and never scored here. The v2 MPS resource benchmark from source `39e838b` projected4,969.20s under the4h cap, with sampled peak132,942,080B under12GiB. No paid service was used.

Three new arms ×two initialization/training seeds received exactly6,000 AdamW updates at batch32 on the same episode stream as the original `oracle_rows_workspace` answer-only baseline. `unlinked_route` preserved that baseline architecture and added coefficient0.2 cross-entropy on the correct visible row at each active read. `linked_answer` and `linked_route` tied query/key projections, carried unprojected right-symbol embeddings and scaled the learned update, with and without route supervision respectively. The linked arms had24,242,178 trainable parameters versus24,766,465 for baseline/unlinked-route; parameter and computation path were not matched. All six new checkpoints completed in2,337.19s total; peak sampled MPS132,024,576B and new artifacts632,698,881B were under the registered caps.

Before scoring, a separate trace/source/resource/suite preflight checked all six checkpoint hashes, all36,000 per-step answer-input/label hashes against the original training traces, finite losses, frozen source files and both suite manifests. The one-shot MPS scoring completed16 arm/seed/suite archives in756.04s, peak sampled MPS100,081,920B and total artifacts653,501,931B. The independent no-model auditor then reconstructed every visible answer, prediction denominator, group score, row-attention hit, source/trace/suite/checkpoint hash and decision margin from all archives. It returned `pass`. CPU checkpoint replay returned `pass` on192 complete logit vectors across all four arms, both model seeds, both suites and four fixed panels; maximum absolute logit difference0.000695, below the fixed0.002 absolute/relative tolerance. Compact audit SHA-256 `fe38a40a8d2d95add7ba9ad12b794d749902e7ead937612df2d5b80097eebbf8`; replay-audit SHA-256 `96b13fddb7c2d0d56187f55369ba241bcd02f2ef41fe07857dc5eadf23b025dc`.

## Fresh confirmation results, seed84511

Each number in the first five columns is correct items out of128; factorial is **exact four-item groups** out of128, not item accuracy. All models receive the correct public row boundaries. The two model seeds are reported separately.

| Arm | Model seed | Two-hop confirm-confirm | Exact factorial groups | First-hop | Direct | Copy | Meets absolute competence gate? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Original answer-only reader | 0 | 95 | 77 | 115 | 120 | 128 | No |
| Original answer-only reader | 1 | 88 | 57 | 115 | 121 | 128 | No |
| **Same reader + row-route labels** | **0** | **128** | **126** | **128** | **128** | **128** | **Yes** |
| **Same reader + row-route labels** | **1** | **127** | **124** | **128** | **128** | **128** | **Yes** |
| Linked reader, answer only | 0 | 16 | 0 | 13 | 17 | 128 | No |
| Linked reader, answer only | 1 | 14 | 0 | 15 | 14 | 128 | No |
| Linked reader + route labels | 0 | 16 | 0 | 13 | 17 | 128 | No |
| Linked reader + route labels | 1 | 16 | 0 | 16 | 14 | 128 | No |

The registered linked-answer margin over the baseline was **−61.72 and −57.81 percentage points**, far short of the required positive15 points. The registered linked-route margin over unlinked-route was **−87.50 and −86.72 points**, also a decisive failure. The linked arms got first target-row attention argmax correct128/128 in both seeds, yet second-target argmax only14–20/128 and two-hop answers14–16/128. Their first-row top ranking cannot be taken as learned retrieval: an exploratory random-initialization check already gave linked first-row argmax128/128 with nearly uniform mass and poor answers. The linked models did not reliably learn even direct and first-hop answers. Their final100-step mean answer losses were~2.60–2.62, and linked-route mean route losses~2.48, whereas unlinked-route mean answer losses were0.016/0.060 and route losses0.192/0.276. The source does not isolate whether the direct carry, tied key/query map, scaled update, normalization or their interaction caused this optimization failure. No trained parameter in the linked checkpoint should be treated as a discovered Voynich latent.

The **unlinked-route** arm improved confirm-confirm by33/128 and39/128 exact answers (25.78 and30.47 points) relative to the original baseline, with paired rescue33/damage0 and rescue40/damage1 on the confirmation panel. It met all registered absolute copy/one-hop/direct/factorial competence gates in both model seeds. Across harder panels it scored125–127/128 on three-hop,123–126/128 on four-hop and118–121/128 on long four-hop OOD, versus substantially weaker baseline; these are still draws from this synthetic generator. The development suite showed the same broad pattern: unlinked-route127/128 and128/128 confirm-confirm,125/128 exact factorial groups in both seeds; linked arms15–22/128 confirm-confirm and0/128 exact factorial groups.

### Exploratory post hoc read-mass diagnostic

After the confirmatory result was audited, a separate CPU checkpoint readout on the **already exposed development seed84411** quantified the first/second target attention probabilities, read-value norms and update norms for all128 confirm-confirm items in every arm/seed. Its script `scripts/teacher0022_posthoc_geometry.py` binds the original suite/checkpoint/audit hashes and reproduces the independently audited exact answers and row-hit counts before saving `results/TEACH-0022/posthoc-geometry.json`. This is descriptive follow-up, not a new preregistered confirmation, and it has no separate independent geometry replay.

| Arm | First target-row mean mass, seeds0/1 | Second target-row mean mass, seeds0/1 | First read value mean norm, seeds0/1 | First learned update mean norm, seeds0/1 |
| --- | --- | --- | --- | --- |
| Original answer-only | .4247 / .4491 | .7580 / .7407 | .6889 / .7033 | .6645 / .6873 |
| Unlinked + route labels | **.7481 / .6662** | **.9920 / 1.0000** | 1.2111 / 1.0956 | 2.3497 / 1.7164 |
| Linked answer-only | .0676 / .0675 | .0736 / .0730 | .2057 / .2053 | .3425 / .3362 |
| Linked + route labels | .0678 / .0677 | .0740 / .0733 | .2061 / .2057 | .3486 / .3428 |

Uniform mass over16 visible rows is.0625. The linked model's first-row argmax was128/128, yet even after training it placed only~.068 mass on that row: its weighted value remained close to a mixture. The successful route-supervised unlinked model gave the true first row~.67–.75 mass and the true second row virtually all mass. In the linked arms, the learned update scale decreased from its initialized0.1 to~.074; despite the “direct carry” interface, the update's observed norm exceeded the carried value's norm. These observations sharpen the diagnosis that first-row **rank** and effective value transport are different, but they do not isolate which tied/identity/scaled component prevented optimization. Attention mass is a measurement, not a causal intervention; the independent TEACH-0024/25 hard-first patches provide the causal evidence for mixture effects in the original reader.

## Meaning and limits

The two decisive observations are (1) the original model can be made to execute these public-row graph queries almost perfectly under explicit **row-route supervision**, and (2) the proposed linked architecture is not a viable replacement under the matched training stream and budget. The first is evidence that the visible graph task has a trainable routing bottleneck; it is not evidence that answer-only learning spontaneously discovers the same route. The row target is computed from a public synthetic mapping and known answer. No analogous supervision exists for the Voynich manuscript. The architecture contrast includes parameter/interface differences, so it diagnoses this particular linked design, not all direct-value or hard-read approaches. Perfect or near-perfect accuracy on this generator also cannot identify a unique internal circuit or show what the manuscript encodes.

The old TEACH-0014 campaign remains incomplete and seed84311 unopened for neural scoring. Future architecture changes must use new frozen suites. The separately registered TEACH-0025 assay tests whether TEACH-0024's native hard-first causal repair generalizes to new graphs without route labels or retraining; its threshold was fixed before this score was inspected.
