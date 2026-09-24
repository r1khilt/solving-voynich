# HERBAL-CONTROL-0001 result: generic feature print misses the pilot gate

The pre-registered Apple Vision revision-2 nearest-neighbor test matched **13/18** chapter-identity drawings across three manuscripts. Its conditional, exact chapter-label permutation probability is **6/216 = 0.02778**, but the [registered conjunctive gate](HERBAL-CONTROL-0001.md) required at least 14/18, at least 2/3 in *every* ordered manuscript pair, and p ≤ 0.05. Casanatense→Egerton scored only **1/3**, so the decision is **pilot feasibility FAIL**. The method is not qualified to assign a plant name to Voynich f35v. No f35v model query or Voynich transcription score was run.

## Exact run and evidence

- Pre-score source/criteria commit: `93e3e4c` on `main`. The nine drawing crops and eight source-image hashes are pinned by `data/manifests/herbal_control_0001_images.json`, SHA-256 `3bb4478fd7d19241078941930f52ed9c3dd908c1f07c1de8d0264afe89bdeedd`.
- Local execution: `swiftc -sdk /Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk -module-cache-path /tmp/herbal-control-swift-cache scripts/herbal_control_0001_vision.swift -o /tmp/herbal_control_0001_vision`; then `/tmp/herbal_control_0001_vision /Users/rikhil/coding/solving-voynich /Users/rikhil/coding/solving-voynich/data/manifests/herbal_control_0001_images.json /Users/rikhil/coding/solving-voynich/results/HERBAL-CONTROL-0001/vision_distances.json`. The first in-sandbox invocation failed during Core Video buffer initialization; the same compiled binary succeeded under the required local escalation. The failed invocation produced no distance result. No training, network inference or paid API.
- Feature-distance matrix: `results/HERBAL-CONTROL-0001/vision_distances.json`, SHA-256 `02cbf2690a17e9e1f5c3ee677bfa3dc05abd823d1c65e9133714dbb5004ba0c8`. Independent Python reconstruction/audit: `scripts/herbal_control_0001_audit.py`; compact `results/HERBAL-CONTROL-0001/summary.json`, SHA-256 `13950f88f981d1f14f8f04b9b79fc313745eeb2e8af4dc0c83535df92b5013de`. The auditor checked all source/crop hashes, nine identical-image distance diagonals near zero, finite symmetric 9×9 distances, all 18 retrievals, all 216 label permutations and the frozen gate. A separate short recomputation independently reproduced 13/18 and the 6/216 tail.
- Environment and validation: macOS 26.6.2 (build 25G83), Apple Swift 6.3.3 with the explicit macOS15.4 SDK, Vision request revision 2. Changed-file Ruff, Python syntax, JSON parsing, audit, diff hygiene and the full repository suite passed; the latter had 1,056 passed, 8 skipped and 23 subtests passed in 98.01s.

| Query manuscript → gallery | Correct / 3 |
| --- | ---: |
| BnF → Egerton | 2 |
| BnF → Casanatense | 2 |
| Egerton → BnF | 3 |
| Egerton → Casanatense | 3 |
| Casanatense → BnF | 2 |
| Casanatense → Egerton | 1 |

Of five misses, four involve the *edera nigra* chapter: BnF ivy is nearer Egerton enula and Casanatense *herba vitis* than the corresponding ivy drawings; Casanatense ivy is nearer BnF and Egerton enula. The fifth is Casanatense *herba vitis* nearer Egerton enula. This describes feature-print rankings, not a historical or botanical correction to the chapter labels. The nine source images include substantial changes in composition and plant depiction; the cross-manuscript task is appropriately harder than matching copies of the same photograph.

The p-value only says this **hand-selected nine-image panel** aligns better with chapter labels than independent within-manuscript relabelings under this exact retrieval rule. It is not a population estimate, a Voynich score, or evidence that f35v depicts ivy. Indeed, the required worst-direction gate fails. A next method could use plant-part-aware crops or globally cycle-consistent matching as in Kaoua et al., but it must be registered as a new exploratory method and then tested on additional chapter classes not used to inspect these errors. Retuning the threshold or cherry-picking only the successful directions would erase the control's purpose.
