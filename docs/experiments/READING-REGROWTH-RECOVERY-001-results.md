# Sustained source-guided revision: modest error reduction, recovery gate failed

The registered producer completed all 388 cells and 347648 proposals under exact
freeze `51d07611ba039df6b397f9c93d563fe6313ca78c`. Both seeds improve errors and used
mapping matches over their root-only control and common initial reading. **Neither seed
recovers any complete record or used dictionary. Both inverse-competence gates fail.**
This is a synthetic exposed development result; Voynich remains unsolved.

The registered full replay audit is running under the same freeze. All results below are
producer-reported until that one audit closes. No extension, retry, best-state selection,
Gold-conditioned proposal, criterion change or empirical label transport has occurred.

| State | Edit operations | Correct used bindings /1246 | Exact records /128 | Complete used dictionaries /64 |
| --- | ---: | ---: | ---: | ---: |
| Common initial | 26012 | 28 | 0 | 0 |
| Root-only, seed92821 | 24835 | 41 | 0 | 0 |
| Regrowth, seed92821 | 24706 | 49 | 0 | 0 |
| Root-only, seed93821 | 24840 | 34 | 0 | 0 |
| Regrowth, seed93821 | 24814 | 40 | 0 | 0 |

Every positive endpoint literally completes both observations, including incorrect long
readings. Gold has18772 letters; edit operations can exceed that number. Regrowth reduces
initial errors by1306 (5.02%) and1198 (4.61%). Its advantage over root-only is129 and26
edit operations, with eight and six extra correct used bindings. Mapping accuracy remains
3.93% and3.21%, far below the fixed90% criterion. Completion is not recovery.

| Seed and arm | Attempts | Complete proposals | Accepted | Inventory-changing acceptances | Cell wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 92821 root-only | 124160 | 6649 | 23 | 16 | 320.528 |
| 92821 regrowth | 49664 | 42967 | 1812 | 400 | 563.582 |
| 93821 root-only | 124160 | 6484 | 27 | 14 | 320.104 |
| 93821 regrowth | 49664 | 42929 | 1781 | 416 | 564.254 |

The allocation was approximately compute-matched using the earlier pilot, but actual
regrowth cost was roughly1.76× root-only cost. This is not an equal-time superiority
claim. Actual total1802.291350wall/1798.609841CPU seconds, peakRSS1074659328bytes,
865114617ignored private archive bytes, $0. All registered6h/21000CPU/2GiB/4GiB
producer bounds held. The source/model inputs and immutable arrays were unchanged.

All97 cases per seed/arm retain the33 nulls and their failure/self-loop accounting in
the compact result. Accepted moves on nonsense cannot establish semantic rejection.
The apparent source-potential gain and large number of accepted edits are not calibrated
confidence or convergence. Repeated exposed cases do not supply a new holdout.

[Registration](READING-REGROWTH-RECOVERY-001.md),
[producer result](../../results/READING-REGROWTH-RECOVERY-001/result.json),
[sealed endpoints](../../results/READING-REGROWTH-RECOVERY-001/sealed-cells.json),
[audit launch](../../results/READING-REGROWTH-RECOVERY-001/audit-started.json).

Next proposed diagnosis: coordinated label transport can revise early row assignments
without rebuilding the whole text, while preserving segmentation and inventory. It therefore
cannot replace regrowth. A separate finite qualification is prepared; empirical cost and
known-answer effects require their own frozen panel. More identical proposals alone are
not supported as a route to practical decipherment by this result.
