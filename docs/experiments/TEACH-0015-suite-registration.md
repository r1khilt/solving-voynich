# TEACH-0015 fresh three-recipient suite registration

**Status:** generation and independent structural audit complete before any
TEACH-0014 final neural score was read. No TEACH-0015 model inference,
intervention, training or mechanism confirmation has run. This registers the
task population for the conditional protocol in
`TEACH-0015-mechanism-preregistration.md`; it does not admit its neural assay.

`src/voynich/workspace/teacher15_tasks.py` fixes namespace
`TEACH-0015-key-transfer-v1`, discovery seed85111 and confirmation seed85121.
Each split has128 independent logical groups, each with66 episode cells:
two queried-F assignments ×three independently remapped G recipient tables ×
two distractor topologies ×marked/fully marker-free ×two row orders (48
composed cells), plus first-hop/direct/copy controls for every F/G cell on
one fixed marked surface (18). All six recipient outputs within a group are
distinct. A single donor `(F1,G0)` can be reused unchanged against
`(F0,G0)`, `(F0,G1)` and `(F0,G2)` bases; the required outputs are the three
recipient-specific `Gj(k1)`, while donor's fixed answer is `G0(k1)`.

Every F-stage and G-stage signal-row family uses the existing TEACH-0014
`confirm` partition, while TEACH-0014 training uses `train` for both. This
establishes stage-family nonoverlap by the frozen partition function. The
no-model `scripts/teacher0015_suite_audit.py` reconstructs each visible
episode's oracle, row positions, partitions and hashes; verifies all66 cells,
matched physical skeleton/role order, both row permutations, distinct outputs,
counterfactual targets and cross-split graph/logical nonoverlap. It also
compares exact IDs with the exposed TEACH-0014 development seed74111/full128,
scorer fixture seed74117/size2, and final seed84311/full128 manifests.

The archived JSON manifests are ignored bulk outputs under
`outputs/TEACH-0015/`. Canonical manifest SHA-256 values are:

| Split | Groups | Episodes | Canonical manifest SHA-256 | Local JSON file SHA-256 |
| --- | ---: | ---: | --- | --- |
| Discovery | 128 | 8,448 | `ff044ec42ceaaea180a4cde43bbb1bd0dd73dc0c4f53c476876855831402261c` | `3714fbc66df69cc53716837e63edcbf0eb72f9cb85aa2aff1aff1438c8d7870e` |
| Confirmation | 128 | 8,448 | `941cce744278d677655d01ef71f4dfa4943352e7e477ce66efcbd5233c4db975` | `cc35fef8e0b64cdd0188e2c0dcc8be7da402863affc6d19dd076f977c4bcba69` |

The three audited TEACH-0014 exposure manifest SHAs, in that order, are
`09930778461abea7618db8d31eb9aa82c610be0e05ca4d87493a7ff9834c98af`,
`f4c47f734810e6e7997c69d72af33cb433a16d5a9fa7240b4ffc8b1c6e8e50a6`,
and `6af176d376921caccc0d40641002b94a462b42670fc4794f5b354457d17fa827`.
The independent full-suite/exposure audit passed in22.864 seconds on local
CPU and is saved under `outputs/TEACH-0015/suite-audit.json`.

To regenerate and validate the exact manifests, use the frozen seeds and
`generate_split`/`split_manifest` functions, then run `audit_splits` with
the three exposed manifests. No model weights, oracle-row inputs during
neural inference, manuscript data or paid service is used in this step.
Future intervention code must source episodes from these frozen manifests
and preserve their SHA-256 values; any new suite is a new registration.
