# TEACH-0015 physical-row pointer counterexample

**Status:** structural falsification exercise, specified after inspecting the
frozen visible suite but before any TEACH-0014 final neural behavior or
TEACH-0015 trained inference. This is a constructed symbolic mechanism, not a
learned model or confirmatory statistical estimate.

## Question and method basis

Does recipient-specific transfer on the registered three-G suite uniquely
identify the intermediate **symbol** `k1` at `query.1`? It cannot if a state
that contains only a physical G-row pointer produces the same counterfactual
outputs. Causal abstraction work distinguishes successful interchange behavior
from the uniqueness of an internal alignment
([Geiger et al., 2024](https://proceedings.mlr.press/v236/geiger24a.html));
the [Anthropic J-space study](https://transformer-circuits.pub/2026/workspace/index.html)
uses a separately defined Jacobian-linked token dictionary and does not turn
arbitrary task success into a symbol-coordinate identification.

The frozen generator renders donor and recipients with the same surface seed.
Changing G remaps output values but preserves the physical row permutation.
A no-neural inspection found the `G_j(k1)` edge at the same physical
`serialized_rows` index across G0/G1/G2 in all1,024 surfaces of **each** split.
This is a source-derived structural observation, not a model result.

## Constructed mechanism and audit

For each donor `(F1,G0)` find the physical row index `r1` of its visible
`(k1,G0(k1))` edge. Define its hypothetical first-read state as an encoding
of **only** `r1`, without `k1` or a key-token direction. In a base recipient
`(F0,Gj)`, define the second read to output the right-hand value of physical
row `r1`. For reverse necessity, obtain `r0` from `(F0,G0)`'s `(k0,G0(k0))`
edge and read that slot in the `(F1,Gj)` recipient. This is an executable
counterexample to *unique* key-symbol identification, not a claim that the
trained network actually uses such a pointer. The encoder can in principle
compute a row pointer from the full visible episode; the downstream read can
consume it without retaining the key's token identity in the patched state.

Apply the exact registered128 groups×eight surfaces×three recipients on both
splits. For distinct wrong/deranged donor controls, use the frozen global
permutations and their physical G-row pointers. The deterministic null chooses
a physical row index by SHA-256 of group/surface/recipient. Native clean
predictions are the known public oracle by construction. Score transfer,
exact triples, changed-G non-injection, reverse, marker conditions and35-point
wrong/deranged/random margins with the same no-model TEACH-0015 scorer. Record
all denominators, pointer-position agreement, control rates, source hashes and
visible-data checksums. This exercise cannot satisfy the neural numerical
identity/replay gate, and its success would only falsify the inference
“passing finite transfer proves a symbol-key code.”

If the pointer baseline unexpectedly fails any registered behavioral clause,
record that failure rather than modifying the pointer or threshold. If it
passes, the TEACH-0015 neural label remains the narrower
`PORTABLE-INTERMEDIATE-STATE-SUPPORTED`; a separate cross-order assay must
challenge the row-pointer explanation.
