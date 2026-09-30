# A bounded search over readings from an uncertain global key

Working derivation and artificial implementation,2026-09-30. Motivated by both
SHARED-KEY-BENCH-001 uniform1,197-key joint-state graphs exceeding500kstates. No
empirical recovery result or claim that this model describes Voynich.

## Prior work and scope

[Huang and Chiang2005](https://aclanthology.org/W05-1506.pdf) develop lazy k-best
derivation methods and discuss reranking and summing equivalent derivations.
Their hypergraph setting is broader than the acyclic lattices here. We use
ordinary prefix A* with an exact backward completion heuristic, not an
implementation of all their parsing algorithms.

[Mohri and Riley's n-best strings work](https://cs.nyu.edu/~mohri/pub/nbest.pdf)
distinguishes weights of strings from individual paths. That distinction matters:
one text supported by many keys can beat every individual key's favorite.
Their main construction uses minimum-path tropical weights; the summed-key
probability objective here differs.
Neither source supplies evidence for our empirical bank or historical language.
The earlier [shared-key derivation](shared-key-mixture-decoding-2026-09-30.md)
and its Voynich/decipherment review still define the probabilistic model.

## Bound derived for this finite-bank model

Fix keys and fitting-only normalized weights w_i. A text tuple x contains one
text for every observed record, with source probability Q(x), including separate
geometric stops and resets. Key i supports it exactly when C_i(x)=y. Then

```
S(x) = Q(x) * sum_i w_i * 1{C_i(x)=y}.
```

For every key i, enumerate its k best compatible *text tuples*, ordered by Q.
Let L_i be that list and u_i the probability of its next tuple, or0if exhausted.
Let L be the union of all positive-weight key lists. Every x outside L satisfies

```
Q(x) * 1{C_i(x)=y} <= u_i  for every i,
therefore S(x) <= U = sum_i w_i u_i.
```

Re-score **every** tuple in L against **every** bank key, not just the keys that
proposed it. Let B be the best resulting S. The global optimum is in
`[B, max(B,U)]`. If B≥U, the selected tuple is a MAP solution within this fixed
bank. It need not be unique. If the interval is open, keep the candidate and
bound rather than claiming exact MAP. Low positive weights remain present;
keys are not silently removed because they are inconvenient or unlikely.

The implementation uses floating log probabilities. It reports separation only
when B exceeds U by an explicit tolerance, or every relevant list is exhausted.
This is an algorithmic bound with floating numerical checks, not a directed
rounding/interval-arithmetic proof. Even a certified bank optimum may be the
wrong plaintext, and the bank may omit all correct mappings.

## Per-key enumeration and shared consistency

A per-record state is observed offset plus sufficient source context. Every
source letter emits a nonempty literal unit, so offsets increase and the graph
is acyclic. Forward reachability followed by backward sums/maxima gives exact
evidence and the best completion score at every state. Prefix A* orders complete
readings using prefix score plus that completion heuristic. Text prefixes remain
distinct in the queue; merging their Viterbi values would lose k-best paths.
For a fixed deterministic key, one plaintext has exactly one path, even if
different letters share an observed unit. No duplicate-path mass is invented.

Each record supplies k+1 readings. A heap over Cartesian indices produces the
k+1 best tuples for the **same** key. A tuple using a per-record index beyond k+1
has at least k+1 alternatives of no lower probability with the other coordinates
fixed, so it cannot improve the needed next-score threshold. Ties may affect
which equal-score tuples are listed; the bound remains valid.

For candidate re-scoring, a compact literal compatibility mask retains only keys
that explain every letter of every record. The source resets between records;
the key does not. Evidence is independently `sum_i w_i * product_r Z_i(y_r)`.
This avoids creating the full joint compatibility-state graph but may still be
expensive: each key needs exact record lattices and enough ranked prefixes, and
the bound may remain loose. Explicit node/edge/prefix limits raise failure; no
beam replacement, smaller-bank retry or assumption of missing support.

## Artificial validation before empirical use

`src/voynich/kbest_suffix.py` is checked against complete tiny-string enumeration,
the original exact per-key decoder and the full shared-key decoder. Tests cover
orders0–3, empty records, ties, unsupported keys, true zero probabilities, tiny
key masses, geometric stopping, record resets, all three caps and shared-key
consistency. An adversarial example has a wrong k1 mixture candidate whose bound
stays open; k2 finds the correct summed-key winner and exhausts the alternatives.
Random panels compare exact evidence, optimal-score bounds and every returned
global key's re-encoding. These tests establish neither full-size feasibility nor
empirical accuracy. A separate protocol must freeze k, resources, bank weights
and evaluation before applying the method to exposed or fresh passages.
