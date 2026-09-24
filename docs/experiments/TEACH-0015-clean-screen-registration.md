# TEACH-0015 clean competence screen before causal intervention

**Status:** prospective registration written while the TEACH-0014 v3 optimizer
campaign is running, before its final neural outcomes are inspected. This is
the entry stage of `TEACH-0015-mechanism-preregistration.md`, not a mechanism
result. No donor patch, VJP, PCA, J-space, neuron or token-similarity claim can
be made from clean answers alone.

The frozen TEACH-0015 discovery/confirmation manifests have128 groups and
66 cells per group, audited separately in
`TEACH-0015-suite-registration.md`. The screening population is all8,448
episodes of **each** split, in manifest order. A source-matched random-weight
MPS benchmark must pass before checkpoint evaluation. The screen will only
load TEACH-0014 v3 checkpoints after its complete no-model artifact audit and
sampled checkpoint/gate numerical replay pass. The shuffled-label null and
public-row oracle must pass their registered validity gates. Each screened raw
arm must separately pass the TEACH-0014 absolute two-hop behavior and parser
gates in **both seeds**: answer-only, causal-supervised, or grammar-supervised
edge auxiliary. Screen every eligible arm, naming its supervision; do not
select the best arm after seeing fresh TEACH-0015 scores. If none qualify,
record `NOT ENTERED: PRIMARY COMPETENCE` without loading any raw checkpoint.

The clean metrics and denominators are fixed here before model results:

- Composed item accuracy: all48 composed cells per group,6,144 per split.
- Exact three-recipient accuracy: all `(F, distractor, marker, order)` triples
  of G0/G1/G2 outputs,16 triples per group,2,048 per split. All three answers
  must be correct.
- Exact marked/marker-free accuracy: all `(F,G,distractor,order)` pairs,
  24 pairs per group,3,072 per split. Both renderings must be correct.
- First-hop, direct and copy item accuracy: all six F/G auxiliary cells for
  each task per group,768 episodes per task/split.

The confirmation competence gate is composed≥90%, recipient triples≥80%,
marked/free pairs≥80%, first-hop≥95%, direct≥95%, and copy≥98%, separately
for **both seeds of the same named arm**. Discovery scores are descriptive and
cannot substitute for confirmation. If any criterion fails, that arm is
`INCONCLUSIVE: FRESH-PANEL COMPETENCE`; its finite patch and geometry assay
cannot receive a confirmatory mechanism label. The complete denominator and
group-bootstrap uncertainty must be reported for every metric/arm/seed.
The intervals use4,000 percentile resamples of the128 logical groups with
fixed `TEACH-0015-clean-bootstrap` seeds; they do not change the gates.

Archive every render ID, prediction and target, plus three full2,064-logit
vectors per split/arm/seed (first, middle, final item), source/checkpoint
hashes, exact manifest hashes, timings and sampled MPS allocation. A no-model
auditor reconstructs the full66-cell groups and every score from the visible
stream; a separate checkpoint replay recomputes the sampled logits on CPU.
No competence label is issued before both checks pass. The screen's benchmark
uses random weights on representative full batches and a conservative
projection of `1.75 × median timed batch × 528 batches per arm/seed × all
three candidate arms × two seeds + 300s`. Six warmup and24 timed batches are
measured. The proposed hard caps are1h wall time,12GiB sampled MPS
allocation and1GiB total screen artifacts; if the source-matched benchmark
does not fit, stop and amend **before** model evaluation. No paid API or
manuscript text is involved.
