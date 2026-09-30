# SUFFIX-READER-002: learn context weighting and isolate its longer-history contribution

2026-09-30. Registered prospective development comparison before any new
calibration or cipher score. No previous FAIL is rerun or reclassified.
Correct dictionaries are supplied. This is an intermediate test of reading
competence; blind-key discovery and actual manuscript decipherment remain goals.

## Reason and source review

SUFFIX001 added histories through12letters but failed all three reading gates.
Its fixed common mass256 gave depths4–12 only9.33% mean mixture weight on that
exposed panel. That observation motivates this hypothesis; it does not prove
that increasing those weights improves accuracy. Test learnable masses, and
ablate the resulting model back to three-letter memory to separate short-context
recalibration from the benefit of longer patterns.

Reviewed [Chen and Goodman1996](https://aclanthology.org/P96-1041.pdf), sections2–4:
smoothing performance depends on data/order, and interpolation estimates need
data disjoint from the counts. Their English word-model comparisons do not
establish Latin character recovery. Reviewed/reused
[MacKay and Peto1995](https://www.cs.toronto.edu/pub/gh/MacKay%2BPeto-1995.pdf)
model/Dirichlet discussion: our count interpolation is related in form, but
we optimize held-out predictive loss, not integrated Bayesian evidence.
[Kambhatla et al.2018](https://aclanthology.org/D18-1102.pdf) supports examining
stronger source scoring in substitution decipherment, under different training
scale and approximate beam search. The [preceding review](SUFFIX-READER-001.md)
covers Hauer/Kondrak's Voynich application and why language scoring is insufficient
to identify a language or reading. No Voynich-specific success is inferred.

## Model, gradients and controlled comparison

Keep the same prefix/suffix-closed count inventory, root add-one-half law,
exact finite-state source transitions and unpruned unit decoder. For depthd,
use positive mass`tau_d=exp(theta_d)`:

`p_d=(count(hc)+tau_d*p_(d-1))/(count(h)+tau_d)`.

Unobserved histories exactly back off. With a fixed full-support source and
nonempty deterministic units, all previous state-sufficiency and exact-MAP
arguments still hold. The twelve masses change how observed histories are
weighted, not the states or dictionary. No adaptive changes occur while reading
ciphertext. This is not an attention model, neural feature discovery, PPM or
full Bayesian parameter integration.

For a calibration position, the local derivative is
`dp_d/dtheta_d=[tau_d/(count(h)+tau_d)]*(p_(d-1)-p_d)`.
The derivative with respect to lower-level predictions is
`tau_d/(count(h)+tau_d)`. Back-propagate the mean negative natural-log loss from
the last level; independently test against automatic differentiation and central
finite differences. Optimize only source calibration text.

Four fixed decoder arms, all on the same newly generated32records:

1. **fixed3:** original three-letter source, mass256 at each depth.
2. **fixed12:** SUFFIX001's twelve-letter source, common mass256; primary baseline.
3. **calibrated12:** source-only selected twelve depth masses, all twelve levels.
4. **calibrated3:** same selected first three masses, all longer histories removed.

The last arm is a direct model ablation, with no separate fitting or model
selection. It distinguishes gains due to longer memory from gains in short
context weighting. This says nothing about a historical cipher mechanism.

## Fixed data, optimization and source-only selection

Use only the pinned50,000-character Caesar/Virgil/Cicero corpora and existing
normalization. Preserve body boundaries, alphabet`abcdefghiklmnopqrstuxyz` and
all original checksums. Source counts use Caesar's50,000letters. Gradient
calibration uses Virgil[0,40000), split at its real body boundary27839.
Checkpoint selection uses Virgil[40000,50000), resetting its start context.
No context crosses a body or the calibration/selection boundary. Virgil is
reused development source material, not a newly untouched validation author.

Fixed maximum depth12; four deterministic initial common masses4,64,256,1024.
For each,500full-batch Adam updates at learning rate.05, beta1=.9, beta2=.999,
epsilon1e-8. Optimize log masses and project each into[log(.25),log(4096)].
No adaptive stopping, new restart, budget extension or altered learning rate.
Keep all501iterates per restart, including initial states, in a hashed archive.
Only steps0,100,200,300,400,500 are eligible for selection:24candidates, plus
the fixed256vector baseline candidate. Choose minimum selection bits/character;
ties prefer baseline, then lower start index, then earlier step. No global
optimum or optimizer-convergence claim is required or implied.

Refit the counts on Caesar+Virgil,100,000letters, holding selected masses fixed.
These counts must exactly equal the frozen SUFFIX001 counts archive. Reuse that
archive after independent reconstruction instead of storing duplicate data.
All four arms share these data; the three-letter arms restrict the same counts
to shorter histories. Do not use Cicero for counts, gradients, hyperparameters,
checkpoint selection or parameter rescaling. Save calibration before opening
Cicero to construct the new panel.

## New known-key allocation and ordered execution

Sixteen Bkeys, seeds119129+104729*i, i=0..15. Same generator: six singleton units
plus17distinct digrams over ABCDEF, permuted onto23letters. Each gets two
224-letter Cicero windows at20000+512*i and20256+512*i. All windows lie inside
the second body[18435,35839); no overlap with DEV001 or SUFFIX001 cipher windows.
Verify against the saved earlier offsets. Check new keys against SUFFIX001 and
against one another. No redraws or performance-based window selection.

This is a new scoring allocation in already processed text, not sixteen
independent author samples. Public seeds and known-key generation provide
procedural isolation, not cryptographic secrecy. It does not reuse the earlier
failed panels for testing. Mean length224 is reflected in the unchanged
geometric stop probability1/225; exact plaintext length is not imposed.

Publish/remote-verify source/protocol/tests before preparation. Publish selected
parameters, source-count binding and whole panel manifest before prediction.
Predict all four arms on all32records, retain per-case artifacts and failures.
Publish/remote-verify all predictions before one evaluation. Do not choose a
different source or subset after seeing readings. Accuracy uses unchanged
integer gates: calibrated12 must reduce edits by at least25% versusfixed12,
achieve overallCER≤2%, and achieve every-keyCER≤5%. All16keys must be present.
Unsupported readings retain deletion penalties and additionally indicate an
implementation failure for these supplied full-support dictionaries.

Separately report whether calibrated12 has strictly fewer total edits than
calibrated3, plus per-key contrasts. This is the predeclared longer-context
ablation outcome, not a p-value or an additional way to pass the reader gate.
Fixed3 is descriptive; it cannot replace the stronger fixed12 baseline if
the primary comparison fails. Record exact readings, lengths and all errors.
No language-identification, structured-null-rejection or blind-search gate is
claimed. Historical transcription and uncertain learned keys remain untested.

## Validation, resources and interpretation

Tests must cover gradient equivalence to autograd/finite differences, uniform
mass regression, source-feature/table agreement, rational full-plaintext
enumeration, bounded deterministic optimization, record boundaries, exact
reverse inference, cap failures and fixed nonoverlapping allocation. Replay
all128marginal/MAP scores and returned-path scores with a separately written
reverse calculation; max numerical tolerance1e-7. Independently replay all128
edit distances. New reference code is root-authored, not a second researcher.
Check all source counts by the existing separate fixed-length window counter.

One CPU thread. Preparation1200CPU/1800wallseconds; prediction1800CPU/2700wall;
evaluation120CPU/180wall. At most3120phase CPU seconds under hard caps; zero paid
services/downloads.8GiBplanning memory,1.2million context cap and500,000 lattice
nodes per record; cap failures stop without pruning or rerunning. Save logs,
full calibration trace and predictions outside Git with hashes; track compact
selection, results and provenance. Run full regression, update notebook/memory,
and publish a verified checkpoint including negative findings.

If the reader passes but the ablation does not improve, gains are not evidence
that longer context was the useful change. If a source-only loss improves while
reading gates fail, retain that distinction. No result here establishes a
Voynich language, mapping, word boundary or meaning.
