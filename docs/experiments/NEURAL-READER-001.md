# NEURAL-READER-001: fresh known-key reading with explicit search-error diagnostics

2026-09-30. Prospective reader design developed while the already-frozen paired
source models are training. Some Pliny development scores are already known;
no reserved-author reader panel has been opened or scored. This is not a blind
historical decipherment and does not assume Voynich is Latin or this cipher.

## Question, prior work and assumptions

Can the broader source data and recurrent representation reduce errors in
actual variable-unit readings, and which remaining errors arise from approximate
search rather than the source score preferring an incorrect reading?

The method review in [the search notes](../research/recurrent-reader-search-notes-2026-09-30.md)
covers [Kambhatla et al.2018](https://aclanthology.org/D18-1102.pdf)'s neural
substitution-cipher search and [Huang et al.2017](https://aclanthology.org/D17-1227.pdf)'s
monotone score bounds and beam-limited guarantees. Our supplied Latin keys,
variable one/two-glyph units and geometric stop law differ. Prior Voynich
limitations and source-data assumptions are in
[LATIN-SOURCE-MODEL-001](LATIN-SOURCE-MODEL-001.md). Preserve all earlier reader
failures and the original large statistical arm's context-cap failure. The
compact storage revision is named explicitly as the statistical source.

Assistance: known normalized23-letter Latin alphabet; literal correct key;
one deterministic state, one nonempty one/two-glyph unit per letter; ABCDEF
observations; exact record boundaries; rho=1/225 geometric stop probability.
Six single-glyph units and17distinct double-glyph units imply full observed-glyph
coverage but permit alternate segmentations. The actual source length224 is
**not** given to inference. No length normalization, true-length oracle, beam
retuning, output editing, unknown-key search or natural-language translation.

## Source prerequisites and comparisons

Before any target-author access, all four original recurrent fits must finish,
be separately audited and be frozen along with both compact source fits/audits.
Verify paired initial weight hashes and selected checkpoint archive hashes.
Model selection remains the original Pliny-only argmin. Do not choose a neural
seed or checkpoint using cipher accuracy. Both large-data neural seeds are
primary; small-data neural fits remain reported in the source comparison but
are not extra reader candidates selected after seeing this panel.

Six fixed reading arms:

- Compact statistical source on small data, exact MAP/marginal inference.
- Compact statistical source on large data, exact MAP/marginal inference.
- Large recurrent seed31103, beam32diagnostic and beam128primary.
- Large recurrent seed31109, beam32diagnostic and beam128primary.

The statistical arms use their independently selected tau64 from
LATIN-SOURCE-COMPACT-001, unchanged order12/context-total cutoff4/counts.
The original full source architecture screen remains unavailable because of
its failed arm; this fresh reader question uses the explicit compact revision.
Do not replace a primary beam128reading with a better beam32reading after
seeing either source scores or accuracy. Report such cases as search failures.

## New authors, keys and controls

Use reserved LATIN-SOURCE-001 authors Nepos(phi0588) and Apuleius(phi1212), whose
entire prepared bodies were excluded from source fitting/selection and included
in the earlier64-letter training-overlap protection. Verify original corpus
manifest SHA256fcade8e816fac499284c103808be3b3e7c88e66711cbb6f426be089ec8ecc9c5,
each archive hash, role and alphabet before access. Conservative segmentation,
normalization, catalogue authorship and possible shorter/fuzzy overlap limitations
remain. No further corpus filtering after performance is known.

For each author list candidate224-letter windows beginning at offsets0,512,1024…
within each saved segment, in its saved order. Sample16without replacement with
Python random seeds571031(Nepos),571288(Apuleius). Do not cross segments, join
short segments or redraw on insufficient slots. Record each segmentID/hash,
offset and windowhash; no plaintext in the tracked manifest. The fixed512spacing
makes windows disjoint within a source segment. Two authors, one window each,
are paired under each of16keys:32positive records/7,168letters. This is a fixed
allocation, not a formal claim that the literary passages are independent.

Key i=0..15 uses unchanged familyB generator seed169129+104729*i.
Reject duplicate literal keys within this panel or against the fixed prior
CONFIRM001/SUFFIX001/SUFFIX002seed schedules; no redraw. Each key's positive
pair has a matched pair of **plaintext-permutation** controls, using seed
keyseed+4,000,000+257*authorindex. Shuffle plaintext letters before encoding,
so all23letter counts and encoded record length are preserved and the true
random plaintext is known. These differ from older glyph-permutation nulls.
Every arm reads64records:32positive and32null,384totalreadings across sixarms.

Null edit rates are diagnostics of imposing a language prior on shuffled text.
They are not a language detector, semantic threshold or key-recovery score.
Artificial exhaustive tests separately include unavoidable channel ambiguity,
missing support and deliberately inadequate beam search. No shuffled case may
be dropped because its decoded text is fluent or its result inconvenient.

## Inference and independent checks

Statistical inference uses the existing frozen exact offset×suffix-state
recurrence with a lazy numeric-storage adapter. A separate string-context
backward recurrence recomputes marginal and best scores and node counts;
literal-history scoring replays the returned path. All match within1e-7nats.
The reference count view is a bounded cache into the same audited raw counts,
not the production numeric-state transition table.

Neural beam search preserves distinct full-history hidden/cell states. Primary
width128, diagnostic32; max200,000expanded prefixes/record. Channel-only suffix
reachability pruning is exact. Every beam-pruned prefix gets a completion-score
upper bound, retained in the output. Numerical bound flags are **not** claimed
as interval-arithmetic optimality proofs; bounds may remain inconclusive.
Full-forward source scoring of each returned path must agree with incremental
scoring within1e-4nats per returned letter. Exact re-encoding is mandatory for
every returned path. No answer archive is loaded during prediction.

After all384predictions are written, frozen, pushed and remotely verified,
evaluation opens answers. Every true path must re-encode exactly. All384edit
counts are checked using both the prior rolling-row implementation and a
separate full dynamic-programming grid. For each neural seed, directly score
the64true paths after freezing predictions and compare with beam128andbeam32
path scores. If a true path scores higher than the primary returned path beyond
1e-4*max(lengths), that demonstrates a missed better candidate. If an incorrect
returned path beats the true path beyond that tolerance, the model objective
prefers at least one incorrect alternative. Neither observation alone proves
information-theoretic impossibility, an exact posterior or optimal edit-risk
decision making. Do not use these diagnostics to change this panel's source.

## Primary gates and complete reporting

For **each** neural seed's beam128arm, all three must hold on positive records:

1. Total edit rate≤2% across7,168letters.
2. Every key's paired448letters has edit rate≤5%.
3. At least25%fewer edits than the large compact statistical baseline. If that
   baseline has zero edits, require the neural arm also to have zero; report the
   relative reduction as undefined and both-perfect, not a fictitious25%gain.

Require both seeds to pass for the joint reader gate. Report per-key/per-author
record errors, exact records, null errors, all source scores/search diagnostics,
both beam widths and both statistical data sizes. No posthoc winner-only result.
These thresholds continue the earlier reader standards; selection loss is not
a substitute. A pass establishes conditional supplied-key competence in this
narrow family and source population, not unknown-key recovery or decipherment.
Any failed/capped/incomplete stage stays visible and cannot be called a pass.

## Stages, costs and stop conditions

Freeze/push/verify all code/tests/this protocol, four neural result/audit records,
compact result/audit records and source references before prepare. The runner
checks every source prerequisite before reserved author access. After prepare,
publish/verify the panelmanifest before predict; after predict, publish/verify
all predictions before evaluate. Use exclusive stage markers and outputs. No
restart, limit increase, selective rerun or model changes within this experiment.

With `PYTHONPATH=.:src`, BLASthread variables1:

```
.venv/bin/python scripts/run_neural_reader001.py prepare --freeze REV
.venv/bin/python scripts/run_neural_reader001.py predict --freeze PANEL_REV
.venv/bin/python scripts/run_neural_reader001.py evaluate --freeze PREDICTION_REV
```

Prepare300wall/300CPUseconds; predict3,600wall/3,600CPUseconds; evaluate900wall/
1,200CPUseconds. Each statistical lattice and separate reference≤500,000nodes;
each neural record≤200,000expanded prefixes. Failure stops the stage and leaves
partial archives, with no qualification result. One GPUprocess, two PyTorch
host threads; no concurrent training during prediction/evaluation. Check8GiB
MPSdriver allocation per decoded record;8GiBhostmemory planning. Zero paidAPI/
cloud use, local energy unmetered. Actual reader throughput has not yet been
benchmarked with trained weights; these are bounded caps, not promised runtimes.
Raw target text, checkpoints and complete readings remain ignored with hashes.
Record costs/failures and update notebook/memory before checkpoint publication.
