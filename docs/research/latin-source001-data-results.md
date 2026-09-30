# A larger Latin source corpus is prepared and independently replayed

2026-09-30. The new training pool has **6,817,650 normalized letters across
12 catalogue author groups**, versus a matched100,000-letter subset. That is
68.1765times as much source text. No language model has yet been trained on it;
more data is an intervention to test, not an observed improvement in decipherment.

## Frozen preparation and complete data allocation

The [source policy](latin-source001-data-plan.md), parser, tests, acquisition
inventory and notices were pushed/remotely verified at
`a06ffc6cc6dca8fefc7f2141184bd407d5556d6b` before the one preparation.
The upstream Perseus revision is`cc843833e101992ba54549005273cfe61a33504b`.
All106Latin XML files,20,435,456bytes, match pinned Git blobs and SHA256s.
Raw source and derived literary text remain outside Git.

| Catalogue author group | Role | Retained letters |
| --- | --- | ---: |
| Plautus | Training |783,572|
| Terence | Training |247,530|
| Caesar | Training |493,762|
| Lucretius | Training |264,555|
| Virgil | Training |476,759|
| Celsus | Training |368,383|
| Horace | Training |244,555|
| Ovid | Training |1,194,214|
| Quintilian | Training |883,443|
| Seneca | Training |1,283,671|
| Vitruvius | Training |209,258|
| Suetonius | Training |367,948|
| **Training total** | **77 works,3,487segments** | **6,817,650** |
| Pliny the Younger | Source model selection |359,276|
| Nepos | Reserved future reader author |113,799|
| Apuleius | Reserved future reader author |353,856|

Cicero, Sallust and Tacitus folders are excluded. Author groups do not settle
historical attribution, shared authorship or quotation provenance. Reserved
text was mechanically parsed and screened; no cipher panel or fitted model has
used it. No fresh performance claim is made about previously exposed evaluations.

The small corpus takes50,000remaining letters each from Caesar and Virgil,
with the same parser and edition choices. It is an exact prefix subset of the
large corpus, including preserved segment boundaries. It differs from the
older Gutenberg source; future comparisons must use matched source data rather
than silently treating that older baseline as identical.

## What was excluded and what was verified

The parser quarantined3,572content units containing uncertainty, foreign text,
editorial insertions/deletions, alternate readings or unsupported alphabetic
content. Several reasons may apply to one unit. It omitted headings, notes,
speaker labels and stage directions. It retained122,641units before duplication
filtering. Context resets prevent the removal of a unit from creating an
artificial adjacent passage. XML paths, exact unit lengths, hashes and a complete
extraction ledger preserve traceability. No uncertain text was guessed.

The64-letter overlap screen removed five Quintilian units totaling1,454letters.
No other training author lost text to that screen, and Pliny needed no removal.
All remaining training text has zero64-letter overlap with reserved-author text,
clean Pliny, and the prior50kCicero/Sallust/Tacitus allocations, within recorded
boundaries. An additional independent BLAKE2b-window scan agrees. Zero hash
intersection guarantees no identical window at this width; it does not prove
absence of short overlap, paraphrase or literary dependence.

A separate audit walked the original XML node paths and used the pre-existing
normalization implementation to reconstruct all3,982retained segments and
122,636unit references across all15groups. Every reconstructed string, author
total, source hash, per-unit length and100k-subset identity matched. It also
replayed the full extraction/removal archive. No parser failure, relaxed rule,
redrawn source or preparation rerun occurred. This is an algorithmic audit by
the same research agent, not replication by an independent researcher.

Full regression before preparation: **1,930tests plus23subtests passed**,
eight skipped,131.80seconds. New source/parser code is lint-clean; the same five
unrelated pre-existing full-tree findings remain. No previous experiment
implementation or failure outcome was changed.

## Model size and local compute feasibility

A separate hardware-only probe used random tokens, not Latin or cipher answers.
Two-layer LSTMs with96-dimensional embeddings, batch16×512characters, were
compared after two warmups and eight timed updates:

| Hidden width | Parameters | Mean training step | Driver allocated memory |
| --- | ---: | ---: | ---: |
|512|3,364,631|0.079217seconds|1,395,523,584bytes|
|768|7,405,079|0.121287seconds|1,773,010,944bytes|

CPU/MPS forward logits differ by at most5.22e-8on the recorded small fixture.
The larger candidate passes the prewritten timing/memory rule. A6,000-update
fit projects12.1minutes; four fits project48.5minutes before validation, saving,
initialization and thermal variation. This is a planning estimate, not measured
language-training runtime, convergence or accuracy. Actual fitting requires a
separate frozen exposure/seed/selection protocol and hard limits.

A512-character window sampler would also impose a data choice: the current
small/large corpora contain97,850/6,497,939letters in segments at least512long.
It must either account for shorter segments explicitly or restrict the
statistical baseline to the same eligible text. Do not claim a matched-data
comparison while silently dropping those segments in only one model.

## Costs, artifacts and next state

Preparation5.390289CPU/5.442316wallseconds; XML replay2.346544CPU/2.456043wall;
independent overlap scan2.400823CPU/2.403204wall. No paid service. Bulk source,
derived author archives and10.64MBcompressed unit ledger are ignored.

[Corpus manifest](../../results/LATIN-SOURCE-001/corpus.json),
[XML replay](../../results/LATIN-SOURCE-001/corpus_audit.json),
[overlap replay](../../results/LATIN-SOURCE-001/overlap_audit.json),
[hardware measurements](../../results/LATIN-SOURCE-001/hardware.json).
Corpus-manifest SHA256`fcade8e816fac499284c103808be3b3e7c88e66711cbb6f426be089ec8ecc9c5`;
extraction-ledger SHA256`0b1a1eed995021255360ec051dace34851164dbe03509e628e0bde30ca63569e`.
Commands and source freeze are recorded in NB-246–247.

Next run a paired small/large data comparison of a statistical source and a
recurrent source, selecting only on Pliny. Keep candidate search error separate
from source preference when returning to encrypted text. This checkpoint
removes the tiny-data restriction and establishes affordable local compute;
it does not establish a stronger reader, a Voynich language, or a translation.
