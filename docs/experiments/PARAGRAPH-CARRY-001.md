# PARAGRAPH-CARRY-001 — transported terminal dependence across paragraph lines

Status: prospective preparation, 2026-10-10 PDT. Root must commit/push and verify
the exact remote freeze before ONE producer and ONE full arithmetic audit.
No scientific manuscript or registered control scores have been computed.

The question is whether a fixed, training-fitted **observable terminal-unit
effect** transports across consecutive lines and diminishes at paragraph starts.
It is not a plaintext decoder, a complete ciphertext likelihood, a hidden-state
identification test, or a causal intervention. A paragraph-level common cause
can produce the same pattern without a terminal-driven state update.

## Inputs and isolation

The runner `scripts/run_paragraph_carry001.py` freezes fourteen paths in `PATHS`.
Tracked code/manifests must match `git show FREEZE:path`; ignored raw/processed
sources must retain their explicit `EXPECTED` SHA256 values. They are project
ZL3b raw `bf5b6d4a...`, GC2a raw `b09570cb...`, ZL train JSONL `49618c7b...`,
and physical-leaf assignment `9fc80cb4...`; full hashes reside in the code.
No validation JSONL or final-test JSONL is opened. GC skips nontraining page
bodies before interpreting glyphs/tags. Canonical `data.leaf_id` keeps the
connected f85/f86/Ros sheet together.

Require whole clean `P0` loci with locator `@`, `+` or `*`, at least two certain words, known hand1–5,
no uncertain separator/alternative/unreadable/rare glyph/drawing interruption
or hand-change annotation. ZL uses six longest-first basic EVA compounds;
GC printable v101 glyphs are independent atomic readings. Exclude other layout
types, malformed marker placement, duplicate IDs, conflicting metadata or
paragraph flags. Start/end markers must be unique and at the locus edges.
Known paragraph depth advances through all paragraph loci, including excluded
loci; it stays unknown until a start marker. Match eligible locus IDs, context
(illustration, Currier, hand), start/end and depth across readings, but do not
force word segmentation or glyph correspondence to agree.

Cross-line events require same page, canonical physical leaf, context, and
exactly consecutive locus numbers, with successor locator `+` or `*`;
previous-end equals next-start. Both true
means paragraph break, both false means continuation. No interpolation across
missing/excluded lines. All exclusions and matched support are reported.
The [primary IVTFF specification](https://www.voynich.nu/software/ivtt/IVTFF_format.pdf)
§6.4 Table8 defines `@` as unspecified/unrelated position, `+` as below the
previous locus, `*` as the next lower line at the left margin, and `=` as the
same line. §7.2 ties position to the preceding locus number. Therefore the
other locators are excluded, and `@` cannot itself establish a junction.
Every selected locus receives one retained/excluded category; editorial
comments are removed before GC uncertainty and glyph interpretation.

Before freeze, an unscored input-availability check exposed an incorrect
`@`-only adapter that retained starts and no junctions. The schema repair above
and positive `@`/`+`/`*`/`;G` fixtures precede all scientific scoring; they do
not change the model, decision thresholds or matched-null rules.
The repaired unscored extraction retained 1,237 ZL and 1,830 GC lines;
828 matched lines span 76 TRAIN physical leaves and yield 247 continuation
and 33 paragraph junctions in each reading. ZL's 4,153 loci account exactly
as 1,237 retained, 1,025 other layouts, 1,889 uncertain/annotated and two
invalid word shapes. GC's 4,139 loci account as 1,830 retained, 1,011 other
layouts, 1,297 uncertain/annotated and one invalid word shape. Clean matched
IDs have 25 metadata/paragraph/depth/locator conflicts. No likelihood,
bootstrap, permutation or scientific control was evaluated for this check.

Five fixed physical-leaf folds come from SHA256-ranked unique training leaves,
salt `paragraph-carry-training-leaf-v1-20261010`, rank modulo5. Other folds fit
each model; all junctions on its held-out leaves are predictions. These are
development cross-fits on previously used training material, not pristine
confirmation or scribe/section/quire-disjoint transfer.

## Fixed normalized emission rule

For context c, estimate within-line next-initial marginal M_c and conditional
E_c(i|last), with +1 global smoothing, context concentration50 and edge
concentration20. Separately estimate line-initial B_c,k(i) by paragraph/start
role k, using the same global/context smoothing. There is no cross-line fitting,
alpha selection, learned segmentation, hidden state, neural training, or
per-folio key. The two candidate emissions are

`reset(i)=B_c,k(i)` and
`transport(i|last)=B_c,k(i)*E_c(i|last)/M_c(i) / sum_j B_c,k(j)*E_c(j|last)/M_c(j)`.

Train-only target vocabulary plus an explicit unknown category; maximum513.
All probabilities are positive and normalized. The within-line odds' transport
to a differently distributed line-start role is an explicit model assumption.
The one-step state is an observed previous terminal, not inferred historical
memory. Scores are conditional initial-unit log2 gains, not whole-text evidence.

## Allocation and fixed decisions

26 complete cells: two manuscript readings plus 24 synthetic documents:
IID-terminal seeds97601/97609 and Markov-terminal seeds97611/97619, each with
`paragraph_carry`, `line_reset`, `always_carry`, `copy_mutate`, `iid`, and
`paragraph_common_cause`. Each synthetic has40 independent leaf identifiers,
24 lines/leaf, six four-glyph words/line, paragraphs of four lines, eight
symbols, and a document-wide fresh relation permutation. These are two
artificial terminal-source families; neither is a new natural-language family
or an unknown historical cipher. Controls reveal reset locations only to
evaluation, never to model selection.

Carry controls insert the keyed previous-terminal→initial relation with .9
probability; line reset removes it at every line; always-carry retains it at
paragraph boundaries. Copy-mutate copies a prior complete word .8 of the time
and changes its second unit; IID has no planted terminal relation. Common-cause
controls independently concentrate initial and terminal emissions on a shared
paragraph style at .8 each. A positive transport classification for this last
family is ALLOWED and demonstrates nonidentification; it is not a failed
negative to be repaired. No claim that general copying/topics are ruled out.

Report continuation mean a, paragraph mean b, and a−b. Conditional clustered
uncertainty uses1000 physical-leaf bootstrap draws over the frozen cross-fitted
gains, WITHOUT refitting training models. Percentiles are fixed integer indices.
Support requires100 continuation events,25 paragraph events,≥5 leaves.
`paragraph_carry` requires a≥.02 and its lower interval>0, a−b≥.05 and its
lower interval>0, b≤0 and its upper interval<.02. `line_reset_compatible`
requires a≤0 and its upper interval<.02. `unbroken_carry` requires a,b≥.02
and both lower intervals>0. Otherwise `undetermined`; no gate retuning.

100 fixed permutations reassign source terminal units only within held-out
leaf/context/kind/depth-bin/left-length-bin/right-length-bin. Depth caps at3;
line-word-count bins are floor(n/4) capped at3. Paragraph identity/topic is NOT
conditioned away. Report total, permutable, and effectively mutable events;
singletons and identical-terminal strata remain unchanged. Monte Carlo
one-sided p uses `(1+#null contrasts>=observed)/(101)`.

Qualification requires all twelve planted carry/reset/always-carry documents
classified correctly and all eight fixed copy-mutate/IID controls avoid
`paragraph_carry`. Common-cause classifications are separately reported.
The manuscript transport signal additionally requires BOTH readings to meet
paragraph classification, permutation p≤.05, and ≥25% effectively mutable
events. Qualification failure preserves all manuscript scores but bars a
qualified transport claim. `causal_carry_identified` and `decipherment` are
always false. A negative result rejects this fixed transported predictor under
this data/representation; it does not rule out language or stateful ciphers.

## Resources, execution and audit

ONE producer:600wall/550CPU seconds,1GiB process RSS,16MiB ignored compressed
ledger, CPU1/$0. ONE auditor uses the same time/RSS bounds. SIGALRM/SIGPROF
raise exceptions inside long cells; per-cell resource checks supplement them.
Expected runtime seconds to several minutes per stage; no real-source pilot.
Exclusive directories/results and failure receipts forbid retries/resumption.
Root handles publication; both results remain required before closing.

`PYTHONPATH=.:src .venv/bin/python scripts/run_paragraph_carry001.py --freeze SHA`

`PYTHONPATH=.:src .venv/bin/python scripts/audit_paragraph_carry001.py`

The auditor independently recounts training distributions, reconstructs
junctions/manifests and evaluates transported log odds for every event and
every permutation. It verifies input/result/archive bindings, resource totals,
all allocations, fold isolation, counts, summaries and gates. Source parsers,
synthetic generator, hash fold assignment, bootstrap and gate helpers are
shared and disclosed; this is alternate same-author arithmetic, not independent
paleography or independent cryptanalysis. Rank p preserves the original audited
saved doubles' tie rule; arithmetic tolerance2e−11bits.

Artificial tests cover normalization/unknowns, alternative full scores,
consecutive/paragraph/hand/leaf exclusions, no nontrain glyph parsing, canonical
Ros grouping, corrupted manifests, full temporary producer/auditor flow and
exclusive outputs. Fixture seeds9901/9909/9921 are distinct from registered
control seeds; shortened synthetic shapes/permutation count in the temporary
pipeline are preparation checks, not a scientific pilot or calibration pass.

Prior evidence and identification limits are in the
[research note](../research/paragraph-carry-mechanism-2026-10-10.md).
