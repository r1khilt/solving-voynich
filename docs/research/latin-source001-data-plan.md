# LATIN-SOURCE-001: broader source evidence before a stronger reader

2026-09-30. Corpus preparation for a controlled data-versus-model comparison.
No language identification or Voynich reading is claimed. The previous goal
turn made progress: it measured context benefit and ruled out true length as a
sufficient repair. The next comparison needs more than nearby parameter tuning.

## Motivation and prior method review

The current source uses100,000normalized letters from Caesar/Virgil. Its known-key
reader and a true-length intervention still fail. A larger source corpus and
a character neural model are now justified candidates, with separate controls
for data and architecture. A bigger model's success remains unproven.

[Kambhatla et al.2018](https://aclanthology.org/D18-1102.pdf), sections2–4, uses
a4096-unit character mLSTM, English Gigaword plus Zodiac-authored letters,
and beam-based key search. Its substitution/homophonic setup differs from our
variable-unit Latin problem; both data scale and author-specific assistance
differ. It motivates a normalized autoregressive source, not copying a reported
accuracy or assuming exact inference survives a neural history representation.
[Latin BERT](https://arxiv.org/abs/2009.10053), abstract/model description, shows
the availability of much broader Latin training resources. Its masked-token
objectives and classical-philology tasks are not a normalized causal string
model or a cipher-reading qualification. We do not import its checkpoint or
unknown training/test overlap. [Hauer/Kondrak2016](https://aclanthology.org/Q16-1006.pdf)
and the existing source reviews document why language scores on Voynich do not
establish a language or decipherment.

The existing local transformer has causal attention and explicit activation
intervention sites, so it is a candidate for a stronger source followed by
causal checks. Training/selection, source-only loss and constrained cipher
inference require a separate frozen experiment after corpus preparation and
hardware timing. This document does not authorize unlimited training or set
cipher success criteria retrospectively.

## Fixed acquisition and rights record

Official [Perseus canonical Latin collection](https://github.com/PerseusDL/canonical-latinLit),
commit`cc843833e101992ba54549005273cfe61a33504b`, tree
`cbe8884218397037e45559cff9b53208c34c5049`. The complete tree inventory is not
truncated. Within fifteen fixed author groups, select the lexicographically
first Latin XML version per work. No English translations or second version
of the same work. This yields106files/20,435,456bytes. All downloaded bytes
match both their expected size and pinned Git blob SHA1, with SHA256 retained.
Acquisition cap40MBtotal/6MBperfile/fourconnections/60seconds per request;
zero paid services. Metadata acquisition was exploratory preparation before
this parser freeze, with no trained model or reading score.

The [repository notice](https://github.com/PerseusDL/canonical-latinLit/blob/cc843833e101992ba54549005273cfe61a33504b/README.md)
and [license](https://github.com/PerseusDL/canonical-latinLit/blob/cc843833e101992ba54549005273cfe61a33504b/license.md)
give CC-BY-SA4.0 by default, subject to per-component exceptions. Inspected all
106headers:31have an explicit matching availability notice;75use the repository
default. Header keyword review found editorial history and public-domain print
source descriptions, with no conflicting license notice. Retain original headers,
metadata, attribution and license; raw XML and derived bulk text remain ignored.
No bulk source or modified corpus is redistributed in Git. This is a source
notice audit, not a guarantee that edition/transcription metadata are perfect.

## Author groups and isolation

Training: Plautus(phi0119), Terence(phi0134), Caesar(phi0448), Lucretius(phi0550),
Virgil(phi0690), Celsus(phi0836), Horace(phi0893), Ovid(phi0959),
Quintilian(phi1002), Seneca(phi1017), Vitruvius(phi1056), Suetonius(phi1348).
Selection author: Pliny the Younger(phi1318).
Reserved future reading authors: Nepos(phi0588), Apuleius(phi1212).
Exclude all files under earlier evaluation authors Cicero(phi0474),
Sallust(phi0631), Tacitus(phi1351) from training. Names were read from pinned
author metadata: phi0836 is Celsus, not Livy. These are catalogue author-group
splits, not adjudications of disputed authorship, coauthorship or quoted text.

The reserved authors were downloaded and may be mechanically parsed/deduplicated;
their plaintext has not been used to fit/select a model or generate a cipher
panel. The researcher-facing format inspection read only training/selection
body excerpts; reserved-body orphan checks concealed text. Procedural separation
is not cryptographic secrecy or a claim the works are unknown to an LLM.

## Frozen parsing policy

Require namespaced TEI with one explicitly Latin edition under text/body.
Reject DTD/entity declarations and oversized files; do not load external entities
or execute any downloaded code. Use outermost prose paragraphs/verse lines,
and quotes outside those units. Nested verse in a quotation is included once.
Omit notes, headings, speaker labels, stage directions and character lists.
Keep ordinary inline emphasis/name/regularized spelling text without injecting
extra whitespace into words. Unaccounted structural alphabetic text or unknown
structural tags cause a failure, not silent data loss.

Quarantine an entire content unit if it contains gaps, foreign/unclear text,
editorial additions/deletions, choices/abbreviations/apparatus or other tags in
the source's declared uncertainty list. Bracketed units and unsupported
alphabetic tokens are also quarantined. Context resets at each quarantine and
book/poem/letter/act/scene boundary. These conservative filters discard some
usable Latin; all reasons/counts are retained. They do not reconstruct an
ancient original or resolve editorial disputes.

Normalize with NFKD/lowercase, ae/oe ligature expansion, combining-mark removal,
j→i/v→u; concatenate23letters `abcdefghiklmnopqrstuxyz`. Remove whole tokens
containing digits; unsupported alphabetic text fails the unit. Preserve XML
node paths, source/normalized hashes, unit lengths, quarantine ledger and
segment boundaries. No context crosses removed material. The preliminary
training/selection format check found no structural parse failures and about
7.18million clean letters before overlap filtering; this was not model scoring.

## Exact overlap removal and two data sizes

Build all64-character windows from reserved-author retained segments and the
already-pinned50kCicero/Sallust/Tacitus allocations, respecting their boundaries.
Remove whole Pliny units touching any such match and reset at removals. Add the
surviving Pliny windows to the protected set. Then remove whole training units
touching protected windows; matches crossing two units remove both. Never
rejoin remaining fragments across the gap. Verify zero surviving64-character
training overlap. This does not prove absence of shorter/fuzzy paraphrase or
literary dependence. No performance-based exclusion or manual deletion.

The large corpus contains every surviving training segment. The small corpus
is the first50,000remaining letters from Caesar and50,000from Virgil, preserving
boundaries and retaining a flagged truncated final unit if necessary. Both use
the same edition/filtering policy. The small corpus is a subset of large, but
is not byte-identical to the older Gutenberg source; do not conflate their
baseline results. Keep author archives separate so the training loader can
reject reserved-role inputs by construction.

## Execution, verification and next gate

Freeze parser/tests/this policy/acquisition manifests before the one preparation:
`PYTHONPATH=.:src .venv/bin/python scripts/prepare_latin_source001.py --freeze <checkpoint>`.
Single CPU thread environment,600CPU/900wallsecond limits,8GiBplanning, no paiduse.
Stop and preserve failures rather than silently relaxing parsing rules. Any
necessary correction is a separately documented preparation revision, not a
new successful run under the original source hash.

Tests cover exact inline extraction, nested verse, all uncertainty classes,
context resets, source-unit coverage, normalization, unsafe XML rejection,
unsupported structures, prefix slicing and boundary-spanning duplicates. Verify
all hashes, all retained segment character totals, source paths and role splits,
small-within-large identity and the overlap screen. Keep raw/derived/ledger
archives out of Git; track compact provenance, exclusions and counts. Update
notebook/memory and publish a remotely verified checkpoint before model fitting.

Next specify a paired small/large data comparison for the same neural source
and the same statistical baseline. Use only Pliny for checkpoint/source selection;
reserve later cipher accuracy for the two declared reading authors. Model and
inference controls, seeds, exposure budget, compute timing and success criteria
must be frozen separately. No neural training or new cipher evaluation has yet
been launched by this preparation.
