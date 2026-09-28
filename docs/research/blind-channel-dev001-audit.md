# BLIND-CHANNEL-DEV-001 independent evaluation audit

2026-09-27. Implemented before exposed-development generation or scoring. This
document describes an audit procedure and toy validation, not development
results. Implementation used the builder/runner/evaluator source schemas only;
no raw corpus, derived corpus, generated fitting/transfer data, or answer file
was opened. The broader objective and limitations remain those in the
[blind-channel design](blind-channel-recovery-design.md).

## Independent calculations

`scripts/audit_blind_channel_dev001.py` imports no production inference,
decoding, edit-distance, model-code or baseline helpers. Its portable
`evaluate_channel(source, channel, records, gold_records=None)` accepts plain
serialized model dictionaries; `channel=None` represents no completed fitted
model. Importing the module performs no file reads.

The evaluator independently validates complete normalized source, initial and
emission rows. It computes marginal observation likelihood with an incoming-edge
log-space dynamic program over glyph offset, source context and channel state.
Every emission consumes a positive number of glyphs, so the lattice is acyclic.
Incoming masses are summed together with a stable maximum shift. An independent
maximum table and backpointers recover one best joint source/channel path.
Exact ties retain the first representative reached in declared model order.
This is a joint Viterbi output, not a most-probable-plaintext calculation.

The code includes initial state mass, every continue/stop factor, source
probabilities, and complete emission-row probabilities. It neither normalizes
over only matching alternatives nor trims an emission to fit an observation.
It verifies both marginal and best-joint log probabilities against the
production evaluator, along with literal predicted strings.

Insertion, deletion and substitution each cost one in a separate linear-memory
Levenshtein implementation. An unsupported observation has null likelihood,
joint score and plaintext, and zero decoded characters; with gold it counts as
deleting the entire gold string. Aggregate likelihood is null if any record is
unsupported. Null cases have no edit, gold-character or exact-record metric.
Every output is strict JSON, without infinities or fabricated empty plaintext.

For the selected fitted channel, model bits are independently counted from all
finite-code field cardinalities, including initial weak compositions and
positive emission-weight compositions. Grid membership and literal support are
checked; its fitting data score uses the same actual coded weights. This audits
the selected final model, not every structural-search proposal.

The auditor independently reconstructs the registered iid-glyph baseline from
fitting glyph counts, its positive denominator-256 weights, denominator-4096
stop probability, enumerative codeword and bit count. Its transfer score keeps
those fitted parameters frozen. It then recomputes the reported fit saving,
transfer gain per glyph and the registered diagnostic flag (at least 32 fit
bits saved and 0.05 transfer bits per glyph), plus the compact diagnostic counts.
These are development diagnostics, not probabilities of semantic correctness.

## Artifact and answer-access checks

Both the callable `audit(..., after_evaluation=False)` and command-line entry
refuse work before explicit after-evaluation authorization, before any artifact
read. The command is:

```text
PYTHONPATH=.:src .venv/bin/python scripts/audit_blind_channel_dev001.py --after-evaluation
```

It requires the evaluator's compact report and verifies its advertised full
prediction artifact before comparing record details. It also checks:

- Manifest and source-selection bytes against their recorded SHA-256 hashes.
- Every case's fitting, transfer and answer artifact against manifest hashes.
- Every frozen-model file against the evaluator's freeze hash.
- Compressed full search output hash and byte count, then equality of its
  selected channel, score and configuration with the compact model freeze.
- Frozen fitting-input and source hashes against the manifest, and case/split
  identity and positive/null status against the answer and observation files.
- Consistency of one syntactically valid source-freeze commit identifier across
  cases, and all common compact/full evaluator fields after removing only the
  deliberately untracked per-record prediction detail.

Known strings, booleans, integers, nulls and collection shapes must match
exactly. Floating results use absolute tolerance `1e-7`; the report records the
largest observed difference. Additional production fields can coexist without
being represented as audited. Hash checks bind the audit to exact selected
models and inputs; they do not by themselves prove an earlier Git publication
time. Source/key publication and remote-ref verification remain the orchestration
workflow's responsibility.

## Pre-data validation and limits

`tests/test_blind_channel_dev001_audit.py` contains **22 passing tests**. Hand
rational fixtures verify ambiguous segmentation, duplicate latent paths,
empty observations, complete-row accounting, initial-state marginalization,
stateful order-one inference, unsupported predictions, null metrics, model-code
fields and baseline-code arithmetic. Known edit examples include empty and
Unicode strings. Temporary fabricated artifact chains exercise the complete
audit, hash corruption at every input/model/prediction layer, and a wrong
prediction whose artifact hashes have been recomputed consistently. The latter
must still fail independent semantic comparison. Tests also verify refusal
before any read and schema compatibility with the production helper.

Validation command:

```text
PYTHONPATH=.:src .venv/bin/python -m pytest -q tests/test_blind_channel_dev001_audit.py
```

Result: **22 passed in 0.18 seconds**. Changed-file Ruff passed. No experimental
audit has run during this implementation block, and no recovery result follows
from these tests. The procedure does not certify global optimization, all search
trace candidates, the historical source language, nonlanguage identifiability,
or Voynich decipherment.
