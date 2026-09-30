# Borg next executable control: lossless quarantine, then verified glyph units

2026-09-29. Preparation and format validation only. **No Borg fit, plaintext
decoding, answer-key comparison, or source-language score was run.** The next
historical control remains useful, but the existing unknown-unit search cannot
honestly consume the current unresolved character inventory.

## Available input and exposure

The locally available input is
`data/raw/borg/transcription-0001r-0204v.txt`, from the
[official transcription URL](https://www.su.se/download/18.6856063019d24ef3ecb1117/1774944956220/transcription-0001r-0204v.txt).
The raw bytes were rechecked, without newline conversion:

| Property | Rechecked value |
| --- | --- |
| Bytes | 166,508 |
| Unicode codepoints | 166,498 |
| SHA-256 | `79950123a2760e92f3f27287aca94d16168169651eac8a4f9c15af37e349dde3` |
| Physical lines | 8,217 |
| Page-header blocks | 409 |
| Original-cleartext-tagged blocks | 0, 1, 407, 408 |

The [source manifest](../../data/manifests/borg_source_inventory.json) pins the
original retrieval, mixed line endings, source version and page boundaries. Raw
transcription bytes remain ignored and are not republished. The new parser
reads no Latin corpus, learned dictionary, published key, corrected text or
translation. In particular, neither final Latin author was opened.

The [earlier inventory](borg-ciphertext-inventory.md) and
[historical feasibility review](historical-controls-2026-09-25.md) remain the
exposure record: two corrected example words in the opening 400 encrypted
characters were literature-exposed, and an earlier diagnostic exposed original
cleartext front matter. None is a crib. This preparation adds full-file format
and codepoint aggregates, not plaintext or key evidence. A metadata comment at
source line 3,993 flags a new scribe and a need to check the symbols/decoding;
this is a transcriber note, not a verified change of encryption mechanism.

## Source-dependent facts and unresolved semantics

Rechecked on 2026-09-29, the
[Stockholm University project page](https://www.su.se/english/research/research-catalogue/research-projects/d/decipherment-of-historical-manuscripts/the-borg-cipher)
describes 34 graphical types including Roman letters and diacritics, original
cleartext at the beginning/end and in some early headings, and catchwords at
the bottoms of some pages. It still gives the same TXT URL for both the
transcription and automatic-decipherment labels. Its glyph count is not a
source-specific ASCII tokenization dictionary.

No key-free glyph-to-ASCII legend or source-specific uncertainty/annotation
grammar was located in this bounded review. The
[DECRYPT portal](https://de-crypt.org/) links general transcription guidance;
the linked 2021 guideline landing page returned a bot challenge, and the
historical-cryptology chapter exceeded the browser's content-size limit. Neither
failure establishes that guidance is unavailable elsewhere. Generic guidance
would still need verification against these actual bytes. The published key
image and answer/translation links were not opened.

Consequently, a whitespace chunk is not established as one glyph or one
plaintext word. Neither a raw ASCII character nor its case/punctuation/digit
status is established as one cipher glyph. Square and angle spans might include
original cleartext, editorial material or uncertain ciphertext; quarantining
them is a conservative exclusion policy, not a semantic classification. The
remaining unmarked body has **not** been proved free of original cleartext.

## Executed structural control

[`scripts/borg_control_parser.py`](../../scripts/borg_control_parser.py) is an
executable, standard-library-only parser for the format quarantine. Its default
command accepts only the pinned raw-byte SHA; artificial tests call the same
parser with toy bytes. It produces no solver record or decoded string.

The parser records a contiguous partition with byte and Unicode offsets,
physical-line numbers, ordinal page-block identity, category and slice hash.
The pinned source plus that partition reconstructs every source byte exactly.
It preserves original mixed line endings and case. Duplicate labels, suffixes
and the `#pahe` typo remain literal metadata; normalized labels are presentation
only. It rejects malformed headers, invalid UTF-8, unmatched/crossed annotation
delimiters and annotations crossing page headers or hash comments.

Whole explicitly cleartext-tagged pages, preamble, hash comments and balanced
square/angle spans are quarantined. For unbracketed `?`, `*` or `/`, it
quarantines the entire remaining physical line because the marker's scope is
unverified. This is deliberately conservative, not a missing-glyph model. It
never removes a marker and concatenates the surrounding pieces into a certain
reading. Every physical line and excluded span remains a separate boundary.
All other nonspace material is called `unresolved_body`, not ciphertext glyphs.

| Result of the fixed lexical rules | Count |
| --- | --- |
| Complete offset-partition spans | 44,317 |
| Byte coverage | 166,508 / 166,508 |
| Square annotation spans | 628 |
| Angle annotation spans outside hash comments | 72 |
| Multiline annotation spans | 12 |
| Lines quarantined for unresolved markers | 1,106 |
| Remaining unresolved-body nonspace codepoints | 101,298 |
| Remaining unresolved-body codepoint types | 59 |
| Lines with unresolved-body material | 6,140 |
| Page blocks with unresolved-body material | 401 |
| Solver records produced | **0** |

The older inventory's 73 angle spans include one five-character span inside the
hash comment on line 8,002. This parser classifies that whole line as a comment;
no bytes are lost. The earlier 120,677-codepoint/66-type view did not quarantine
whole marker-bearing lines, so its totals should remain different. The new
59-type count is no more a validated graphical alphabet than the old 66.

Run without printing any source text:

```sh
.venv/bin/python scripts/borg_control_parser.py
PYTHONPATH=.:src .venv/bin/pytest -q tests/test_borg_control_parser.py
```

An optional `--ledger /absolute/new/path.json` writes only offsets, hashes,
categories and page labels, refuses overwrite and emits its hash. The verified
local ledger is 10,241,054 bytes, SHA-256
`7b7e85c1f2ddb3da7a3d7dbc7638295bcf7296fedb68ed5dde31a46035df62b9`;
it is regenerable and is not tracked. The command-line summary is redacted.

All **32 artificial tests passed** in 0.05 seconds. They include mixed newline
round trips, multibyte offsets, cleartext before an explicit marker, both
cleartext language tags, nested/multiline annotations, marker-scope quarantine,
duplicate/suffixed labels, preserved literal case/punctuation, malformed input,
hash mismatch and altered partitions. Changed-file Ruff checks pass. A separate
regex-mask calculation using the pinned source manifest's page intervals
reproduced all 59 codepoint counts, 101,298 remaining codepoints, 6,140 lines,
401 blocks and 1,106 marker lines. This second calculation was authored by the
same agent and is an alternate implementation check, not independent-agent
review. No paid resources were used.

## Exact compatibility obstruction

The implementation in
[`unit_channel_search.py`](../../src/voynich/unit_channel_search.py) assumes one
state and one deterministic nonempty emission unit per source letter. It
allows identical units for different source letters; that is not the same as
allowing several alternative glyphs for one source letter.

For the current 23-letter source and maximum emission length two:

- The initializer explicitly requires source alphabet size at least glyph
  alphabet size. Both 59 and 66 fail that guard; even the project's described
  34 graphical types would fail it if all were distinct modeled glyphs.
- The complete literal pool is `G + G²`: 3,540 at G=59; 4,422 at G=66; and
  1,190 at G=34. The search cap is 256 units. The higher-order refiner calls the
  same pool builder with its default 256 cap.
- Even after removing these software guards, 23 deterministic rows of length
  at most two can contain at most **46 distinct emitted glyphs**. Treating all
  59 or 66 unresolved types as required glyphs cannot support the full input.
  This is an elementary family-capacity bound, not a historical claim about
  Borg or a result of fitting it. The 34-type description does not by itself
  exceed this mathematical bound, but does exceed the current initializer and
  pool-cap assumptions.

Increasing a numeric cap, discarding rare characters, merging case or forcing
the transcription into six glyphs would not establish correct units. Nor does
this obstruction imply that Borg needs a stateful mechanism. If verified
tokenization establishes homophones or nulls, the generative family must model
those explicitly and be qualified on matching fresh synthetic controls first.

## Next runnable historical stage

The concrete missing input is a **key-free transcription legend** linking
graphical glyphs to ASCII codes, with source-specific meanings/scopes of square
spans, angle spans and uncertainty marks. A manually verified transcription-to-
image correspondence can substitute for the legend, but must be made without
the published plaintext key. Catchword positions and the duplicate/suffixed
physical-page identities also need explicit handling before page-level splits.

Once those facts are established, extend the offset parser with explicit glyph
tokens and uncertainty alternatives; retain all quarantined bytes in the
ledger and report any selection loss. Freeze a pilot and transfer assignment
by physical leaves before key fitting, excluding the literature-exposed opening
from confirmation. Qualify the actual resulting alphabet and channel family on
fresh synthetic positives and wrong-language/shuffled controls before fitting
Borg. Freeze the learned key before any published-key or corrected-text
comparison, and distinguish literal transcription decoding from editorial
correction. Current artifacts supply the reproducible format layer for that
stage; they do not claim that the glyph-semantic or historical-recovery gate
has passed.
