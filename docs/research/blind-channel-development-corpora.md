# Development-only Latin extraction

2026-09-27. The [metadata plan](blind-channel-corpus-plan.md) is now implemented
for **Caesar P1, Virgil P2 and Cicero D only**. Sallust/Tacitus remain unopened.
No new source-language or cipher score was used to choose these texts or spans.

The official [Virgil PG227](https://www.gutenberg.org/ebooks/227) and
[Cicero PG226](https://www.gutenberg.org/ebooks/226) catalogue text links were
followed and downloaded with curl, `--max-filesize 2097152 --max-time 60` on
2026-09-27. Sandbox DNS failed before execution could fetch data; the authorized
network escalation succeeded. Files are 495,876 and112,390bytes, respectively.
Caesar reuses the earlier pinned169,899-byte PG218 copy. The catalogues list
public-domain status in the USA, and the raw files retain their complete PG
license notices. The exact source print editions/editors are not established;
these are pinned electronic editions, not critical editions.

The [manifest](../../data/manifests/blind_channel_development_corpora.json)
records raw/source-code/derived hashes, body byte intervals, excluded byte
intervals with hashes, and counts. No raw or derived literary text is in Git.
The extractor verifies the raw SHA and2MiB cap before decoding strict UTF-8.
Header counts must match4Caesar books,12Virgil books and4Cicero orations.
Cicero's Latin `ARGUMENTUM` summaries are excluded using the unique authorial
opening of each oration; being Latin does not make them authorial text.

Whole square-bracket spans, including contents, are excluded and replaced by
token separation. This conservative policy excludes editor-marked material,
including potentially genuine ancient variants; it is not a philological
judgment. Nested brackets must balance. Whole numeric/mixed alphanumeric tokens
are excluded, including one `1o` in Cicero. Body headings and PG apparatus are
outside the selected spans. Punctuation delimits tokens; spaces are omitted
from the model alphabet. NFKD/lowercase, Latin ae/oe-ligature expansion,
combining-mark removal and j→i/v→u give the declared23-letter classical
engineering alphabet. Unknown alphabetic characters fail closed. Caesar has
two æ codepoints; the new two files contain only ASCII. Roman ASCII numerals
within authorial text are retained as letters. Orthographic/transcription
errors are preserved; no language model repairs the text.

Eligible full-body letters: Caesar121,204; Virgil365,310; Cicero70,778. Take the
first50,000eligible letters from each in publication order, retaining book
boundaries and a normalized-token→raw-byte map. A final partial token is
explicitly marked. This prefix rule was implemented before any cipher scores.
The selected prefixes have **zero shared exact64-letter windows** in all three
author pairs. This screen does not exclude shorter/fuzzy quotations, common
phrases, genre effects or editorial influence. Within-author repetition is not
removed. These exposed authors are not fresh final confirmation.

Seven artificial extraction tests cover brackets, foreign-letter rejection,
Latin-editorial exclusion, raw-byte spans, cap clipping and exact overlap
counts. A separate code/manifest-only reviewer checked their logic and source
hash, but did not independently inspect raw authorial content. The exact file
boundary audit is therefore root-reviewed, not independently philologically
certified. Source estimation and a bounded exposed recovery pilot follow in
[BLIND-CHANNEL-DEV-001](../experiments/BLIND-CHANNEL-DEV-001.md).
