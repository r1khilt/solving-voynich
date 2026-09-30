# Final Latin corpus acquisition and provenance

Prepared 2026-09-29. **Implementation and artificial tests only: neither Sallust
nor Tacitus has been fetched or opened by this preparation task.** No empirical
key, sample or recovery score has been generated. Follow the separate experiment
registration and [research protocol](PROTOCOL.md).

The [existing catalogue review](blind-channel-corpus-plan.md) selected Sallust,
*C. Sallusti Crispi De Bello Catilinario Et Jugurthino*, [PG7402](https://www.gutenberg.org/ebooks/7402),
for ciphertext fitting; and Tacitus, *Germania and Agricola*,
[PG9090](https://www.gutenberg.org/ebooks/9090), for transfer. Their already
recorded advertised endpoints are `https://www.gutenberg.org/ebooks/7402.txt.utf-8`
and `https://www.gutenberg.org/ebooks/9090.txt.utf-8`. The catalogue review lists
both as English/Latin and public domain in the USA. That is a metadata screen,
not completed edition-level or worldwide rights clearance. Both contain English
material, so treating the whole file as Latin is prohibited. No catalogue body
or ebook was opened for this implementation.

**2026-09-30 pre-text transport amendment:** The first frozen acquisition refused
the advertised endpoints' HTTP downgrades before receiving any book bytes.
The failed `blind_channel_confirmation_acquisition.json` is preserved. HEAD-only
checks verified the same files at direct HTTPS cache paths, PG7402/PG9090, with
503,378/436,490 advertised bytes. After a new published pre-text freeze, use
`https://www.gutenberg.org/cache/epub/7402/pg7402.txt` and
`https://www.gutenberg.org/cache/epub/9090/pg9090.txt`; record that separate attempt
in `blind_channel_confirmation_acquisition_v2.json`. All scientific choices and
no-overwrite/HTTPS checks remain unchanged. This is an explicit registration
amendment, not automatic retry or alternate-edition selection.

## Ordered stages

1. Freeze and publish the whole recovery pipeline, extraction policy, fixed
   windows, seeds, budgets and decision criteria **before acquisition**.
   `scripts/prepare_blind_channel_confirmation_corpora.py acquire --freeze
   <40-character-commit> --registration <tracked-registration>` checks its own
   source, normalizer, tests, this policy, the development manifest and the
   experiment registration against that commit. It performs one sequential
   request per fixed endpoint, with a 2 MiB raw cap and a 60-second socket timeout
   per request. It accepts only HTTPS Gutenberg redirects; there is no alternate
   edition, automatic retry, paid service, or truncation. Raw bytes, complete
   license notices, resolved URL, time, optional ETag/Last-Modified, size and
   SHA-256 are preserved. Console output contains metadata only. A failed second
   request leaves the first raw file and a failed acquisition manifest; the
   script refuses to rerun over either success or failure.
2. The evaluator audits source quality, rights, headings, authorial boundaries,
   editorial apparatus and quotations **without running a source model,
   cipher/oracle solver or recoverability test**. All eligible original-Latin
   authorial bodies belong in publication order. Body and editorial-exclusion
   byte intervals, individual hashes, reviewer and edition notes go into a
   tracked extraction plan. This is a human/evaluator bibliographic audit:
   ASCII-looking English and Latin editorial prose cannot be identified by the
   normalizer. Ambiguous separation requires a recorded stop before generation;
   do not silently choose an easier edition, passage or replacement author.
3. Commit and publish the raw hashes and reviewed extraction plan before
   `prepare --freeze <new-commit> --registration <same-registration> --plan
   <tracked-plan>`. Preparation checks the acquisition and plan at that freeze,
   and requires the extraction code/policy/registration still to match the
   original pre-acquisition freeze; the later commit may add reviewed spans but
   cannot silently change the method after exposure. It
   rebuilds pinned full development bodies, then applies the reviewed final
   spans and the exact development normalization. It returns counts, hashes,
   maps, fixed offsets and overlap diagnostics. No key or cipher is generated.
4. Freeze/publish the prepared manifest and all extraction evidence before
   generating the separately registered eight independent cipher keys. Only
   the evaluator/generator reads private plaintext; the fitter receives its
   assigned ciphertext. This is procedural isolation, not cryptographic
   protection against someone deliberately reading local answer files.

## Extraction and window contract

The extractor requires strict UTF-8, exact reviewed bytes and one unambiguous
Gutenberg start/end boundary; body spans must be ordered, disjoint, within those
boundaries and uniquely named. Editorial exclusions must be ordered, disjoint,
hashed, inside their body and carry a reason. They are replaced by separators,
never joined into an invented word. The existing `normalize_segment` is reused
unchanged: NFKD/lowercase, ae/oe ligature expansion, combining-mark removal,
`j→i`, `v→u`, balanced square-bracket exclusion, complete numeric/mixed-token
exclusion, punctuation token separation, omitted spaces and strict rejection of
unknown alphabetic symbols. This is a lossy engineering convention, not a
philological assertion. Roman-letter numerals retained by development remain
retained here; no spelling or language-model repair is added.

Keep the first **50,000** eligible normalized letters of each final author in
publication order. Keep original authorial body boundaries and each normalized
token's original raw-byte interval. If the prefix ends within a token, retain
its original whole span and flag its clipped selected end. Full eligible-body
hashes, counts and body-level spans remain recorded even beyond the selected
prefix. Fewer than 50,000 eligible letters is a failure; no adaptive extension
or replacement is permitted.

For each selected body in publication order, enumerate complete **224-letter**
windows starting at body-relative offsets **0, 256, 512, …**. A window never
crosses an authorial body boundary or the 50,000-letter cap; stride resets at
each new body. For key index `k = 0,…,7`, Sallust fitting uses candidate indices
`16*k + [0,1,2,3]`; Tacitus transfer uses `16*k + [0,1]`. Candidate indices,
body index, body-relative offsets, global normalized offsets and individual
record hashes are recorded. Insufficient candidates or identical selected
records within either role fails; there is no shifting to a different passage.
All eight keys use disjoint records within each author.

`fixed_windows(payload, role, key_index)` is the evaluator/generator interface:
`role` is `fit` or `transfer`; it returns `text`, `start`, `end`, `body_index`,
`body_offset` and `candidate_index`. This return value contains private
plaintext and must not be logged or committed. The preparation manifest stores
only hashes and offsets for those records. Each source exposes
`derived_path`/`derived_sha256` for checked downstream loading.

## Duplicate and leakage checks

Before generation, enumerate every exact **64-letter** window in every eligible
authorial body of all five authors. Rebuild Caesar, Virgil and Cicero from their
pinned raw bytes and original extraction rules, and verify each recorded full
normalized hash. Thus the screen includes their full eligible bodies, not just
the 50,000-letter prefixes used by the source model. Do not fabricate windows
across work boundaries. Compare literal strings, not only hashes; report all
ten cross-author pairs, unique shared windows and position counts. Any overlap
involving either final author stops preparation, records a blocked manifest,
and emits no derived final payload. Genuine ancient quotations are still a
reported failure, not silently deleted material. Within-author repeated windows
are counted separately; they are not automatically removed. Identical selected
224-letter records are a separate failure.

Zero exact overlap does not rule out short/fuzzy quotations, common phrases,
editorial influence, genre or edition effects. Caesar/Virgil/Cicero remain
exposed; only the final author allocation is fresh relative to this pipeline's
development. An LLM's possible pretraining exposure is not certified absent.

## Artifacts and extraction-plan schema

Raw files live under ignored `data/raw/blind-channel-confirmation/`; derived
payloads under ignored `data/processed/blind-channel-confirmation/`. The compact
tracked current manifests are `data/manifests/blind_channel_confirmation_acquisition_v2.json`
and `data/manifests/blind_channel_confirmation_corpora.json`. No full text or
token map is added to Git. The prepared artifact includes source/code freeze,
plan/acquisition/development hashes, URL/time metadata, normalization provenance,
spans/exclusions, overlap results and window policy. All artifact writers refuse
overwrites. A structural exception preserves acquired raw files and the frozen
plan for diagnosis; root must record that failed attempt in the notebook.

The reviewed plan is JSON with `schema_version: 1`,
`status: "mechanical_body_review_complete"`, `prefix_cap: 50000`,
`duplicate_window: 64`, and `sources` keyed by `sallust`/`tacitus`. Each source
requires `raw_sha256`, `rights_review: "public_domain_usa_notice_checked"`,
nonempty `edition_review` and `reviewer`, and `bodies`. Each body requires
`body_id`, `raw_byte_start`, `raw_byte_end`, `sha256`, and an optional
`editorial_exclusions` list. Each exclusion requires the same byte/hash fields
and `reason`. End offsets are exclusive and refer to the immutable original
UTF-8 bytes. Do not invent a raw hash or an exact edition identity before the
corresponding evidence exists.

Artificial tests cover exact normalization and byte maps, ligatures and partial
tokens, malformed hashes/spans/UTF-8, apparatus exclusions, all fixed-window
indices, duplicate detection, safe download caps/redirects, incomplete acquisition,
no-overwrite/freeze guards, and complete successful/overlapping fixture
preparation. They do not establish real edition quality or acquisition success.
