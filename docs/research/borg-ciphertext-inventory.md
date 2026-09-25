# Borg ciphertext acquisition and format inventory — 2026-09-25

The [official candidate TXT](https://www.su.se/download/18.6856063019d24ef3ecb1117/1774944956220/transcription-0001r-0204v.txt) has been acquired and verified as a **ciphertext transcription containing original cleartext and annotations**, rather than a file of automatic Latin decipherment. This is a format inventory only. No decoding key, recovered plaintext, language-model score, or cipher-family fit was produced. Verification does not establish that every untagged character is encrypted.

The [source inventory](../../data/manifests/borg_source_inventory.json) pins the bytes, retrieval, original page boundaries, aggregate symbol counts and limitations. The original bytes and retrieval evidence remain ignored under `data/raw/borg/`; no third-party transcription is added to Git.

## Acquisition and exposure

The [official project listing](https://www.su.se/english/research/research-catalogue/research-projects/d/decipherment-of-historical-manuscripts/the-borg-cipher) was rechecked on 2026-09-25. Both the transcription label and automatic-decipherment label still point to the same URL. The listing was saved, followed by a HEAD request to that exact URL and then one GET. No answer, key, translation or other data link was requested, and redirects were not followed.

| Property | Observed value |
| --- | --- |
| Retrieved | 2026-09-25T22:44:01.232126+00:00 |
| Bytes | 166,508 |
| SHA-256 | `79950123a2760e92f3f27287aca94d16168169651eac8a4f9c15af37e349dde3` |
| HTTP last modified | Tue, 31 Mar 2026 08:15:56 GMT |
| HTTP content type | `text/plain;charset=UTF-8` |
| Encoding | Strict UTF-8; no byte-order mark |
| Retrieval | macOS system curl with normal TLS verification |

Python 3.14 urllib initially failed certificate verification before downloading. System curl succeeded; certificate checking was never disabled. The URL's resource timestamp and HTTP date are the available source version; there is no supplied repository commit. Raw response headers remain ignored and contain temporary server cookies. The tracked manifest includes only a noncredential header allowlist.

The first 2,048 raw Unicode characters were inspected mainly through redacted structural summaries. Explicit cleartext markers precede front matter, while subsequent unbracketed text uses mixed letter/digit strings. Aggregate inspection confirms a predominantly ASCII code transcription; the file is **not** the Unicode-name token stream suggested by generic DECRYPT transcription guidance.

**Exposure disclosure:** one diagnostic unintentionally printed nine bracketed single-field strings from that opening sample, including original cleartext front-matter words and abbreviations, while trying to distinguish annotation labels. They were not used as cribs and are not reproduced in the tracked artifacts. No deciphered ciphertext or key was displayed. Separately, the earlier literature review exposed two corrected example words from the first 400 encrypted characters; that opening segment remains literature-exposed. Aggregate format statistics now cover the entire ciphertext, so a later transfer split must be described as unseen by key/language fitting, not as completely unread ciphertext.

## Observed syntax and boundaries

There are 166,498 raw Unicode characters and 8,217 physical lines under Python `splitlines`. The bytes contain 7,894 CRLF endings, 316 bare LF endings and seven bare CR endings. Reading in universal-newline mode changes a re-encoded byte count; provenance hashes must always use the original bytes.

The file has 420 hash-comment lines, 412 blank lines and 409 page-header blocks. Normal headers use `#page` followed by a folio label. A source `#pahe` typo occurs at line 6,450 for `0157v`; some labels lack zero padding. Page identity has unresolved defects:

- `0049v` occurs twice, at lines 2,012 and 2,032; `0049r` is absent.
- `0099r` is absent.
- `0150r.01` and `0150v.01` occur in addition to the unsuffixed versions. They must not be silently merged, presumed duplicates, or counted as independent physical pages.

The manifest preserves original labels, block order and inclusive source-line boundaries. Its zero-padded labels are presentation aids only. The 409 observed blocks should not be equated with the project's description of a 408-page manuscript.

Four standalone markers occur: one `<CLEARTEXT-AR>` and three `<CLEARTEXT-LA>`, on page blocks labeled `0001r`, `0001v`, `0204r`, `0204v`. There are 628 balanced square-bracket spans and 73 balanced angle-bracket spans. One square span and eleven angle spans cross physical lines. Of the angle spans, four are the explicit cleartext markers, 68 contain whitespace, and one contains a short numeric field. Their contents were not reproduced. A per-line regular expression is insufficient to remove these spans safely.

The raw file contains 1,465 question marks, six slashes and no asterisks. Generic DECRYPT guidance cannot be substituted for source-specific semantics: a question mark might mark an uncertain reading, a missing unit or another annotation. Neither its exact scope nor the meaning of every bracket/angle span has been established. Case distinctions, punctuation and literal-looking quantities must remain intact until those roles are resolved. Catchword positions have not been identified or removed.

## Conservative counting view, not a decoder parser

For inventory only, balanced square/angle spans were replaced with spaces while preserving newlines; hash-comment lines and the four explicitly cleartext-tagged page blocks were omitted. Every other case-sensitive codepoint was retained. This produces:

| Unit | Count |
| --- | --- |
| Nonempty candidate page blocks | 401 |
| Candidate lines | 7,246 |
| Whitespace-separated chunks | 24,661 |
| Nonspace codepoints | 120,677 |
| Residual codepoint types | 66 |
| Residual question marks | 1,394 |

Question-mark runs have lengths one (1,349 runs), two (21 runs) and three (one run). These are descriptive counts, not a missing-character model. Full case-sensitive codepoint counts and per-page sizes appear in the manifest without any text snippets.

**None of these units is yet a validated glyph or plaintext word.** The 66 residual types include punctuation, possible literal numerals, rare symbols and unresolved annotations; they do not contradict or reproduce the project's reported 34 graphical cipher types. There is no supplied character-to-glyph dictionary in this acquisition. Counting whitespace chunks as glyph tokens or forcing all 66 types into a bijective 23-letter key would be unjustified. Blanket span masking may also remove uncertain ciphertext, which is why this view is not proposed as the final decoding input.

## Next decoding feasibility

The volume is ample for a bounded Mac-scale substitution control. The immediate missing work is a verified ciphertext parser and a registered treatment of uncertainty, quantities, punctuation and catchwords—not a larger model. Resolve the ASCII code units without consulting the answer key; keep anomalous page blocks and original cleartext outside the first clean pilot, or score them separately under explicit rules. Preserve raw-to-parsed offsets and uncertain alternatives rather than silently deleting difficult material.

Then freeze physical-page/block fit and transfer assignments before fitting a key. Use independent Latin/Italian source priors, synthetic recovery controls matched to the observed parser output, wrong-language controls and an oracle-objective diagnosis only after the learned-key freeze. Do not supply exposed front-matter words as cribs. Any later published-key comparison must separate literal symbol decoding from editorial corrections. This remains historical calibration under disclosed language/family assistance, not evidence about Voynich meaning.

## Validation and rights

The ignored inventory program `data/raw/borg/inventory.py` records its own checksum in the manifest and can reproduce the manifest byte-for-byte without network access or answer inputs. A second implementation checks counts independently using regex span masking and page delimiters. Raw-byte checksums, per-page totals, source-line continuity and Git ignore status are checked before handoff. No target score or experiment was run, and paid spend is zero.

The earlier [feasibility review](historical-controls-2026-09-25.md) remains the record of prior-source exposure. [DECRYPT's portal](https://de-crypt.org/) describes Apache 2.0 for code while excepting database content; its [database terms](https://de-crypt.org/termsofuse.php) do not establish blanket redistribution permission. No Borg-specific dataset license was found on the listing. Raw material stays ignored; acquisition does not imply permission to republish the transcription.
