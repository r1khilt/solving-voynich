# Borg historical-control feasibility review — 2026-09-25

This bounded source review ran alongside NAIBBE-002. It selected no decoding key and opened no Borg file, plaintext, key, translation or automatic decipherment. Unlike Naibbe's manufactured control, Borg would test real transcription and remove supplied codebook grouping/role structure. Its known cipher family may be simpler, so this is a different validation dimension rather than a claim of uniformly harder cryptanalysis.

## Primary-source observations

The [official Stockholm University project page](https://www.su.se/english/research/research-catalogue/research-projects/d/decipherment-of-historical-manuscripts/the-borg-cipher) describes a408-page manuscript with34 character types and lists a162.6kB transcription. It supplies prior knowledge of Latin/some Italian and describes cleartext material, human transcription and subsequent correction/translation. These are prior assistance, not things a new solver may claim to discover blindly.

The [listed TXT](https://www.su.se/download/18.6856063019d24ef3ecb1117/1774944956220/transcription-0001r-0204v.txt) appears under both the ciphertext-transcription and automatic-Latin-decipherment labels on that page. **Contents have not been verified.** A future acquisition must isolate and inspect file identity before letting it into a ciphertext-only pipeline. Do not assume the filename or one link label is sufficient evidence. Download size and34-type count above are site descriptions, not independently counted from the file.

The [official historical-cryptology transcription chapter](https://de-crypt.org/Esslinger-017_Chap03-Historical-Cryptology.pdf) documents glyph transcription conventions including Unicode symbols/names, spaces, lines, punctuation, diacritics and uncertainty. Exact downloadable syntax and token counts remain unchecked. A parser must preserve uncertain glyphs and avoid equating whitespace with plaintext word boundaries.

The [DescryptTool primary system paper](https://www.tandfonline.com/doi/full/10.1080/01611194.2026.2693460) describes a monoalphabetic-substitution route for Borg using transcription cleanup and a Latin solver. This supports a disclosed simple-family historical control first; it does not justify treating all34 observed types as letters or adopting unpublished cleanup decisions as ground truth.

The [DECRYPT portal](https://de-crypt.org/) describes Apache2.0 for platform/tool code with database content excepted. [Database terms](https://de-crypt.org/termsofuse.php) are not a blanket dataset redistribution license. The review found no Borg-specific code download or explicit dataset license on the project page. Keep any raw acquisition ignored, track retrieval/hash/rights, and clarify redistribution before publishing third-party assets.

## Exposure and proposed isolation

A general primary-literature search unexpectedly showed the reviewing agent a snippet containing two corrected example words from the opening400 characters. The words were not relayed or used, and no answer file was opened. Nevertheless, mark that opening segment **literature-exposed**; it cannot be a clean confirmation block. This is an exposure disclosure, not a reason to abandon later untouched pages.

A future registration should pin a ciphertext-only parser and explicit substitution-family hypothesis; exclude cleartext headings/front/back material and catchwords from scored recovery; fit a shared key on a bounded early encoded-page block and freeze it before later-page decoding. Disjoint Latin/Italian priors, wrong-language controls and synthetic calibration matched to the observed alphabet/missingness should precede answer scoring. After key freeze, use the published key for literal key/text recovery and score editorially corrected plaintext separately. Preserve the possibility that a bad score reflects transcription or editorial mismatch. Compare the searched and oracle-key objectives without using gold to restart search.

Compared with NAIBBE-002, no six-table partition, known23-letter table bijections, linked uni/prefix/suffix roles, codeword inventory or preconstructed lattice should be supplied. Human glyph transcription and known language candidates remain assistance. This review is preparation only; **no Borg decoding experiment or data download occurred**.
