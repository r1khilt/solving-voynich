# WMD-0001 result archive

Read the [results and limitations](../../docs/experiments/WMD-0001-results.md) and [preregistration](../../docs/experiments/WMD-0001.md).

- `joint-*` and `actions-*`: immutable training manifests and summaries.
- `trained-a0/a2.json`, `untrained-a0/a2.json`: all 64 development-world scores per condition. `a2` means two supplied synthetic oracle correspondences. The JSON method named `neural` includes compiler search; the filename and `model_condition` identify whether weights were trained.
- `*-audit.json.gz`: lossless compressed inference audit sidecars, including candidate origin, order, checks, projection repairs and bounded search reports. No gold target participates in ranking. Python's `gzip.open(path, 'rt')` reads these as JSON. Decompressed bytes match `qualification-manifest.json`.
- `comparison.json`: source counts, exact untrained replay checks and explicitly post-run descriptive paired bootstrap differences. These are not confirmatory significance tests.
- `interventions.json`: identity, donor, targeted and matched-random controls on one fixed model pair.
- `demo-inference.json`: the fixed index-zero command-line demonstration, which abstains; action diagnostics are therefore not applicable.
- `manuscript-format-manifest.json`: checksums and inventory for the read-only training-format export, with no manuscript decoding.
- `artifact-manifest.json`: retained artifact and local checkpoint hashes. Checkpoints, generated worlds and raw manuscript data stay outside Git.

Scientific source is `eeba1468d01e438a5c9009283327abe156f86f22`. The subsequent backend final-test guard is documented as post-run hardening. Reproduction commands and the exact source-check requirement are in the results document. All accepted candidates are nonsemantic copy-family surface explanations; no procedure or taxonomy was recovered. No historical decipherment or multilingual foundation-model training is claimed.
