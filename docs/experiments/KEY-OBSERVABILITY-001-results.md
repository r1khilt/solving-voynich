# Known source resolves every used key row in these 64 cases

2026-10-01. The single registered source-supplied oracle run completed all64
exposed development cases: **64unique used dictionaries,1248/1248used rows
determined, zero search caps**. Every true used dictionary was present. The
oracle received the original plaintext and unsegmented cipher, without the
dictionary or unit boundaries. This does not establish ciphertext-only recovery.

In plain English: once we supply the answer text, the computer can work out
every key entry needed to encode it. The remaining uncertainty concerns letters
that never appeared. That separates those two problems; it does not tell us
how to find the answer text when it is unknown.

| Unused source rows | Cases | Compatible full dictionaries per case |
| ---: | ---: | ---: |
| 2 | 5 | 42²=1764 |
| 3 | 32 | 42³=74088 |
| 4 | 19 | 42⁴=3111696 |
| 5 | 6 | 42⁵=130691232 |
| 6 | 2 | 42⁶=5489031744 |

Under the declared iid uniform validation prior, even the source-supplied
Bayes oracle has only8981533/2744515872≈0.00327254expected complete-key matches
across all64cases. A union bound gives at most0.327254% probability of any exact
complete key, conditional on the supplied source/cipher information and that
prior. These are posterior expected probabilities, not hard finite-sample
limits or probabilities of the observed fixed experiment's outcome. Gold keys
and RNG provenance are excluded from the oracle's inference information.

Thus zero complete keys is unsurprising when2–6entries are unobserved in every
case. This does **not** excuse the small model's49/1248used-row matches: that
model must infer the plaintext as well, and this oracle does not measure how
ambiguous ciphertext-only inference remains. Full-row proper loss retains
legitimate uncertainty and is unchanged; no target masking is introduced.

One run4906, terminal0, at remotely verified source freeze
e240b2def5727d6bcb835add099143b9168890f6. Pure Python search visited8328nodes
total; stage.116041wall/.106949CPU seconds,212,221,952peakRSS. One audit1911,
terminal0, replayed all64searches, literal re-encoding of every solution,
summaries and hashes; stage.113885wall/.108348CPU seconds,210,436,096RSS.
The auditor uses the same counting algorithm and author; it is not independent
scientific confirmation. A separate full-dictionary algorithm exhaustively
checked tiny fixtures before launch. Prelaunch full2480tests+23subtestsPASS,
13skips; changed lint and whitespace checks passed, five older lint findings
remain. Zero model inference, training, GPU, paid spending or new holdout use.

The59,445-byte ignored case trace has SHA256
c9cff734ec0cbbfe0ce712449bb7af76be1eb14834698748b4d6b6cdc2fa291f.
The2055-byte compact result has SHA256
8a3d50bf885e97c9cd454ad70e18e55bd3eb6e610c845dceaa573bbe25aa7b28.
The full main training integrity audit remains pending. The original95M fit
continues unchanged. No historical channel or plaintext is inferred.

The [collapsed source/key derivation](../research/collapsed-source-key-inference-2026-10-01.md)
shows a candidate way to remove irrelevant unused-key guessing from future
reading search without deleting legitimate uncertainty. It has only exact
small-case arithmetic checks; no realistic ciphertext-only decoder is trained
or qualified by this result.
