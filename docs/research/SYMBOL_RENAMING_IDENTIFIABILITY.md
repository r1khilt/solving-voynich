# Symbol renaming, null identifiability, and a falsifiable architecture test

Status: mathematical deduction for the *synthetic* EXP-0030 channel, written before EXP-0031 output. It is not a property asserted of the Voynich manuscript.

## The symmetry that the benchmark actually has

Let \(A\) be the 90-character `CIPHER_POOL`, and let \(G=S_A\) be all bijections of these characters that leave spaces fixed. For a sample with visible text \(X\) and keep mask \(M\), a relabeling \(\pi\in G\) applies the *same* replacement to every occurrence of each cipher symbol, preserving positions, equality, spacing, and the true mask. In the EXP-0030 WORLD_C generator, the ordered 36- or 40-symbol working alphabet is sampled uniformly without replacement from \(A\); the cipher's letter mapping and spare homophones are constructed from that ordered sample. The three null families are random characters from that alphabet, periodic selections by alphabet index, and nearby copies with name-blind substitution/insertion. For every latent generator state producing \((X,M)\), relabeling its working alphabet and mappings produces \((\pi X,M)\) with the same probability. Thus

\[
P(X=x,M=m)=P(X=\pi x,M=m),\qquad P(M=m\mid X=x)=P(M=m\mid X=\pi x)
\]

whenever the conditioning events have positive probability. This statement is **specific to the three selected families and the uniform arbitrary-key synthetic design**. The separate `random_pseudoword` family injects fixed-script nuclei and does not satisfy the same full-symbol symmetry. Nor does the theorem imply that an unknown historical alphabet is freely relabelable: scribal shapes or symbol classes might matter there.

The existing `copy_aware_features` depend on spaces, character equality, one-edit equality and position, so they are invariant under such bijections. `TinySignalModel`, however, adds a learned embedding of each raw symbol to those invariant features before its BiLSTM. It is not constrained to make the same mask prediction after a relabeling. This gives a direct *causal input intervention*: keep every equality, repetition, space, position and gold label fixed; change only arbitrary symbol names; measure embedding, hidden-state and output changes. A material output change diagnoses a symmetry mismatch in this trained model. It does not by itself prove that this mismatch caused its poor reconstruction or that any individual neuron implements a cipher algorithm.

The idea is related to [Kambhatla, Born and Sarkar (2023)](https://aclanthology.org/2023.findings-eacl.160/), who use recurrence encoding to remove arbitrary substitution-symbol identities, and to the general symmetry principle in [Cohen and Welling (2016)](https://proceedings.mlr.press/v48/cohenc16.html). Their successful substitution-cipher settings differ from a mixed insertion channel, arbitrary unknown language and the Voynich manuscript. [ALICE](https://arxiv.org/abs/2509.07282) studies interpretable substitution recovery, but its bijective head is inappropriate as a universal constraint when homophones, merged digrams and nulls are allowed. These sources motivate a testable invariance, not a borrowed decipherment result.

## An impossibility control that scaling cannot beat

Consider a deliberately restricted control: a signal \(Y\in A^m\) whose characters are independent uniform draws from \(A\), insert \(n-m\) independent uniform characters from the *same* \(A\), and choose an insertion merge uniformly among masks with exactly \(m\) kept positions. For every observed \(x\in A^n\), all such masks have the same likelihood. Hence

\[
P(M=m_0\mid X=x)=\binom{n}{m}^{-1}
\]

for every allowable mask \(m_0\). No decoder, regardless of parameter count or optimization, can recover a particular mask from \(X\) above chance without extra information. The actual EXP-0030 source is **not** uniform iid: repeated letters, spaces, word structure and cipher constraints may carry recovery information. The control specifies where that information must come from. A model that appears to solve the uniform-iid version is leaking a seed, channel trace, mask, or source identity.

## Decision the next measurement enables

EXP-0031 uses the frozen EXP-0030 checkpoint and held-out Finnish examples as an **exposed exploratory architecture diagnostic**. If many keep decisions change under valid renamings, a larger raw-glyph embedding model alone is a poor next bet; a recurrence/canonicalization or genuinely permutation-equivariant encoder should be compared on a new source. If predictions are already nearly invariant, the next bottleneck is more likely weak language/context inference, generator ambiguity, objective choice or data scale. Monte Carlo group-averaged predictions are an exploratory, parameter-free rescue attempt, not a new trained decoder. Neither outcome supplies manuscript meaning or a historical encoding rule.
