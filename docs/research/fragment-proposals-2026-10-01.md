# Coherent source-fragment proposals

Reviewed 2026-10-01. New solver hypothesis, not a recovered language or mechanism.
The four exposed SOURCE-GUIDE keys remain development examples. Existing
SOURCE-REVISE and SOURCE-COUPLING failures motivate this change; they do not
establish that our Latin or 23-to-6 variable-unit assumptions fit Voynich.

## Primary-source review before implementation

- Hauer, Hayward and Kondrak, *Solving Substitution Ciphers with Combined
  Language Models* (2014), [§§3–5](https://aclanthology.org/C14-1218.pdf),
  selected sections inspected. Their candidate keys arise from compatible
  word n-grams, altering multiple letters together, followed by global
  scoring and bounded search. They combine word and character information
  and acknowledge vocabulary limitations. This is direct prior art for
  coherent fragment proposals; our proposal is not a novelty claim. Their
  pattern equivalence assumes equal-length strings and a one-to-one letter
  substitution. We have neither injectivity nor observed source boundaries,
  so their pattern index cannot be copied without changing the channel.
- Ganguly, Shah and Thankachan, *Parameterized Pattern Matching — Succinctly*
  (2016), [primary abstract](https://arxiv.org/abs/1603.07457), abstract-only
  review. It explicitly defines a one-to-one alphabet correspondence. Such
  indexes are useful mathematical context, but their guarantees do not
  apply to duplicate-allowing, variable-length units. No full proof read or
  implementation of their index is claimed.
- Hauer and Kondrak, *Decoding Anagrammed Texts Written in an Unknown Language
  and Script* (2016), [§4 evaluation and §5 Voynich application](https://aclanthology.org/Q16-1006.pdf),
  selected passages inspected. The artificial anagram experiments report
  vocabulary limits and distinguish pipeline components. Its Voynich
  language/anagram assumptions differ from ours; neither that application
  nor our fragment matches establish a historical reading. Restricting
  candidates to a library can exclude the answer.
- Chu, Valenti and Knight, *Solving Historical Dictionary Codes with a Neural
  Language Model* (2020), [primary abstract](https://arxiv.org/abs/2010.04746),
  abstract-only review. It motivates constrained candidate lattices scored
  by a language model. Its word-based historical code setting is different;
  no lattice algorithm, neural replication, or transfer performance is
  asserted here.

## Own derivation and executable rules

A template x of twelve source letters matches an observed glyph prefix c
when there is a non-erasing map f from used source rows to one- or two-glyph
units such that f(x)=c[:m]. Repeated source rows reuse the same literal unit;
different rows may share units. Each first occurrence branches over the
two possible lengths, then its unit is forced by the observed substring.
Each later occurrence is a literal equality check. This enumerates every
compatible used-row dictionary once: any valid dictionary fixes its first
occurrence length, and induction forces all subsequent offsets and units.
The end m is determined by that dictionary, not supplied from a true length.

The library is the existing source count archive's **20,202 retained depth12
contexts**, not the roughly 1.45 million contexts summed over depths. It is
neither a dictionary of every Latin expression nor a complete source support:
the smoothed Markov model assigns positive probability outside this set.
Counts, preprocessing, authors and checksums remain unchanged. No new corpus,
model training, model weights, or final holdout is accessed.

At each observed offset, scan every template and binding, with an explicit
50-million-node failure cap. Retain the union of the top16 under log(count)
and top16 under log(count) minus M log42, where M is the number of used rows.
The second ranking multiplies an empirical fragment frequency by the iid
partial-key prior; constants shared by all templates are omitted. Both
rankings are proposal heuristics. Neither equals the complete-text marginal
posterior or accounts for location selection. Ranking truncation is recorded
separately from incomplete enumeration; a node-cap hit raises a failure.

For each two-record case, choose sixteen evenly spaced glyph anchors per
record and their valid offsets −1,0,+1. This gives at most96 windows and
requires no inferred source boundaries. A nearby true boundary exists for
1/2-unit channels, but a selected anchor need not yield an in-library true
twelve-letter fragment. The fit never receives true text, key, boundaries,
source length or coverage diagnostics.

Overwrite the template's used rows in the incumbent whole key, preserving
all other rows. Evaluate every new whole key under the **unchanged original
all-source-path native likelihood** and uniform full-key prior. At each of
three rounds all candidates use the same round-start key; the next round
starts from the earliest globally highest-scoring retained key. This permits
composition of compatible fragments while retaining the baseline. There is
no claim of detailed balance, an unbiased posterior, exact MAP, irreducibility,
or convergence. Individual fragment compatibility does not guarantee that a
whole key supports both complete ciphertext records; zero masses stay zero.

The paired random arm uses the same ordered masks and target lengths, with
literal targets drawn uniformly from units of that length (seed77111+case).
It tests whether source-coherent choices are more useful than length/mask
matched random changes. It does not test a shuffled or structured-nonsense
manuscript null and need not expend equal CPU or score identical unique counts.

## Interpretation and next decision

Close both arm predictions before inspecting truth. Then measure whole-key
score, edit distance, exact readings, used-row recovery, true boundaries,
true fragment library inclusion, and whether the ranked proposals kept its
binding. This separates unavailable library support from lost ranking and
whole-key search/reader failure. A good partial template can still be absent
from the full scored bank, and a lower-error output is not semantic decoding.
If this fails, do not inflate it into evidence that Latin is wrong. Determine
which of library coverage, proposal ranking, consistent composition or
global scoring limited recovery before a new registered method.
