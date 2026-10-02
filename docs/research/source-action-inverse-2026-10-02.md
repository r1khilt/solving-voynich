# Supervise the reading trajectory and expose its shared dictionary

2026-10-02 PDT. New engineering implementation, not a trained competent inverse.
The benchmark registration fixes cost measurement; substantive language training
and a fresh recovery panel require their own prospective registration.

## Evidence that changes the task

Both unchanged TEMPERED-RECOVERY-001 and TEMPERED-RECOVERY-DIAG-001 full replays
have now passed. Recovery remains FAIL. Supplying the generating dictionaries
leaves14edits in1152unique source letters; every original bank misses available
much better likelihoods, and every particle lacks the generating used-code
inventory even after arbitrary row permutations. The full-key posterior bank
mass bounds are extremely small under the declared model, not historical or
plaintext-marginal certificates.

This is an inverse-search bottleneck on four development synthetic keys. It
does not establish Latin, this channel, source adequacy or identifiability for
Voynich. A stronger source alone cannot make an absent dictionary appear.
Do not reinterpret the positive known-key control as recovery.

## Already tried; do not relabel old work as a new solution

SOURCE-FRAGMENT-001 used twelve-letter source templates with partial-key
overwrites and greedy full-key acceptance. All eight cells retained their
starting key. Its literal template library excluded every selected true window;
shorter fragments had much better coverage but greedy composition was not
tested. SOURCE-STATE-SYSTEMS-001 subsequently DID implement compatible joint
source/key state search. SOURCE-PRUNE-DIAG-002 found wide-beam true-path loss
after only5–6source letters in three failing cases. Existing source particles,
Bellman/shared-dictionary guides and tempered inventory kernels also remain
relevant failed or bounded methods; this work is not their first invention.

JOINT-KEY-TRAIN-001 already completed four20k-update fits, including95M models.
Its proper dictionary density improved, but no complete key was recovered and
larger fits lost the preregistered capacity comparison. Actual Latin-window
training differs from the statistical source that generated the search panel.
The old order intervention did not show strong suffix-order dependence; it
did not identify a neuron or prove all sequence information was ignored.
The existing [next-objectives memo](key-inverse-next-objectives-2026-10-01.md)
already proposed plaintext supervision and action policies. The new contribution
is an executable reading-action objective with auditable prefix-only bindings,
not a newly invented general idea. The earlier communication action model
learned an entirely different finite seen-input domain, not this unknown-key task.

## Primary-method review and its limits

[Corlett/Penn2010](https://aclanthology.org/P10-1106.pdf), selected§§2–4, uses
partial mappings and relaxed Viterbi scores as admissible A* bounds. Its channel
is bijective single-character substitution; reported tests have thousands of
characters and disclose the space mapping. The monotone relaxation principle
is useful; its exactness and runtime do not transfer to ambiguous variable units.
[Hauer/Hayward/Kondrak2014](https://aclanthology.org/C14-1218.pdf), selected§§3–6,
uses word-based multi-letter mutations and finite MCTS/beam alternatives.
Equality-pattern filters rely on bijective substitutions and cannot be imposed
on duplicate-allowing length1/2 codes. This supports considering coherent moves,
not an accuracy forecast for our channel.

[Kambhatla/Born/Sarkar2023](https://aclanthology.org/2023.findings-eacl.160.pdf),
selected§§3–5/7–8, trains recurrence-encoded ciphertext-to-plaintext generation
and compares ciphertext-plus-plaintext versus target-only supervision. Its
main homophonic setting has substantially more synthetic examples and different
channel/length/spacing constraints; Borg is not a Voynich success. Its evidence
supports explicit sequence objectives and controls. Our encoder currently sees
all observed ciphertext and has NO causal ciphertext-reproduction auxiliary
loss. Thus this is not a replication of its best causal joint objective.
Attention-derived keys in that paper do not supply a causal mechanism here.
Prior Voynich literature review remains in the project knowledge records; no
new prior Voynich use of this exact action environment was established by this
selected-method pass. No frontier-scale ability is inferred from these papers.

## Implemented environment and neural proposal

State consists of partial row-to-unit bindings, consumed glyph offsets and
past plaintext for each record. Select the unfinished record with the smallest
consumed-glyph fraction, ties by index. An action chooses a source row and
emission length1or2. The observed next glyph(s) determine that unit. An unbound
row binds once; a bound row can only emit its existing unit. Different source
rows may share units. Every action consumes at least one glyph. Stop when all
observed records end. The model receives neither gold lengths nor future
bindings nor supplied source boundaries. Record-end offsets are observations,
not hidden source lengths. A path can dead-end and is not restarted or repaired.

Gold source/key pairs compile TRAINING targets. Each network snapshot is
independently reconstructed from preceding actions, with every consumed source
prefix literally re-encoded and no unvisited bound slot allowed. Unused gold
key rows never enter features or targets. The next action itself enters only
the loss; only the preceding action reaches the decoder query. Future action
states are causally masked at every decoder layer.

The new transformer has a full-cipher encoder, causal action decoder, observed
offset/record embeddings and explicit partial binding memory. Each row/unit
PAIR has its own embedding. Summing separate row and unit embeddings would
erase the assignments when pooled; the pair encoding avoids that particular
symmetry. Pooling23slots is still a finite architectural choice, not proof of
perfect retrieval. A prospective matched `binding_input=False` control retains
identical parameters, legal-action constraints and past-action context but zeros
the binding contribution to the neural query. Record IDs deliberately break
record-permutation invariance; position indices reset for each cipher record.
This differs from the earlier invariant whole-key proposal and requires explicit
order controls in any future empirical study.

The next experiment should compare explicit binding access under matched source
episodes, source/channel assumptions, seeds and actual resources, with complete
readings/used-key recovery as primary outcomes. Losses of a23-row dictionary
head and a variable-length action head are different quantities and cannot
rank architectures directly. Teacher-forced improvement alone cannot qualify
free-running behavior, rare units, transfer or latent circuits. Once free-running
behavior is competent, causal slot/prefix interventions can distinguish use of
dictionary memory from copying/source-only guessing. Mechanistic analysis is
enabled by the exposed state, not yet performed or justified by trained behavior.

## Probability account and finite checks

For a deterministic schedule, each compatible complete plaintext tuple has
one action order and one binding pattern. With iid-uniformU-unit row prior,
its joint source/channel mass after integrating unused rows is
`Q(X) U^(-M(X))`, where M counts distinct visited rows. Geometric EOS/reset
factors belong to Q; binding penalties are applied ONCE, not per letter.
The new environment imposes the literal constraints but does not replace the
existing source likelihood or claim to compute this evidence itself.
Exact rational tests compare trace enumeration with FULL dictionary/text
enumeration over36two-record panels, including duplicate codes and unused-row
completions. A separate fourteen-record one-row test verifies that uniform legal
action probabilities sum to1over successful AND failed leaves.

The learned proposal has normalized conditional action probabilities. Its
whole-path density is a product along the visited trajectory, including failed
prefixes. Mean whole-path NLL is used, not division by gold target length;
target-dependent normalization would tilt the conditional distribution.
There is no learned or statistical source prior silently multiplied into this
proposal. Finite floating softmax and CPUcategorical arithmetic are numerical
implementations, not exact rational densities or full-support guarantees.

Terminal partial dictionaries have overlapping full-key completions and can
arise through multiple plaintext trajectories. Their probability is a SUM over
those paths. Returned path log probability is NOT that dictionary marginal.
No key Metropolis/importance weight may use it as `q(K|C)`. Completing unused
rows, conditioning on success, greedy repairs or beam selection also change
the law. This interface is initially for verified SEARCH, with unchanged full-key
rescoring/reading and explicit failure rates; no posterior/evidence/mixing claim.

Tiny CPUdouble tests verify future-target and unused-gold-key invariance, literal
constraints, variable-length padding, whole-path batch loss, all parameter
gradients, zero-gradient binding ablation, and sampled rollout-density replay.
One fixture initially expected the wrong error message and was corrected;
this was preparation, not a repeated scientific source or training run.
The reference rollout re-encodes the cipher at every action and has no KV cache.
That cost is unresolved and must be measured/engineered before long-path use.

## Next bounded step

[SOURCE-ACTION-BENCH-001](../experiments/SOURCE-ACTION-BENCH-001.md) measures
maximum training shapes on the Mac, with5random-only updates per declared
scale and short CPUdouble/MPS checks. No Latin, benchmark-panel, source-count,
manuscript or holdout files are opened; no language fit is launched. Projected
training time excludes inference/audit/preparation, and cannot alone admit the
next campaign. Preserve both original empirical failures and all frozen inputs.
Voynich remains UNSOLVED.
