# An exact conditional teacher for a model that can revise its world

Research proposal, not a trained architecture or registered experiment. The implemented
[window mechanisms](window-repair-2026-10-10.md) are still awaiting scientific finite
qualification and original-source cost admission. The [replica comparison](../experiments/TEMPERED-READING-RECOVERY-001-results.md)
failed recovery; its full audit subsequently passed. None is a Voynich reading.

## What the failed solver actually needs to learn

The earlier append proposal predicts the next legal source-row/unit-length action from
observations and its own previous bindings. Most teacher actions reuse existing bindings;
literal consistency comes from the external mask. Its supplied dictionary embedding is
not evidence that it inferred an unknown global key. A completed wrong reading has no
append action. Increasing hidden width leaves that interface limitation intact.

The new symbolic action instead means: choose an observed interval and replace its entire
current interpretation. The candidate world contains a complete hypothesized text, its
visited key and literal alignment. Actions can undo a mistaken equality, split/merge
units or change an interior-only binding. The environment computes consequences exactly.
This is a coding-state world/action model, not a physics world model or a claim that the
manuscript uses this channel. A learned component would propose revisions; it need not
learn a second approximate simulator for already exact symbolic dynamics.

The exact cold teacher, conditional on a selected window's unchanged exterior, assigns

    pi(z | outside,cipher) = T(z) / sum_{y in C} T(y).

It evaluates the whole future source consequence until contexts synchronize. It does not
assume the current reading is correct, use Gold to choose a repair or force the generating
letter as a target. A teacher can prefer a locally wrong interpretation when the exterior
is wrong. That limitation is scientifically useful: imitating this teacher cannot be
mistaken for learning an oracle decipherer.

## Architecture hypothesis, beyond forecasting glyphs

Encode the observed records, current source hypotheses with observed-span alignment, and
visited row/unit pairs. Use repeated cross-attention between observed positions, candidate
source positions and dictionary rows, followed by an iterative shared state. Candidate
repair embeddings encode the window, one/two proposed source rows and binding changes.
A proposer scores legal complete-world repairs; an optional value head estimates the
longer search benefit. This is a proposed recurrent inference architecture, not an
implemented model, proven workspace or argument for any particular parameter count.

Training could combine a proper conditional distribution loss on complete enumerated
teacher families with separately controlled long-horizon revision supervision. Supplied
candidate bindings must remain inputs, not hidden Gold. State distributions must include
wrong free-running hypotheses, not only true teacher prefixes. Source uncertainty and
language/channel families need explicit controls; supervising the existing Latin critic
does not add semantic or historical evidence. The old interrupted training is preserved;
any fit would require a new seed/data/compute registration, never a quiet restart.

There are two distinct possible uses. First, amortize an expensive conditional table so
the model proposes a promising local repair cheaply. Correct MH then needs its actual
forward/reverse probability, including the window selection and any quantized support;
it cannot inherit uniform-family symmetry. Second, use the model only to order candidates
for a bounded search while exact target/literal validation decide retention. Search scores
then are not posterior samples. These uses require different claims and audits.

Neither the new teacher nor a bigger model fixes long-range coordination automatically.
If local conditional Gibbs also fails known-text recovery, more capacity to imitate it is
not the primary remedy. The first empirical question is whether adding the actual repair
action improves the sealed synthetic inverse task. Only then does measuring learned
amortization efficiency make sense. Original-source table cost must be measured before
registering training-data volume, batch size, model size or a paid compute campaign.

## Diffusion connection, with the distribution stated

Uniform compatible local replacements form a reversible corruption kernel with respect
to uniform counting on the finite visited-reading space, using fixed observed anchors and
invalid-boundary identities. Starting from a generated true reading and applying a fixed
number of such steps yields literal but potentially wrong hypotheses. This could support
denoising supervision. It is a discrete constrained corruption process, not a trained
diffusion model, and uniform collapsed states are not the original independent full-key
prior. A reverse model would depend on the starting-data distribution and recorded noise
law. Calling a repair denoising does not establish a correct posterior or historical code.

The exact cold heat-bath is instead a target-invariant inference kernel. These are two
different uses of the same legal state graph. Their reverse distributions and objectives
must not be conflated. A degree-k uniform MH step has irrational acceptance but qualified
bounded decisions; normalized irrational heat-bath weights are still unsupported.

## Mechanistic interpretation should test repair decisions

Follow [the repair-sensitive mechanism note](repair-sensitive-mechanisms-2026-10-10.md):
probe variables the model must infer, rather than simply read from an explicit key input.
Candidates include uncertainty among unseen bindings, whether a repeated observed unit
needs distinct source letters, and which wrong committed interpretation demands revision.
Separate model outputs from symbolic candidate-family changes. A hidden patch with that
family fixed can test downstream neural computation; a symbolic mask change can itself
cause a discontinuity without any learned causal circuit.

Fit geometry on discovery keys and erroneous states, then test selective necessity and
rescue on held-out recipients/seeds. Measure actual repaired-read accuracy and teacher
probability/log loss under norm/site/wrong-direction controls; large activation shifts
or token cosine alone are insufficient. Aggregate future intervention effects before
claiming a shared subspace, and compare supplied-binding retrieval with unseen-binding
inference. The previously failed JSPACE assay and task-specific TEACH repair results
remain limits, not evidence that this new model contains a global workspace.

These are executable research directions once the symbolic repair and inference outcome
qualify. They are not new experimental observations, a parameter-scale promise or a
retroactive success story for the failed replica search.
