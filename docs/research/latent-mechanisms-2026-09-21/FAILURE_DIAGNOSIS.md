# Failure diagnosis before latent-geometry experiments

2026-09-21. Exploratory inspection and bounded CPU diagnostics of existing checkpoints; no training, manuscript scoring, fresh final-holdout evaluation, paid service, or claim of decipherment. This document distinguishes archived measurements, new development-set measurements, source-code deductions, and proposed explanations. It does **not** establish one unique causal explanation for every failure.

## Immediate conclusion

WMD-0001's immediate failure is a **joint inference and selection failure inside a representable synthetic domain**. The checkpoint's masked reconstruction metric conceals large differences among fields and inference conditions; its all-masked proposal prior generalizes poorly to the development grammar/morphology combination. Its compiler prunes syntactic hypotheses before checking semantic execution. Its copy alternative accepts the same surface syntax without comparable semantic restrictions. Existing final-readout patches cannot determine which internal computation causes these failures.

The appropriate next experiment is a controlled decomposition of evidence use, representation gauge, joint consistency, and semantic search—not another final-layer probe or more training selected solely by aggregate masked loss. Some weak useful information exists: the results do not justify calling the model wholly context-blind or every probability distribution collapsed.

## Inputs and provenance

Read repository `AGENTS.md`, `MEMORY.md`, latest `NOTEBOOK.md`, [research protocol](../PROTOCOL.md), [communication system](../../COMMUNICATION_SYSTEM.md), [mechanisms review](../world-models-diffusion-2026-09-21/MECHANISMS_AND_IDENTIFIABILITY.md), and its [source ledger](../world-models-diffusion-2026-09-21/MECHANISM_SOURCES.md). The external-memory reminder to distinguish failed search from wrong hypotheses agrees with the repository's current protocol; it was not used as experimental evidence.

Checkpoint: `outputs/WMD-0001/joint/best.pt`, step 2,000, SHA-256 `f65cda998500f9bd2c750320b56f3867cf5153ea106d03c4788d71e600b3479a`. Existing configuration: 4,944,924 parameters, six layers, width 256, eight heads; `JointLayout(24,128,4)`; training seed 41021. Diagnostics used the existing `.venv`, CPU, two Torch threads, `eval()` and `no_grad()`.

Source hashes for `schema.py`, `denoiser.py`, `search.py`, `training.py`, and `worlds.py` match the archived WMD training manifest. Current `pipeline.py` hash is `69ce2857db1df625b683374bf79f743d2666ef67a8f60ecc9ef5ebdd95553cc0`; the archived version predates the documented direct-API final-test guard. These diagnostics use `make_episode`, `collate`, and `verify_candidate`; no final-test API was invoked.

New measurements use **only the already-exposed 64 synthetic development worlds**, `make_episode(i, 'validation', layout, seed=41021, anchor_count=a)`, indices 0–63. The worlds share held-out `(VOS,prefix)`, with 22 procedures, 22 taxonomies, and 20 copying controls. The two anchor conditions regenerate the same worlds. These are exploratory development measurements, not fresh confirmatory results. No historical evidence enters them.

## What the previous causal work does and does not establish

| Record | Established | Still not established |
| --- | --- | --- |
| [EXP-0004](../../experiments/EXP-0004-results.md) | Supervised readouts recover synthetic hidden-state information near a known-filter reference; IID labels remain unrecoverable. State-direction patch KL 1.431043 is effectively the norm-matched random result 1.431330. | A readable direction need not be a causally sufficient state. There is no unsupervised decoder or isolated update algorithm. |
| [EXP-0006](../../experiments/EXP-0006-results.md) | Supervised rank-three late steering works; learned KL 0.034799 versus output-weight span 0.036336. Early alignment does not consistently beat shuffled supervision. | Late success does not distinguish a state representation from direct manipulation of output evidence. Poor early steering does not prove no state exists: intervention site, geometry, and distributed computation remain alternatives. |
| [EXP-0008](../../experiments/EXP-0008-results.md) | Text-only partition fitting finds useful familiar-key predictions, with clear new-key and omitted-family failures. Forecast-probability clustering often beats residual clustering. | Neither causal state recovery nor key-invariant inference follows. Lag-copy passes mostly through offset 1; offsets 2/3 remain near uninformed. Separate future marginals do not establish a joint process. |
| [EXP-0009](../../experiments/EXP-0009-results.md) | Earlier component replacements affect predictions after one/three shared new symbols and beat controls across three synthetic seeds. Final readout projection changes only the immediate answer. | The successful intervention transfers whole component vectors, often across eight or 128 positions. It does not isolate one latent variable, extract a transition rule, or establish long-lived transfer. Shared evidence largely resynchronizes paths by horizon 8. Manuscript confirmation fails. |
| [WMD-0001](../../experiments/WMD-0001-results.md) | One fixed pair yields selected-slot TV 0.03208 versus random 0.01085, with exact identity/full-donor controls. | All argmax outputs are unchanged. This one-pair final-readout intervention is not a semantic circuit or cross-task workspace. |

**Source-code deduction:** WMD's `hidden_patch` acts after all six Transformer layers and final normalization, immediately before the pointwise linear output head. Consequently, an untouched slot cannot change through downstream attention: there is none left. Exact preservation of other slots and exact donor reproduction at swapped slots are therefore implementation controls, not evidence of a discovered modular causal variable. Meaningful propagation tests need an earlier-layer hook and later computation, or a patch at one denoising step followed by unpatched later steps. A persistent patch at every step tests forced control and must be reported separately.

## New checkpoint measurements

### Aggregate reconstruction hides the hard fields

The registered masked validation calculation was reproduced on CPU: **0.723897108 nats**, versus archived 0.723897094. Its corruption RNG was seed `41021+10007=51028`, batch size 16, uniform masked count from 1 through each world's active-slot count, then uniform positions without replacement.

| Field | Masked targets | CE, nats | Argmax accuracy |
| --- | ---: | ---: | ---: |
| Global inverse-key entries | 479 | 2.057738 | 0.271399 |
| Keep decisions | 1,185 | 0.334965 | 0.874262 |
| Boundaries | 1,225 | 0.540128 | 0.650612 |
| Grammar | 31 | 1.703910 | 0 |
| Morphology | 35 | 0.788100 | 0.571429 |
| Family | 38 | 1.104562 | 0.263158 |

Keep/boundary fields account for **2,410/2,993 = 80.52%** of these loss targets. Grammar, morphology, and family together contribute only 104 targets. This is a measured weighting imbalance, not proof that a particular reweighting will fix recovery. Many keep decisions repeat information deterministically implied by a global key; the training objective nevertheless treats them as separate predictions.

The compiler uses a different query: all mutable latents masked, anchors fixed, noise level explicitly one. Under that query with two anchors:

| Field | Targets | CE, nats | Argmax accuracy |
| --- | ---: | ---: | ---: |
| Mutable key | 802 | 2.168523 | 0.273067 |
| Keep | 2,422 | 0.393795 | 0.860859 |
| Boundary | 2,422 | 0.690569 | 0.547069 |
| Grammar | 64 | 2.086763 | 0/64 |
| Morphology | 64 | 1.143426 | 0/64 |
| Family | 64 | 1.090735 | 25/64 |

The held-out VOS/prefix combination is not recovered by separate argmax choices. Whether this comes from insufficient learning, weak compositional generalization, posterior ambiguity, or inference-query mismatch requires new controls. The prior is not literally out of training support: all-mask corruption has positive training probability, approximately one divided by active-slot count per episode. It is nevertheless much less frequent than partial-mask conditions, and generated erroneous latents are absent from training input.

### Sequential evidence has a small measured influence in this query

Shuffle only the observation symbols within each development document, keeping its length, symbol multiset, anchor evidence, fixed-key positions, and all-mask query unchanged. The two-anchor aggregate CE moves **0.795139282 → 0.796049994**, a difference of 0.000911 nats. Key and keep argmax accuracies remain identical; boundaries change slightly. The first shuffle uses one CPU generator, seed 92341, continuously across the 64 worlds.

A separate perturbation audit reseeds that shuffle generator to `92341+batch_start` for each 16-world batch. Mean total variation over mutable slots is 0.007478 with no anchors and 0.007656 with two anchors; 119/5,966 and 131/5,838 mutable argmax outputs change. This is **low sensitivity to this order perturbation**, not a universal causal claim that the model ignores all observations. Shuffling can move inputs off the generator's support; these development perturbations alone do not localize the responsible module.

### Marginal modes must not be confused with null probabilities

With no anchors, the all-mask key argmax is null on **929/930** present-symbol slots; the remaining choice is canonical symbol 5. True null keys are 212/930. Non-null exact-ID accuracy is 0/718. With two anchors, argmax null occurs on 715/802 mutable keys; exact-ID accuracy on true non-null mutable keys is 16/590. These results are compatible with heavily fragmented signal-class marginals.

Crucially, **no key has P(null) at least 0.5** in either condition. Mean P(null) is 0.223489/0.199418. One null category can beat every individual signal category even while total signal probability is high. The apparent conflict between a null key mode and a keep mode is not, by itself, proof of miscalibration or absence of information.

| Descriptive probability check | No anchors | Two anchors |
| --- | ---: | ---: |
| Mean P(null), truly null keys | 0.235478 | 0.209471 |
| Mean P(null), truly signal keys | 0.219949 | 0.195806 |
| Pooled null ROC-AUC | 0.654386 | 0.596194 |
| Mean absolute gap between P(keep at position) and 1−P(null for its symbol) | 0.094652 | 0.097254 |

Probability-gap statistics omit positions whose symbol is anchored: denominators 2,422/2,035. AUC pools keys within correlated worlds and has no confirmatory uncertainty interval. It shows some null discrimination, not a trustworthy null decoder. The independent keep head is more signal-biased than the global key distribution; deterministic projection later discards inconsistent local choices.

### Known valid solutions exist; the model/search misses them

On these same 64 development worlds:

1. All **64 gold configurations** pass the existing verifier: 22 executable procedures, 22 compatible taxonomies, 20 nonsemantic copying cases.
2. Give `search_grammar` the true grammar/morphology/family and key log scores zero at the true value, negative infinity elsewhere, with beam 32 and expansion cap 20,000. It returns at least one valid candidate for **64/64** worlds.
3. Relabel each gold configuration's family as `copy`, leaving its key and boundaries untouched. **All 64 remain valid**, including all 44 semantic worlds.

The first two are privileged diagnostic upper bounds, not ciphertext-only recovery. They establish that these targets are supported by the declared executor and compiler path. They do not establish that the inverse is uniquely identifiable from observations, or that ordinary finite search can find it. The third exposes a structural ambiguity: copy-family validation accepts renderer-compatible strings, without evaluating their copy-history probability or independent grounding. Validity alone cannot identify a semantic source against this broader support.

The initial three neural diagnostic scripts took 6.818, 3.809, and 1.862 seconds; the copy relabel check took approximately 0.40 seconds, totaling approximately **12.9 seconds**. The document's reproduction recipe was subsequently rerun once and reproduced both tables and all three 64/64 checks. These bounded checks, brief archive reads, and artifact validation remained below the five-minute CPU budget. No new model was fitted.

## Search failure is more specific than an insufficient expansion budget

Independent aggregation of the existing compressed candidate audits gives:

| Trained denoiser rejection reason | No anchors | Two anchors |
| --- | ---: | ---: |
| Incomplete three-word clause | 165 | 172 |
| Nominal word violates stem/case morphology | 56 | 62 |
| Predicate not a declared operator | 35 | 22 |
| Total | 256 | 256 |

Thus all trained denoising proposals fail at the syntactic stage before semantic execution can demonstrate anything. Post-hoc projection enforces key/keep consistency and a few boundary invariants, but does not enforce complete word/clause structure. Exact observation re-encoding is weak here: it preserves supplied alignment and observed symbols under the proposed key and is not an independent semantic likelihood.

In `search.py`, beams retain grammar phase, partial key, boundaries, null count, and score—**no evolving world-state or taxonomy constraint**. Surface candidates are offered under each family, globally ranked, and truncated to 16 before `pipeline.py` runs the independent semantic verifier. Semantically impossible parses can therefore consume beam and final-candidate capacity. The effective beam is often below the nominal 32: 6–32, median 18/19 in the trained zero/two-anchor audits. Every true `(VOS,prefix)` search is pruned. Not exhausting the global expansion cap does not mean the relevant states survived.

Compiler semantic rejections include 104/91 impossible procedures and 14/45 reflexive taxonomies in the two trained conditions. These are failures among retained candidates, not proof that the first lost gold path was rejected for the same reason. Instrumenting gold-path survival on synthetic development data is required to locate that cause.

The final ranker is another mismatch: it sums factorized scores from a single all-mask query across the complete candidate, including many keep/boundary slots. It is explicitly not a joint posterior or normalized generative likelihood. The compiler itself uses key/global scores plus a fixed 0.15-null preference; the generated null-rate mixture is 0, 0.1, or 0.25. Neither score validates source-family identification.

## Mechanistic hypotheses and experiments that distinguish them

These are proposed successor experiments, **not results**. Freeze fresh generator seeds/keys, independent discovery and confirmation worlds, budgets, selection rules, and rejection thresholds before execution. Retain old development diagnostics as diagnostics only.

| Hypothesis | Distinguishing test | Interpretation of an informative result |
| --- | --- | --- |
| Aggregate CE rewards easy redundant fields | Compare original versus balanced field loss at fixed data/updates; select by complete gauge-aware recovery and per-family execution, not CE alone. | Better complete recovery with similar proposal budget supports an objective bottleneck; lower weighted CE alone does not. |
| Partial-truth denoising does not learn inference from observations | Stratify by corruption level; compare clean partial latents, all-mask starts, and equal-rate on-policy erroneous latents. Include text-order and evidence-removal controls. | Improvement confined to clean partial-truth queries identifies a conditional reconstruction skill, not a working inference procedure. |
| Global naming symmetries fragment coordinate predictions | Evaluate one consistent entity permutation per document, restricted by anchors; compare original targets with explicitly orbit-aware training. | Better orbit-aware structural recovery without exact ID recovery supports gauge fragmentation. Per-slot best permutations are invalid because they can describe no coherent world. |
| Compiler pruning removes viable semantics | Record when gold-equivalent prefixes disappear; compare syntactic beam, semantic-prefix beam, larger/diverse beam, and oracle-key/grammar assistance under equal expansion accounting. | Locate proposal, prefix-ranking, semantic-pruning, or final-cap failures separately. Adding only a larger global cap is not a decisive test. |
| Model uses surface priors instead of transferable computation | Paired exact glyph renamings, new keys, new grammar combinations, and multiple independently rendered views of one world; match lengths and symbol frequencies. | Equivariance and query transfer can reject simple naming/length shortcuts; familiar-key probes cannot. |
| Geometry contains a useful latent state | Compare raw residual, forecast/output span, learned supervised directions, and Jacobian-derived geometry, all selected on discovery only. Test matched swaps at earlier layers and later denoising steps with random, wrong-donor, and complement controls. | A useful geometry must improve causal transfer or executable prediction beyond output steering. Clustering/readability alone is insufficient. |

For the causal experiments, use a competent synthetic task with a known finite inference reference before making model-wide claims. EXP-0004's calibrated generator and EXP-0009's continuation setup provide an existing positive starting point; use **new contexts and keys**, since their previous tests are exposed. Require state distinction and equivalence, joint continuations, counterfactual update consistency, and cross-seed stability. For WMD, use hypothesis changes with genuine downstream consequences rather than arbitrary donor documents sharing no controlled variable.

Interface recommendation: expose hooks that separately identify `(layer, stream, position, denoising_step)`, capture pre/post attention and MLP writes, and permit a single intervention followed by ordinary computation. Return a source manifest, input identities, per-factor probabilities, accepted/rejected candidates, projection changes, gold-equivalent survival diagnostics on synthetic-only runs, and explicit oracle-assistance flags. Avoid a single ambiguous `hidden_patch` result being described as a circuit.

## J-space: useful direction, specific limits

The user's reference is Anthropic's [A global workspace in language models](https://www.anthropic.com/research/global-workspace), July 6, 2026, read here at the public summary level. It describes a Jacobian lens connecting internal directions to potentially reportable words, together with reporting, control, multi-step reasoning, and flexible-use interventions. Those task properties go beyond merely finding directions with large output derivatives. The accompanying paper and implementation need separate method-level inspection before a faithful replication.

WMD does not have a natural-language vocabulary with learned word meanings, instructions to report hidden thoughts, or the cited reasoning tasks. Its token IDs are shared typed-field categories; IDs denote different things at key, keep, and family positions. Applying a language-model word lens to that shared head would create misleading semantic labels.

A rigorous **symbolic analogue**, proposed here, would define an output functional before computing derivatives: for example, gauge-invariant key-equality relations, grammar-compatible boundaries, or independently scored answers to finite world-state queries. Compute sensitivities from earlier residuals through later computation to those functionals. Compare a Jacobian-derived subspace with the output-weight span, random subspaces of matched dimension/energy, and context-dependent Jacobian controls. Test whether a single variable swap transfers appropriately across independently held-out queries and denoising steps while preserving irrelevant variables. A final-head Jacobian is just the linear readout and is a necessary trivial control.

The existing checkpoint can support layer/slot/step sensitivity, evidence-use, and constrained inference diagnostics. It cannot by itself support natural-language reportability, instruction control, or a general workspace conclusion. A multitask finite-world model would add those symbolic tests but changes the scientific target; an open-weight language-model implementation would test the original method more directly. Neither would establish a Voynich meaning without independent manuscript evidence.

## Reproduction recipe for the primary new table and oracle check

Run from the repository with `.venv/bin/python`; this uses existing development worlds only. The script below reconstructs the primary masked and all-mask measurements and privileged reachability check. The additional shuffle/count/AUC diagnostics are defined completely above, including batch and RNG policies; they are exploratory supporting analyses rather than registered gates.

```python
import torch
from voynich.communication.training import load_checkpoint
from voynich.communication.pipeline import make_episode, collate, verify_candidate
from voynich.communication.search import search_grammar

torch.set_num_threads(2)
model, layout, record = load_checkpoint('outputs/WMD-0001/joint/best.pt', 'cpu')
model.eval()
episodes = [make_episode(i, 'validation', layout, seed=41021, anchor_count=2)
            for i in range(64)]
blocks = {'key': layout.key_slice, 'keep': layout.keep_slice,
          'boundary': layout.boundary_slice,
          'grammar': slice(layout.global_start, layout.global_start + 1),
          'morphology': slice(layout.global_start + 1, layout.global_start + 2),
          'family': slice(layout.global_start + 2, layout.global_start + 3)}
for mode in ('registered_masked', 'all_mutable_masked'):
    rng = torch.Generator().manual_seed(51028)
    totals = {name: [0., 0, 0] for name in blocks}
    with torch.no_grad():
        for start in range(0, 64, 16):
            batch = collate(episodes[start:start + 16], layout)
            clean = batch['clean']
            active = clean.ne(0)
            if mode == 'registered_masked':
                masked = torch.zeros_like(active)
                for row in range(len(clean)):
                    positions = active[row].nonzero().flatten()
                    n = int(torch.randint(1, len(positions) + 1, (1,), generator=rng))
                    chosen = positions[torch.randperm(len(positions), generator=rng)[:n]]
                    masked[row, chosen] = True
                noise = masked.sum(-1) / active.sum(-1)
            else:
                masked = batch['fixed'].lt(0)
                noise = torch.ones(len(clean))
            logits = model(clean.masked_fill(masked, 1), batch['condition'], noise,
                           latent_types=batch['latent_types'])
            logits = logits.masked_fill(~batch['allowed'], -torch.inf)
            for name, section in blocks.items():
                use = masked[:, section]
                prediction = logits[:, section][use]
                target = clean[:, section][use]
                if len(target):
                    totals[name][0] += torch.nn.functional.cross_entropy(
                        prediction, target, reduction='sum').item()
                    totals[name][1] += (prediction.argmax(-1) == target).sum().item()
                    totals[name][2] += len(target)
    print(mode, {k: (v[0] / v[2], v[1] / v[2], v[2]) for k, v in totals.items()})

gold_valid = oracle_valid = copy_relabel_valid = 0
for episode in episodes:
    gold = layout.unpack(episode.clean, episode.observation)
    gold_valid += verify_candidate(episode.observation, gold)['valid']
    copy_relabel_valid += verify_candidate(
        episode.observation, {**gold, 'family': 'copy'})['valid']
    scores = [[-float('inf')] * (layout.alphabet_size + 2)
              for _ in range(layout.alphabet_size)]
    for index, value in enumerate(episode.world.inverse_key):
        scores[index][value] = 0.
    cfg = episode.world.config
    result = search_grammar(episode.observation, grammar=cfg.grammar,
        morphology=cfg.morphology, family=cfg.family, key_log_probs=scores,
        beam_size=32, max_expansions=20000, max_candidates=16)
    oracle_valid += any(verify_candidate(episode.observation, row['hypothesis'])['valid']
                        for row in result['candidates'])
print({'gold_valid': gold_valid, 'oracle_valid': oracle_valid,
       'copy_relabel_valid': copy_relabel_valid})
```

## Status

Completed read-only diagnosis plus this document. The embedded reproduction script parsed and ran successfully, every local document link resolved, and the saved checkpoint hash was unchanged after diagnostics. No source edit, new fit, final-test scoring, or commit was performed by this diagnostic subtask. The main campaign must register its fresh experiments, preserve this exploratory status, update the notebook, and validate/commit/push the combined checkpoint. The evidence supports prioritizing joint consistency and inference competence before interpreting geometry as an executable semantic mechanism.
