# Deep review: learning rules, testing mechanisms, reaching meaning

Completed research checkpoint: 2026-09-21. **93 source records covering 92 distinct works and resources.** Reading depth: 75 selected-section/documentation records, 17 abstract-only records, and one fully read short research note. DreamCoder appears in two topic ledgers but counts once. This is a substantial targeted review, not an exhaustive survey or 92 full-paper reads.

**The main recommendation is to build a learner of unfamiliar encoding processes and test its state updates explicitly.** Training a larger Voynich-only predictor remains a useful comparison, but our completed experiments do not support making it the sole route to decipherment.

No new model was trained during this review. No manuscript word or historical rule was decoded. The final manuscript holdout remains unopened. The successor architecture below is designed, not yet implemented.

## Start here

| Document | Purpose |
| --- | --- |
| [Next design and experiments](NEXT_DESIGN.md) | Concrete representations, model interfaces, objectives, controls and bounded work packages |
| [Architecture review](ARCHITECTURES.md) | DeepSeek V4/V4.1, Qwen, Kimi, Mamba, recurrent expressivity, memory, tokenization and scarce-data training |
| [Mechanistic interpretation](INTERPRETABILITY.md) | Causal abstraction, sparse features, circuit fidelity, state updates and selective interventions |
| [Decipherment review](DECIPHERMENT.md) | Prior Voynich work, substitution methods, historical scripts, images and semantic anchors |
| [Theory review](THEORY.md) | Identifiability, HMMs, predictive states, learning across tasks, uncertainty and model comparison |
| [Source catalog](CATALOG.md) | Searchable bibliography with primary links and reading-depth labels |
| [Source audit](SOURCE_AUDIT.md) | Corrected metadata, checked recent claims, version conflicts and coverage limits |

## What changed in our thinking

**1. Learning one spelling system is easier than learning how to infer a new one.** EXP-0008 improved familiar synthetic keys but failed its stronger unfamiliar-key criterion. The next learner should encounter fresh tasks and keys throughout training. We should explicitly test what happens when every observed symbol is renamed, then separately change the actual mechanism.

**2. We need to predict combinations of future symbols.** Two processes can give identical probabilities for each future position while producing different pairs. The theory review gives a simple example. A useful hidden state must preserve those relationships and update correctly as observations arrive. Attractive clusters are not enough.

**3. A small explicit model is a strong scientific tool.** A probability vector over hidden states and a transition table make predictions and updates inspectable. Compare that with a neural recurrent model and our compact transformer. If the explicit model wins, use it; if the neural model wins, determine which capability accounts for the difference.

**4. Interpretation must explain continuing behavior.** Our earlier broad synthetic patches changed later outputs, but did not isolate a state-only variable. The next test changes one candidate variable, checks later joint predictions and verifies that unrelated behavior remains intact. Sparse features and circuit diagrams come after a model demonstrably solves the controlled task.

**5. Meanings need independent constraints.** A text predictor can discover regularities without knowing which ones concern plants, numbers or procedures. A real decoder must state its rules, account for exceptions and make independent predictions. Image/layout evidence and historically grounded language/channel assumptions could help, but cannot be selected only because they match a preferred translation.

These are project conclusions drawn from the review and [completed campaign](../../experiments/CAMPAIGN-0001-results.md), not new empirical findings.

## Newer research that matters

The review checked primary sources published or released after April 2026, including the September DeepSeek V4.1 report, June scarce-data regularization, May identifiable sparse autoencoders, June reproducible subspaces, August interference analysis and June historical-image/cipher studies. Some bibliographic pages have conflicting dates; those remain explicit in the audit.

The newer work changes candidate components and evaluation methods. It does not remove the small-data problem or guarantee hidden-rule recovery. Frontier-scale architecture improvements must survive controlled comparisons on our tasks.

A particularly useful correction: improved sparse-feature reconstruction can accompany worse feature stability on real activations. We therefore measure reconstruction, stability and causal usefulness separately. [Identifiable SAE paper, Tables 1–2](https://arxiv.org/pdf/2605.31245)

Another: direct image decryption trained with a known cipher tests a different problem from discovering an unknown key. [Copiale image study](https://arxiv.org/html/2606.27700v1)

## What the Mac should do next

The proposed sequence starts with encoding transfer, then explicit joint-state inference, then selective causal tests. Manuscript representation checks follow only when the synthetic controls qualify. Initial budgets total at most six hours for those four packages, preceded by a hardware pilot; these are ceilings, not measured runtime estimates or launched jobs. The design includes conservative memory targets and no paid APIs.

Useful compute here means more independent tasks, seeds, competing mechanisms and interventions. It need not mean the largest model that fits in 64 GB. A model that can recover an unseen simple rule is a more informative stepping stone than a larger model that only predicts familiar text better.

Before execution, freeze exact generators, splits, seeds, statistical thresholds, source revision and measured resource estimates in experiment registrations. All previous synthetic final pools have been exposed and are unsuitable for adaptive confirmation.

## A durable, searchable knowledge base

The four `*.sources.json` ledgers are authoritative. Each entry records authors, version/date, primary URL, inspected sections, supported claims, limitations and a project decision. The catalog is rebuildable and searchable without paid embeddings or a database:

```sh
python3 scripts/research_catalog.py --query "joint" --depth selected_sections
python3 scripts/research_catalog.py --decision adopt_candidate
python3 scripts/research_catalog.py --check
```

Claims from a source, our proposed extensions, implemented methods and observed outcomes remain separate. Candidate decisions mean “worth testing,” not “validated on Voynich.”

The ambition remains actual decipherment. This checkpoint supplies a much stronger route for testing whether our tools can recover rules at all, and identifies what additional evidence would be needed to connect those rules to meaning.
