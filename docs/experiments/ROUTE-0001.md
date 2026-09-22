# ROUTE-0001: does unchanged earlier context bypass a local edit?

Registered after JSPACE-0001/NEURON-0001 failed, before new outcomes. **Exploratory development-only diagnosis**, not a fresh confirmation and not a replacement verdict. Same model, precision, fitted lens and frozen accepted answers; no retraining, dose search, final examples or manuscript scoring.

## Evidence and rationale

Single-position country swaps produced17/151 final successes; full donor residuals at that position65/151. All tested early development layers were resistant. Those observations motivate testing position coverage. The [reference methods](https://transformer-circuits.pub/2026/workspace/index.html) describe swapping across token positions, including a band of layers for flexible use. This successor isolates position coverage at one layer at a time, and remains a restricted adaptation.

An earlier residual state can supply keys/values to later attention, bypassing an edit at the final prompt position. Replacing all positions at layerL makes the *first-token* downstream computation donor-like when sequence lengths match. It does not replace cached keys/values at layers<=L, so later generated tokens may still differ from the donor. First-token agreement is a numerical positive control; complete answer agreement is measured separately.

## Fixed inputs and interventions

Use all48 anchored-alias development records, both paraphrases and pair directions. Source/donor rendered prefixes have exactly equal length. Add12 copying controls: for each of the six distinct source/donor codebook contexts, append `Copy only the literal word amber.` or `Copy only the literal word velvet.` after the existing blank-line separator. Keep the donor codebook and same literal output. Assert distinct IDs, equal rendered lengths, source/donor pairing and no final examples.

Layers23 and27 (zero based), chosen from the already exposed developmental resistance/onset. **Primary diagnostic layer23; layer27 descriptive.** At each layer compare nine conditions: identity; last-position, earlier-positions-only and all-position full donor residual differences; last/earlier/all-position country-coordinate swaps; all-position raw-output-coordinate swaps; all-position matched random changes. Swap strength1; raw and Jacobian rows identical to the completed study. Random edits match each position's Jacobian-swap displacement norm, with independent deterministic directions, seed51031. No edits to newly generated positions.

60records ×2layers ×9conditions =1080 generations, max12tokens. Natural donor states come from the matched question only. Broad donor patches are localization controls, not selective concept edits. Copying probes only a narrow nuisance behavior.

## Numerical gates, metrics and resources

Before scientific scoring, require full-prefix field execution to reproduce the established adapter for clean, last-site and multiple-site generic patches (same argmax, max logit error<.002). Require every all-position donor first-token output to match the clean donor argmax, with max logit error<.002, and every identity generation to match exactly. Abort on failure; preserve partial logs.

Report complete-answer donor success among both clean source and clean donor correct, alongside the original clean-source-only denominator and all-case rates. Report first-token donor agreement, retained original answers, every relation/family/pair and all copy controls. At primary layer23, label the **earlier-context bypass diagnostic supported** only if earlier-donor success exceeds last-donor by>=.20 and earlier-donor copying>=.95. Separately label **position coverage improves the restricted lens** only if all-position swap exceeds both last-position swap and all-position random by>=.20, and swap copying>=.95. No broad significance claim; shared facts and only three pairs remain.

Freeze exact new source/task/lens hashes before generation. Retain all input IDs, lengths, states, outputs, errors and norms. Local hard cap20minutes,45GB MLX allocation, no paid service or simultaneous dense model. Expected~8–12minutes from previous generation throughput. A negative diagnostic narrows this explanation only; it does not justify unlimited retuning or treating the consumed final split as fresh.
