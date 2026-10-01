# Bellman-backed completion potentials for early source/key search

2026-10-01. Own proposed method with finite mathematical validation; no empirical recovery outcome yet. Motivation: [pruning diagnosis](../experiments/SOURCE-PRUNE-DIAG-002-results.md), where three wide true paths die after6/6/5source letters.

## Relevant primary work and read depths

Kambhatla, Bigvand and Sarkar (2018), introduction and sections2–3.3, use neural language models and global rest-cost estimation to rank partial substitution decipherments. Their generated completions and frequency heuristic concern1:1/homophonic substitution, rather than our duplicate-allowing one/two-glyph units and unknown segmentation. It supports investigating future-aware ranking; sampled rest costs are not an admissible bound or a shared-key future marginal. We use the already qualified statistical source, not their pretrained neural model. [Paper](https://aclanthology.org/D18-1102.pdf).

Lin, Chen and Liu (2013), introduction and selected sections4.1–4.2 through finite-state exact-lookahead sums, distinguish delayed weighting from using future observations to construct proposals; their marginalization can be expensive. Their SMC guarantees require proper importance weighting/resampling. This deterministic beam experiment claims neither posterior sampling nor their variance guarantees. [Paper](https://arxiv.org/pdf/1302.5206).

Zhao et al. (2024), selected section3.1–3.2 optimal twist recursion/Propositions3.2–3.3 and the selected section4 soft-Bellman interpretation, motivates estimating expected terminal compatibility rather than scoring a locally attractive prefix alone. Their learned twists and LLM experiments are not evidence of our critic competence or Voynich recovery. The PMLR PDF link returned an error; the arXiv PDF was inspected instead. Our method is an explicit finite backup without neural learning. [Paper](https://arxiv.org/pdf/2404.17546), [publication](https://proceedings.mlr.press/v235/zhao24c.html).

Prior Voynich-specific assumptions remain those reviewed in [pruning methods](source-prune-diagnosis-2026-10-01.md): Hauer/Kondrak's anagram/abjad and language-identification experiment does not establish the manuscript's language or channel. No new manuscript/corpus evidence is used here.

## Own recursion, interpretation and limits

Let a(s,c) be the original source/context/once-binding-prior edge coefficient, including immediate EOS when a record finishes. Choose one record deterministically by observed-glyph progress. No branch over possible schedules is allowed. With true shared-key future completion mass H(s),

`H(s) = sum_c a(s,c) H(c)` and `H(complete)=1`.

Let h0 be the current full-remaining-cipher iid guide, with root source frequencies, unknown units resampled per occurrence, epsilon1e-8 smoothing and log floor−2000. Define an unfloored backup `h(d+1)(s)=sum_c a(s,c) h(d)(c)`, with completed states absorbing at1. The NEW compiled scorer backs up exactly ONE source action, carrying the child's newly bound key row into BOTH records' remaining iid tables, while using the actual source context for that next action. It sums possibilities; it does not select a best next letter or multiply a heuristic into original edge weights.

The tail still relaxes all subsequently unassigned-row sharing and later source contexts. One step is a heuristic, not the true H, a lower/upper bound, an unbiased evidence estimator or a learned world model. The final ranking clamps backup log values to−2000; zero support in ranking does not invent any actual edge or completed reading. Negative infinite raw values remain available in the bounded tiny arithmetic oracle. Full-depth mathematical backups, before this final search floor, become exact when depth reaches total remaining glyphs because all surviving branches finish and no tail is queried. This is computationally exponential and is not attempted on the full source.

The linear nonterminal operator has nonnegative coefficients with row sums at most1−rho. Thus it does not amplify a uniform absolute tail error, and an unfloored depth-d error admits `(1-rho)^d * sup_error` on the finite acyclic state domain with equal terminal boundary. This is an OWN simple operator bound, not a pointwise monotonic improvement guarantee. At rho1/225 the one-step contraction factor is near1, so it gives no strong practical promise. Epsilon/floor/tail mismatch still matter; the production final floor is outside this operator claim.

## Mathematical checks completed before empirical registration

All36tiny observation pairs, two deterministic schedules, two-row Markov source, six units, rho1/4. Independent rational enumeration integrates every full remaining dictionary and plaintext string from both records' supplied contexts; it uses no action scheduler or local backup. Across312root/one-step-prefix states, full-depth Python backups agree with rational future probabilities to2.7755575615628914e−17. Compiled values at depths0/1/2/full-depth and unpruned search agree with the independent reference in tests.

Both directions of iid error have non-roundoff witnesses: with first record already completed and row0boundto0, the second record00has H=113/1536≈.073568, iid≈.048611, one-backup≈.057292. For second record01, H=35/1536≈.022786, iid≈.027778, one-backup≈.026042. These examples prove the guide is neither a future upper nor lower bound; they do not prove that every backup improves every state's estimate. Compact witnesses and checker code are pinned in the experiment's preparation record.

A first zero-support test mistakenly used01, which a single two-glyph dictionary unit can emit. Corrected the test to010, unsupported by one active source row with fixed1/2glyph emissions. This was a test-assumption failure before any empirical call; no engine probabilities changed.

## Realistic ablation before larger models

Use depth1 only on consumed-glyph layers0..15, chosen prospectively after the exposed first-loss diagnosis, then revert to original iid ranking. Compare depth0 with identical states/source/reader/config; depth0 must exactly reproduce old wide search. Both width4096, preliminary16384 and original finite state/work/cache budgets. Added child-table work is explicit, so equal beam limits are not equal CPU. At most16×16384lookahead calls and12,058,624legal child-action evaluations per case, below hard64Maction cap. Record actual guide table builds and wall/CPU rather than claiming free improvement.

No true source/key/hidden lengths enter fitting or prediction. Fit/prediction artifacts close before known-answer diagnostics. No new neural fit/GPU/paid use. An exposed-data signal requires at least10% aggregate edit reduction and improvement in at least2of4cases, with every reading present. That gate is exploratory and supplies no historical/fresh recovery qualification. Later matchedCPU/null/new-key tests remain required.
