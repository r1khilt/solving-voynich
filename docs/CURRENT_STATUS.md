# Agent handoff and current status

Read `AGENTS.md`, `MEMORY.md`, the latest `NOTEBOOK.md` entries and `docs/research/PROTOCOL.md` first. The user authorizes active bounded local research, delegation, routine commits/pushes and useful GPU/RAM use. Explain results in plain English. Preserve the original charter, existing results, dataset splits and final manuscript holdout.

## Established work

EXP-0001 through EXP-0007 are complete and published. `docs/PROGRESS_EXPLAINED.md` explains every experiment and the terminology. No Voynich translation, signal/filler assignment or historical cipher has been established. Synthetic state/steering percentages concern known artificial generators. Manuscript final test remains unscored.

Repository: `/Users/rikhil/coding/solving-voynich`, branch `main`, remote `https://github.com/r1khilt/solving-voynich`. Latest completed-results checkpoint before this campaign is `4914052`; the explanatory guide is `c81ebef`. Preserve the user's uncommitted work if the checkout changes.

## Current task

User asked to run all three proposed larger ideas in parallel. CAMPAIGN-0001 passed final integration checks (295 tests plus23subtests, Ruff and whitespace); this checkpoint publishes the setup before immediate central execution. This tracked handoff remains frozen during the runs; use the live status path below for actual state:

- EXP-0008: four synthetic language models /16,000updates plus blind predictive partitions; hidden truth used only after frozen extraction. Own code `src/voynich/blind_recovery.py`; registration `docs/experiments/EXP-0008.md`; config `configs/exp0008_blind.json`.
- EXP-0009: frozen-model mapping across48sites, discovery/confirmation separation, horizons1/2/4/8 and negative controls. Own code `src/voynich/causal_mapping.py`; registration `docs/experiments/EXP-0009.md`.
- EXP-0010:21runs across context lengths256–2048, compact controls and input-only locus-marker removal. Exact matched token exposure perseed; fixed1200-update endpoints; final test excluded. Own code `src/voynich/long_context.py`; registration `docs/experiments/EXP-0010.md`; config `configs/exp0010.json`.

The first preparation attempt was interrupted by an agent usage limit; code survived. Only two short artificial-input GPU benchmarks had run. Three agents resumed their own code/review and completed the missing registrations. Do not mistake prepared code or hardware benchmarks for completed scientific runs.

## Execution and monitoring

The launcher uses `configs/campaign-0001.json` and requires a clean tree:

```sh
.venv/bin/python scripts/run_parallel_campaign.py
.venv/bin/python scripts/run_parallel_campaign.py --execute
```

The first command prints the plan; the second runs it. Inspect `outputs/CAMPAIGN-0001/status.json`, per-track `.log` files and `resources.jsonl` for live truth. A missing status means no launch has been recorded there; inspect processes before assuming. The launcher refuses overwriting a nonempty campaign directory. Root must commit/push all source and registrations before execution, then keep tracked files unchanged until every track ends.

All jobs run concurrently on one GPU with2CPUthreads each; MPS allocator high/low fractions0.23/0.18. Total conservative process-RSS guard48GiB. Internal track limits3h/2h/3h, supervisor grace60s, overall8h cap. These are ceilings, not elapsed-time forecasts. Do not change device memory policies globally or use broad process-kill commands. MPS generally needs the approved outside-sandbox tool path.

Outputs: `outputs/EXP-0008`, `outputs/EXP-0009`, `outputs/EXP-0010/campaign`; synthetic data `data/processed/exp0008-blind`; weights/activations/raw text stay ignored. EXP-0010's artificial benchmark is `outputs/EXP-0010/benchmark.json`, estimating44.17minutes training alone before evaluation and shared-GPU contention. Final reports must audit manifests, frozen artifacts, data/checkpoint hashes, matched exposures and all failures before publication. Archive compact reports under `results/`; write results narratives, update notebook/memory/README, commit/push and verify remote agreement.

## Important pending review/interpretation constraints

- EXP-0008 extracts predictive clusters, not a complete automaton or plaintext decoder. Neural weights transfer unchanged to evaluation keys, but extraction adapts to unlabeled visible text perkey. Copy/IID key labels denote fresh streams, not different encodings. No claim of zero-shot rule recovery.
- EXP-0009 must freeze discovery-selected sites before confirmation. All patches remain within the original prefix, with no future labels leaking into pair selection. Effective final-answer steering is insufficient; future consequences and matched controls matter.
- EXP-0010's full2048-prefix validation subset has only4pages/2leaves. Primary targets span24pages, often with shorter available prefixes. Different target samples must not be compared as if they were training progress.
- If a job fails integrity/numerical checks, preserve the failure. Source fixes require a new clean published revision and explicit rerun record; do not silently change code under other active jobs.
- Check actual completion files and processes after an interruption. The earlier conversation's statement that a handoff was already ready was not verified at that time; this file is the actual maintained handoff.
