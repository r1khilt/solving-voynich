# SOURCE-ACTION-TRAIN-001 — paired 96M reading-action fits

Status: prospective registration; no actual preparation or language fitting yet at registration. Question: does explicit access to past dictionary bindings improve actual unknown-key reading recovery across two matched seeds? Exploratory. [Design, review, metrics and limits](../research/source-action-training-design-2026-10-02.md) are binding.

## Admission and immutable inputs

Require SOURCE-ACTION-CACHE-001's 16 full-size GPU equivalence cells, own array/hash/resource closed check, and JOINT-KEY-TRAIN-001's completed source/input/full audit. Freeze every path in `scripts/run_source_action_train001.py:BASE_PATHS`, including original source roles, model/environment/cache code, target construction, training/auditor/tests, prior manifests and new method/registration. First exact published revision freezes preparation and its one audit. Then freeze BASE_PATHS plus inputs/prepare/input-audit receipts at a second exact published revision before the one four-fit campaign. No edits to frozen dependencies during an actual stage. Private bulk remains ignored and hash-bound.

## Allocation and budget

Four sequential large models, 96,039,982 parameters each, seeds 92203/92209, explicit dictionary query on/off; same paired initialization and episode stream. 20,000 updates each, batch four, 320,000 total training episodes; five checkpoint allocations per fit. Fresh 64 Pliny development keys, seed 92221; first 16 paired shuffled and IID glyph controls plus one ambiguity witness, control seed 92251. Complete specifications and gates are in the linked design. Twenty model files occupy about 7.68 GB before ledgers, traces and archives. Require at least 15 GiB free disk at campaign start.

Prior measured large max-shape update plus packing .813684 seconds implies about 18.08 hours for four fits, excluding source/target generation, all validation, audit and thermal variation. This is a planning extrapolation, not an actual training runtime. Longest 896-step cached generation measured about 2.76 seconds. Twenty checkpoint evaluations of 64 positives plus 33 controls have a naive longest-path estimate about 89 minutes, plus teacher forcing, CPU-double short checks and serialization. Actual conditional shapes may be shorter; source-trace generation was not included in the old packing-only figure. Budget up to eight wall hours and 24,000 absolute CPU seconds **per fit**, 33 wall hours for the sequential campaign (including child timeout margins); no retries/resume/extensions. A timeout or resource failure is an outcome, not permission to continue the same run.

Each preparation/input audit: 600 wall/500 absolute CPU seconds. Completion audit: 10,800 wall/9,000 absolute CPU seconds. Host peak RSS at most 4 GiB; sampled GPU driver allocation at most 8 GiB; private bulk at most 3 GiB per fit (12 GiB across four). CPU threads two, OMP/OpenBLAS one, Torch 2.14.0/Python 3.12.13/macOS 26.6.2. MPS required; CPU fallback disabled. Local cost zero USD; no API calls, downloads, reserved authors or manuscript panels. Memory sampling is not continuous measurement. Caffeinate prevents idle sleep; power loss or app termination can still fail a job. No automatic restart.

## Commands and one-call lifecycle

Use the project interpreter and disabled CPU fallback. GPU stages require the already authorized sandbox escalation for Metal access. Commit/push and verify the actual exact origin first, substituting that verified immutable revision for each placeholder.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.run_source_action_train001 prepare --freeze PREPARATION_REVISION
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.run_source_action_train001 audit-inputs --freeze PREPARATION_REVISION
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 caffeinate -i .venv/bin/python -u -m scripts.run_source_action_train001 campaign --freeze TRAINING_REVISION
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.audit_source_action_train001

All starts use exclusive files. The campaign attempts each predeclared arm once and retains terminal errors/timeouts; other allocated arms continue within the campaign bound. The sole completion audit follows actual terminal campaign closure; no result/audit reruns. Compact results/inputs/notebook are published; large checkpoints, source text, predictions and ledgers remain ignored. Audit PASS describes validation coverage, separately from the recovery gate.
