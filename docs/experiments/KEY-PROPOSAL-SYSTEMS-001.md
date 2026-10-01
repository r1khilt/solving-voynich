# KEY-PROPOSAL-SYSTEMS-001 — bounded whole-key model resource measurement

Prospective engineering measurement, registered 2026-09-30 before hardware
execution. Not a synthetic recovery qualification or language training campaign.
Question: can the actual conditional whole-key model perform optimizer updates
and generate complete keys at realistic input sizes within finite local caps?
Rationale/source review: joint-key-proposal-implementation-2026-09-30.md and its
linked primary papers and earlier uncertainty derivation.

## Fixed inputs and arms

Generate four artificial episodes with NumPy seed72121. Each has one iid-uniform
23-row dictionary from all42one/two-glyph units overABCDEF (duplicates permitted),
and four independent uniform224-source-letter records. Encode literally, then
canonicalize only observed glyphs; pad each record to448glyphs withPAD6, no
truncation. Retain all source indices, keys and canonical metadata in an ignored
checksum-bound compressed input file. No corpus, original cipher-panel case,
answer file, trained checkpoint, manuscript transcription or final holdout is
opened. These artificial supervised targets are intentionally uninformative
about linguistic generalization. No train/test performance conclusion.

Fresh processes, **sequential fixed order**: small-b1, small-b4, large-b1,
large-b4. Batch1is the first episode of batch4; both scales use the same four
episodes. Control5,423,146parameters, large94,981,674parameters; actual constructors
must match. Model seed72131, CPU sampling seed72139. Each matched batch pair starts
from identical weights, verified by raw name/shape/tensor digest. Torch2.14.0,
NumPy2.5.3, Python3.12.13 / uv.lock; record actual versions in arm metadata.

## Optimizer and proposal workload

Each arm executes **15 random-data optimizer updates**, not zero training or
language training. AdamW1e-4, decay.01, foreachFalse, full23-row mean cross entropy,
clip norm1; all loss/norm values finite, all gradients present/finite on first
and last steps. First3steps warmup; remaining12step times are arithmetic averaged.
Matched exposure across capacities at each batch size; batch1vs4exposure differs.
CPUthreads2/BLAS1, oneMPSprocess at a time. CPU fallback explicitly disabled by
PYTORCH_ENABLE_MPS_FALLBACK=0. MHA fastpath disabled consistently for references.

After updates, encode the first full episode once; generate8sampled complete
keys attemperature1plus1greedy key, with all23causal choices logged. No KV cache,
deduplication, exact fitting or decoder search. Initial/final random-trained
float32 checkpoints retained outsideGit with artifact+rawstate digests. These
checkpoints are resource fixtures, not language-trained decipherment models.

## Criteria and limits

PASS per arm requires all15updates, nine valid23-row keys, full trace, finite
native JSON outputs, exact parameter counts, matched initial weights, and:

- Initial/final first16glyphs of allfourrecords, complete23-row decoder logits:
  MPSfloat32 vs separate CPUfloat64 maximum error<=5e-5.
- Nine generated full-key joint log probabilities under **full448glyph inputs**
  versus separate CPUfloat64 teacher-forced replay: maximum error<=.002nats.
- Sampled Metal driver memory<=16GiB and process peakRSS<=8GiB. Sampling is not
  a continuously enforced physical-RAM or whole-machine memory limit.

Per child300wall/240CPU seconds after admission, outer timeout315seconds,
at most21minutes child scheduling envelope. No retry, extension or altered
workload if an arm fails. Continue and retain all four outcomes, including
missing/partial artifacts and terminal logs. Parent exit0does not imply armsPASS.
Resource traces after initialization/references/every optimizer/proposal choice;
checksummed trace stores actual driver/current tensors, RSS, wall/CPU, environment.
Child elapsed/CPU exclude source admission and imports; campaign wall and
all-childCPU include subprocess setup. Proposal timing includes synchronization
and callback trace overhead, not CPU reference computation.

One additional **CPU-only auditor**,450wall/400CPU seconds after admission,
twoCPUthreads,8GiBsampledhostguard. It checks the entire fixed inventory,
generates exact artificial inputs again, hashes all complete and partial
artifacts, replays success initial/final checkpoints/short logits/full-key log
probabilities, and verifies random work counts, timer arithmetic and guard traces.
Failure/missing outcomes stay failures; same-author alternate audit is not
independent-agent approval. Auditor-started exclusive marker prevents repeat
execution. Audit limit/failure is not repaired through a quiet rerun.

Total scheduled envelope about28.5minutes plus admission/import/parent overhead,
maximum60randomoptimizerupdates, zero paidAPI/cloud spend. Retain approximately
1.6GBof initial/final checkpoints if all arms finish; ignored local outputs,
compact metadata only inGit. Disk verified1.5TiBfree before launch. No unknown
language/key-solving, global optimum, fresh-panel PASS or mech-interp claim.

## Publication and command

Publish source/protocol/tests/notebook and verify exact remote ref before a
single invocation. PATHS binds model, runner, auditor, tests, method memo,
protocol, lock and imported resource/admission/archive helper sources. Results
have exclusive started/result paths; preserve all failures.

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=.:src \
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -u \
scripts/benchmark_key_proposal_systems001.py campaign --freeze <published-commit>

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=.:src \
.venv/bin/python -u scripts/audit_key_proposal_systems001.py
```

After complete outcomes and the one audit, publish actual resources and any
failures. Use observed costs to register a finite corpus/key training campaign
separately; do not silently convert this resource run into recovery evidence.
