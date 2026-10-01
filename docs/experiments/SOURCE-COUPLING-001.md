# SOURCE-COUPLING-001: assisted key-interaction diagnosis

2026-10-01, exploratory registration BEFORE one invocation. Question: do
SOURCE-REVISE-001 endpoints miss single-row gains, or does correcting known
mistakes expose joint effects? Method/prior-work review and exact derivation:
`docs/research/source-key-coupling-2026-10-01.md`.

## Inputs and status

Exactly16exposed selected keys from the fully audited SOURCE-REVISE-001,
using the same four SOURCE-GUIDE-001 fixtures (64,64,224,224 letters per
record;2records/key; seeds75501..4), particle seeds75511/75513 andconstant/iid
guidance. No new fixture, old cipher panel, reserved author, manuscript test,
new download or neural checkpoint. Parent result/audit/16compact cells and
prediction/fixture bytes/hash bindings retained. Four repeated keys are not
16independent populations. GOLD IS USED TO CONSTRUCT THESE INTERVENTIONS:
this is an assisted diagnostic, never unassisted recovery or qualification.

Original Latin order12/23-row source/counts/selected tau remain unchanged.
Counts19,566,187bytes SHA256
9769415ac460c56778a461af7646635abf5906fa726484ba84d98d4f401815f6.
Exact native build/source admission unchanged. Clone original root COUNT
arrays into a separate CompactSuffixSource(order0)/DenseSuffixAdapter;
same smoothing0.5, root probability row exactly identical, zero transitions,
new readonly arrays≤4096bytes. No root renormalization or source fitting.
Native ABI/compiler/source/library hashes bound by the existing build manifest.

## Fixed candidate enumeration and objective

For every selected base, score all23×41=943single-row alternatives. Define M
as incorrectly mapped gold-used rows. Score ALL gold-target anddecoy-target
subsets of M of sizes1,2,3 in increasing size/lexical row order. Each decoy
unit has goldtarget's length, excludes incumbent/gold, and is drawn once from
the lexically ordered42unit pool with Python random.Random seed
76111+101*case+(population_seed-75511). Same seed across paired guidance arms,
but exclusions/baselines may differ. Keep all other rows fixed. Score gold-M
endpoint, decoy-M endpoint and generating FULL key. Memoize unique full keys;
retain every intervention-to-score reference, including repeats.

For EACH candidate use both contextual andIID exact logsum native marginals
of ALL literal source paths, rho1/225 andconstant−23log42full-key prior.
Record nodes/edges, eachrecordscore, total andsupport. Originalbase andgold
full-key scores must match frozen prior diagnostics within1e-7. Supportmask
context/IID must agree exactly because both source models have strictly
positive probabilities. Graphcaps2Mnodes/8Medges PERrecord raise realfailure;
no beam, clipping, fake zero or candidate pruning. No textual decoding or
policy/model training in this diagnostic.

At most1+943+2*(C(23,1)+C(23,2)+C(23,3))+3=5041unique keys/cell,
80656summed across16, at most322624native record scores (2models×2records),
before caching reductions. This is not exhaustive23-row search. One fixed
draw of decoys does not estimate a full random-control distribution.

## Recorded observations and prespecified diagnostic signals

Every model/family records bestsingle gain, whether all943alternatives are
within0.1nat of base, all1/2/3face counts/support, bestgains/indices,
improvingfaces(>1nat), anchored finite-difference range/median on supported
faces, joint-only gains andfull-M endpoint gains. Derivatives UNDEFINED if
anysubface unsupported. A joint-only improving pair/triple hasgain>1nat
andALLpropernonempty subfaces gain≤0.1nat orzero support. Structural support
bridges requireALLsuchsubfaces zero. Interpret onlywithin these chosen binary
faces, neverasglobalbarriers orminimumnecessaryblocksize.

Two separate descriptive signals, no statistical significance:

- Missed-single-improvements SUPPORTED if≥8of16contextual cells have best
  all-single replacement gain>1nat; otherwiseNOT_SUPPORTED.
- Gold-assisted-joint-advantage SUPPORTED if≥8cells have bestgold pair/triple
  gain>max(0,bestall-single gain)+10nats ANDstrictlymoregold joint-only
  improvingfaces thandecoy joint-only faces; otherwiseNOT_SUPPORTED.

Also report context-versus-IID differences foridentical candidates. Goldhelp
anddevelopmentselection prevent anyrecovery/semantic/causal-neuron claim.
No fitted solver gets gold andnoaccuracythreshold is relabeled PASS.

## Bounds and verification

ONErun andONEcompletionaudit, each1800wall/1600absoluteCPU seconds,
2GiB sampledpeakhostRSS; totalbulk run output≤128MiB, ignored allscore/face
archives. BLAS/environmentthreads1, oneCPUworker,zeroGPU/training/newmodel
fit/paidAPI/download/holdout; actualfixedstatisticalsourceprediction occurs.
Expected5–15min/stage usingpreviousnative rates/plannedcandidateupperbound;
differentlattices maycostmore, hard30min/stage. No retry/resume/extension.
Failures retained withcompletedcell count; noafter-results cap adjustment.

Oneaudit replays EVERYkey underbothnative source models, independently
reconstructs literalcandidate inventory andvalidatesallparent/archive/source
hashes. Summary/probe reconstruction reusesouranalysiscode andis labeled
as such. Alternate full-record PythonDP checks baseline/fullgold/endpoints,
bestall-single andbestjointkeys underbothmodels:≤10uniquekeys/cell,
≤40scalar record/model scores/cell,≤640overall, tolerance1e-7 includingzero
support. Sameauthor/nativebackend; noindependentagent/scientificreplication.
Tests validate multilinearcoefficients, hard-supportpair/triplewitnesses,
decoyconstraints, unchangedcontextarrays/IIDroot, invalidscores/support,
andactual16cell runner/auditor transport onmock source.

Publish source/tests/protocol/notebook/memory; verifyexactremote before run.
Originalfour-fitMPS campaign anditsall-terminalcompletionaudit unchanged.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=.:src
    .venv/bin/python -u scripts/run_source_coupling001.py --freeze <registration>

Sameenvironment, onceafterterminalsuccessfulrun:

    .venv/bin/python -u scripts/audit_source_coupling001.py
