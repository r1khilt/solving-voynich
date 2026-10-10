# Escape routes also need a usable route back

Own derivation and artificial preparation after the relaxed-channel source admission.
The current31ba50c admission/dispatcher stays frozen. No new collapsed sampler, fitted
policy, empirical return-rate experiment, fresh recovery allocation or manuscript reading
is implemented by this note.

## What is now observed

WINDOW-READING-RECOVERY-001 completed with zero exact messages/used keys and a full audit.
The separate relaxed-state finite qualification proves its stated finite laws; the new
CONSTRAINT-READING-ADMIT-001 producer/full auditor both pass all5152locals/1932swaps/
1288cold references. Original-source cost is about.024CPU seconds per local attempt.
Short four-sweep archived-frame trajectories demonstrate coverage/conformance, not equilibrium,
mode traversal, recovered keys or efficient warm-to-hard transport. Do not tune that frozen
admission ladder from its outcomes or extend it into recovery.

## Primary methods and their limits

[Atchade, Roberts and Rosenthal, author manuscript revised2010, §§1–2](https://probability.ca/jeff/ftpdir/mcmcmc.pdf)
connect replica spacing to useful transport. Their0.234 optimal swap rate concerns a stated
high-dimensional product/power-target asymptotic setting and squared jumping distance.
It is not a universal target for our discrete constraint ladder with an exact-zero endpoint.
Their illustrative example also distinguishes high acceptance from useful transport.

[Liu1994, §4 operator comparison](https://www.cs.columbia.edu/~blei/fogm/2023F/readings/Liu1994.pdf)
provides a general mathematical precedent for integrating variables out rather than sampling
them. Its Gibbs comparison does not prove our proposed MH rewrite has a better spectral gap.
[van Dyk and Jiao2014v2, introduction, §§2.3/3.2](https://arxiv.org/html/1309.3217v2)
shows why reduced conditioning plus stale conditional MH updates can change the intended
stationary distribution. A fully marginalized chain followed by a fresh exact conditional
reconstruction is the proposed route here; do not simply substitute a marginal score into
our existing key-dependent proposal. The2008author-PDF fetch timed out; the2014methods
paper was directly accessible and read instead. No source simulation transferred to Voynich.

Earlier project applications: the [hard-channel counting derivation](collapsed-source-key-inference-2026-10-01.md)
already integrates unused dictionary rows and counts compatible keys; particle/regrowth/
tempered experiments still failed competence. The [support counterexample](window-support-and-constraint-relaxation-2026-10-10.md)
shows a separate missing-edge obstruction. This extends the soft-state sum over visited
bindings, rather than reviving pure row-label search or importing a known plaintext oracle.
No new prior Voynich success or historical noisy-channel premise is claimed.

## Exact stationary return identity for the CURRENT explicit-key ladder

Let Z_e=sum_z B(z)e^D(z) over the finite extended states; Z_0=sum_{D=0}B(z)>0.
With hard state x and warm y, the current swap accepts precisely when D(y)=0,
and then its ratio is1. At stationarity the expected hard/warm exchange acceptance is

    Pr_e[D=0] = Z_0/Z_e.

This identity uses the warm marginal distribution at stationarity. It is not an estimate
from the admission's four sweeps, all replicas starting hard, nor does it imply the
accepted warm state came from another mode. Large numbers of accepted identical or local
swaps can be uninformative about remote transport.

With theta=log(e)>-infinity, ordinary finite-sum differentiation gives

    d log Z/d theta = E_theta[D],
    d² log Z/d theta² = Var_theta(D).

Small-step directed KL is one half Var_theta(D)(delta theta)² plus higher-order terms.
This suggests measuring violation histograms/overlap to design a ladder on separate
calibration material. It gives no fixed universal spacing; theta=-infinity is a singular
endpoint and needs the exact mass ratio above. No thermodynamic integral is estimated here.

## An avoidable source of mismatch: uncertain visited bindings

Let a shape s consist only of source texts x and observed one/two-glyph span widths w.
For each visited row r and possible unit u, accumulate its contribution across ALL records:

    d_r(u;s) = sum_{occurrences of r} edit(u, observed span),
    G_r(e;s) = sum_{u in pool} e^d_r(u;s).

The source likelihood depends on x, not the dictionary once x is fixed. Independent uniform
visited-unit priors therefore give an EXACT collapsed target

    Tbar_e(s) = Q(x) U^-m product_r G_r(e;s).

Proof: D=sum_r d_r(k_r;s), so the sum over all U^m visited assignments factors by row.
Unvisited rows remain integrated out. Rows are named/source labels are retained; do not
multiply another row-permutation count or collapse the records separately. At e0 every
G_r is either0 or1: all occurrences must have one identical unit, which is then uniquely
forced. Thus the hard collapsed target is exactly the original hard reading target, with
one key reconstruction per compatible shape. At e1 each G_r=U, giving Tbar_1=Q(x);
the visited prior cancels under the sum, rather than becoming an extra adjustable score.

For positive e a fresh exact key draw is independently, per visited row,

    Pr(k_r=u | s,e) = e^d_r(u;s)/G_r(e;s).

A fixed compatible shape can occupy many inconsistent key assignments in the explicit-key
warm representation. Collapse integrates these possibilities exactly instead of waiting
for every randomly sampled row to match simultaneously. It does NOT eliminate incompatible
source labels/partitions or guarantee mixing; that structural problem remains.

Example conditional sector: glyph2/unit6, fixed singleton spans containing0, distinct source
rows occurring once. Unit costs are0,1,1,1,1,2; G(e)=1+4e+e². With m rows, probability of
a consistent key is G(e)^(-m). At m23/e1/8 this is.00007020847074665044; at e1 it is
1.2662551980515063e-18. These are analytical fixed-shape sector examples, NOT the full
Latin/R23/glyph6 posterior or empirical manuscript rates. Actual repetitions change costs.

Collapsed replica swaps need their OWN ratio. Source and visited priors cancel between
replicas, but row polynomials remain:

    rho = [product G(e_i;y) product G(e_j;x)] /
          [product G(e_i;x) product G(e_j;y)].

At hard/warm, an incompatible incoming shape is rejected. For compatible x,y the ratio is
product G(e;x)/product G(e;y), generally NOT1. Reusing the explicit-key swap acceptance
would be wrong. Collapsed relabel/boundary proposals must be selected on shapes alone;
the old random fresh-binding factors are no longer that proposal's density. State-dependent
neural selection additionally needs positive reverse support and the actual selector ratio.

## Preparation checks and the next decisive questions

New test_constraint_marginal_theory.py directly sums ALL4308key/shape states over12small
constant/contextualR3/R4panels at e0,1/8,1/2,1 and compares every shape with the per-row
product. Independently computed source integer factors and finite edit cases are used;
state enumeration and the current swap law are shared, not independent expert review.
The tests also compare exact stationary explicit-key swap mass to Z0 and enumerate216keys
in the three-distinct-row fixed singleton sector.13testsPASS.73s after a preparation-only
source-fixture signature error (initial6FAIL/1PASS.50s) was fixed. This is algebra/fixture
validation, not an additional registered qualification or stochastic recovery campaign.

Before investing in a bigger model: implement/qualify a SHAPE-only proposal and collapsed
exchange, independently validate source cost, then allocate a separately frozen fresh-key
comparison against explicit relaxation and a hard baseline. Keep exact text/used-key
competence and nonsemantic controls, seal outputs before Gold, match/report real work.
Measure whether genuinely changed hard shapes arrive from warm exploration; accepted swaps,
source scores and correct software alone are insufficient. Fix ladder/calibration on
separate material, not those future scoring keys.

A subsequent repair policy/world-action model could operate on these shapes and propose
coordinated row/partition revisions. Error-state training and causal recipient rescue
would then test learned inference rather than supplied-key readout. That is still a
separate competence-gated model program, not an implemented diffusion model or mech-interp
finding. A larger network cannot repair absent moves, and exact marginalization cannot
by itself teach it language or establish the historical cipher family.
