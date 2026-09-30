# SUFFIX-READER-002-DIAG-A: posterior ambiguity and supplied-length intervention

2026-09-30. **Answer-informed, posthoc diagnosis on the exposed SUFFIX002 panel.**
Its failed gates remain failed. Freeze this plan and code before execution.
No fitting, new candidates, source changes, new key search, paid services or
new held-out data. All sixteen keys and all thirty-two records are required.

## Question and reviewed basis

The calibrated source still makes229edits with correct keys. A net56-letter
shortfall motivates checking length uncertainty, but does not by itself account
for these errors. Are the remaining readings sharply preferred, or does this
model distribute probability across many alternatives? What changes if the
true plaintext length is supplied as extra information?

The [prior inference review](../research/blind-channel-next-inference-review.md)
already identifies a true-length oracle as a way to separate segmentation
from dictionary error. The previous small sampled edit-risk experiment
[DEV004](BLIND-CHANNEL-DEV-004-results.md) changed no readings; do not rerun it
as if it were a new hypothesis. The current source and allocation differ.

[Eisner2001](https://www.cs.jhu.edu/~jason/papers/eisner.fsmnlp01.pdf), section3,
defines probability-weighted additive path expectations and gives length as an
example. We use that algebra in an acyclic exact decoder, with log-stable
normalized moment merging. No EM or Bayesian parameter integration is claimed.
[Dou et al.2015](https://aclanthology.org/P15-1081.pdf), sections2–4, uses posterior
sampling and source priors for word decipherment; our diagnostic instead sums
all paths for a fixed character source and a supplied deterministic dictionary.
Its word embeddings, vocabulary and translation results do not validate our model.
[Hauer/Kondrak2016](https://aclanthology.org/Q16-1006.pdf), sections3–5, applies
source-language/anagram scores to Voynich under restrictive assumptions.
A posterior here is conditional on our assumptions, not a probability that a
Voynich translation or historical mechanism is correct.

## Exact quantities and controlled intervention

For every frozen record and all four existing arms, compute:

- `Z = sum_y w(y)` for all plaintexts whose dictionary encodes the ciphertext.
- Mean additive log weight, mean length and second length moment.
- Posterior entropy `H=(log Z - E[log w])/log(2)` in bits.
- Exact number of supported plaintexts, MAP probability, and the true text's
  posterior surprisal and score gap from MAP.

A state is observed offset plus the existing sufficient suffix state. Merged
prefixes carry `(log mass, E[log weight], E[length], E[length²])`. Appending an
edge of weight`w` adds`log w`, adds1to length, and changes the second moment by
`2E[length]+1`. Merge by normalized mass, not unweighted averaging. The final
geometric stop factor is constant; it cancels from entropy but is included in
reported joint and marginal weights. No posterior truncation or beam.

The deterministic letter-to-unit mapping gives one path per plaintext string.
Thus path entropy equals plaintext entropy here. It would not automatically
equal text entropy for a transducer with multiple latent paths per plaintext.
The posterior and Bayes-optimal exact-record probability are properties of
this model, not an information-theoretic bound for natural Latin. Summed MAP
probabilities are a model-implied expected exact-record count for these fixed
observations; comparing with observed success is descriptive, not a calibrated
population test. Different keys use different nearby windows of one author.

Then, for **calibrated12 only**, add source-step count to the state and restrict
terminal count to the actual224letters. All source probabilities, dictionary,
text and stop law remain unchanged. Prune only count states mathematically
unable to finish at that length given the minimum/maximum unit lengths.
Report all32counterfactual readings and edits, exact records, posterior entropy,
and original posterior mass on the true length (`Z_restricted/Z`). The restricted
joint mass is not itself `P(ciphertext|length)`; that would additionally divide
by the original length prior. The resulting conditional plaintext posterior
normalizes correctly using the restricted mass.

True length is an explicit oracle, already known from construction. Improvement
would show the value of that extra information for this model; it would not
make the information available for Voynich, prove the best possible ordinary
reader is limited, or pass SUFFIX002's gates. No positive outcome is required.

## Validation and execution limits

Preserve all old source/protocol/evaluation hashes. Ten tiny exact rational
full-string enumerations cover zero/two-history sources, duplicate units,
merged histories, fixed length and empty observations; also test cap failure,
impossible support, invalid settings, subset scores and unchanged parameters.
A separate raw-count backward recurrence replays all160diagnostics, checking
entropy, both length moments, total/MAP weights and exact integer path counts.
Tolerance1e-6for posterior moments/scores,1e-5for variance cancellation; original
and returned-path score equivalence1e-7. Compare all128unrestricted readings to
the existing frozen readings and all128edit totals to evaluation. Independently
check each new length-constrained reading's re-encoding, length and edit count.
References are root-authored, not independent researcher replication.

Single execution after source/plan/results checkpoint is pushed and verified:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python scripts/run_suffix_reader002_diag.py --freeze <checkpoint>`.
One thread,1200CPU/1800wallsecond hard caps,500,000states/record and8GiBplanning.
Stop and preserve failure without pruning, restarting or extending limits.
No paiduse/downloads. Track compact quantitative results; keep counterfactual
plaintext in ignored hashed archives. Run full regression, record actualcosts,
update notebook/memory, publish and verify completed positive or negative outcome.
