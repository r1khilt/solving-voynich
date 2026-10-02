# Preserve the action law while retaining inference state

2026-10-02 PDT. New separate implementation; the original nine-input benchmark
freeze/reference action model is unchanged. Tiny CPUdouble qualification only,
not GPU or original-language inference throughput/recovery.

The training shape benchmark supports a substantial experiment, but the
reference rollout re-encodes the whole observation and recomputes every previous
decoder query at every action. An unmeasured evaluation phase must not be hidden
inside the12.54-hour shape-only training projection.

`source_action_cache.py` encodes observed ciphertext ONCE per path. Each decoder
layer retains its cross-attention projected keys/values and previous self-attention
keys/values. The next query uses the same previous action, current record/offset
and exact partial row-unit pair memory as the reference. It computes only that
query, then retains its self keys/values. Training architecture/weights/objective
are untouched. The cache exists locally for one sequential path, not across
weight changes/observations/reordered or branched queries; inference/no-gradient
scope required and unsupported attention architectures refused.

An implementation trap deserves an explicit check. With a single current query
and many retained past keys, PyTorch's non-square `is_causal=True` aligns the
triangular mask to the upper left, which would expose the FIRST key instead of
all past/current keys. Our stored keys already contain only allowed history,
so self-attention uses no further causal mask. Cross-attention uses the inverted
padding mask: SDPAbooleanTrue means allowed, unlike MHAkey-paddingTrue meaning
excluded. Dropout is explicitly0. These semantics were checked against the
[official PyTorch2.14 API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html).

Four tiny tests cover both binding-input conditions and four duplicate/mixed
length dictionaries, unequal source lengths/record-end schedules. Every legal
cached-prefix logit agrees with the separate full causal decoder within2e-12
CPUdouble. Eight teacher traces/several sampled continuations retain the same
states/actions/status and path logdensity within2e-12. Repeat scoring reuses
the same query; skipping a query, illegal branches, training-mode construction
and changed prenorm architecture are refused. A forced unsupported prefix is
reported as a dead end, not refilled. Model mode is restored even on failure.
This is numerical equivalence to the existing reference, not an independent
source/language evaluator or proof of posterior calibration.

No GPU cached inference or long-path timing has run. In particular, CPUdouble
agreement does not certify MPS kernels, full-size memory, accumulated float
differences or stochastic-action agreement near categorical boundaries. The
next fixed admission panel should compare complete prefix logits/logdensities
on short and maximum-length artificial traces, report actual cache timing/RSS/
driver bounds and include literal failure controls. Then choose a bounded
fresh-key actual-Latin reading/binding objective comparison. No expensive
training or manuscript claim follows automatically from either implementation.
