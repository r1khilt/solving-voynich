"""Probability, causal-prefix, recurrence, and blind-fit reference tests."""

import itertools
import json

import numpy as np
import pytest
import torch

from voynich.episodic_models import (
    EdgeHMM, ModelSpec, SignedDeltaLayer, build_model, fit_edge_hmm, raw_log_probs,
)
from voynich.episodic_data import canonical_raw_probs, canonicalize


@pytest.fixture(scope="module", autouse=True)
def single_thread_reference():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model(kind, canonical=False, device="cpu"):
    torch.manual_seed(923)
    return build_model(ModelSpec(kind=kind, canonical=canonical, width=16, layers=2,
                                 heads=2, ff_width=24, context=16)).to(device).eval()


@pytest.mark.parametrize("kind", ["transformer", "gru", "signed_delta"])
@pytest.mark.parametrize("canonical", [False, True])
def test_shapes_future_masking_and_finite_gradients(kind, canonical):
    model = make_model(kind, canonical)
    tokens = torch.tensor([[0, 1, 1, 2, 3, 0], [3, 2, 0, 2, 1, 3]])
    other = tokens.clone()
    other[:, 3:] = (other[:, 3:] + 1) % 4
    logits = model(tokens)
    assert logits.shape == (2, 6, 4 + int(canonical))
    torch.testing.assert_close(logits[:, :3], model(other)[:, :3], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(logits[:, :3], model(tokens[:, :3]), atol=2e-6, rtol=2e-5)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                             tokens[:, 1:].reshape(-1))
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    assert model.parameter_count == sum(parameter.numel() for parameter in model.parameters())


@pytest.mark.parametrize("kind", ["gru", "signed_delta"])
def test_step_encode_forward_identity_and_restoration(kind):
    model = make_model(kind)
    tokens = torch.tensor([[0, 1, 0, 2, 1], [2, 3, 1, 3, 0]])
    state = model.initial_state(2, tokens.device)
    outputs = []
    for symbol in tokens.unbind(1):
        logits, state = model.step(symbol, state)
        outputs.append(logits)
    torch.testing.assert_close(torch.stack(outputs, 1), model(tokens), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(state, model.encode(tokens), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(model.readout(state), outputs[-1])
    torch.testing.assert_close(model.encode(tokens[:, :0]), model.initial_state(2))
    saved = model.encode(tokens[:, :3]).clone()
    first, first_state = model.step(tokens[:, 3], saved)
    model.step((tokens[:, 3] + 1) % 4, saved)
    restored, restored_state = model.step(tokens[:, 3], saved.clone())
    torch.testing.assert_close(first, restored, atol=0, rtol=0)
    torch.testing.assert_close(first_state, restored_state, atol=0, rtol=0)


def test_signed_factors_match_explicit_matrix_product_and_norm_bound():
    torch.manual_seed(51)
    layer = SignedDeltaLayer(7).double()
    inputs = torch.randn(4, 7, dtype=torch.float64)
    state = torch.randn(4, 7, dtype=torch.float64)
    factors = layer.factors(inputs)
    torch.testing.assert_close(factors["keys"].square().sum(-1), torch.ones(4, 2, dtype=torch.float64))
    assert ((factors["beta"] >= 0) & (factors["beta"] <= 2)).all()
    assert layer.keys[0].weight is not layer.keys[1].weight
    assert not torch.allclose(factors["keys"][:, 0], factors["keys"][:, 1])
    moved = state
    for index in range(2):
        key = factors["keys"][:, index]
        matrix = torch.eye(7, dtype=torch.float64)[None] - (
            factors["beta"][:, index, None, None] * key[:, :, None] * key[:, None, :])
        moved = torch.einsum("bij,bj->bi", matrix, moved)
    expected = factors["retain"] * moved + (1 - factors["retain"]) * factors["write"]
    actual = layer(inputs, state)
    torch.testing.assert_close(expected, actual, atol=1e-12, rtol=1e-12)
    bound = torch.maximum(state.norm(dim=-1), factors["write"].norm(dim=-1))
    assert (actual.norm(dim=-1) <= bound + 1e-12).all()


def test_canonical_raw_probabilities_mask_and_redistribute_new():
    logits = torch.tensor([[2.0, 20.0, 30.0, 40.0, 3.0], [1., 2., 3., 4., 100.]],
                          requires_grad=True)
    counts = torch.tensor([1, 4])
    inverse = torch.tensor([[2, -1, -1, -1], [3, 1, 0, 2]])
    logp = raw_log_probs(logits, counts, inverse)
    torch.testing.assert_close(logp.exp().sum(-1), torch.ones(2))
    two_event = torch.tensor([2., 3.]).softmax(0)
    torch.testing.assert_close(logp[0].exp(), torch.tensor([two_event[1] / 3, two_event[1] / 3,
                                                          two_event[0], two_event[1] / 3]))
    expected = logits[1, :4].softmax(0)[torch.tensor([2, 1, 3, 0])]
    torch.testing.assert_close(logp[1].exp(), expected)
    (-logp[0, 2] - logp[1, 0]).backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[0, 1:4]) == 0
    assert logits.grad[1, 4] == 0


def test_canonical_zero_seen_is_uniform_and_renaming_is_exact():
    logits = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    counts = torch.tensor([[0, 1, 2], [2, 3, 4]])
    inverse = torch.tensor([[[-1, -1, -1, -1], [2, -1, -1, -1], [2, 0, -1, -1]],
                            [[1, 3, -1, -1], [1, 3, 0, -1], [1, 3, 0, 2]]])
    permutation = torch.tensor([2, 0, 3, 1])
    renamed = torch.where(inverse >= 0, permutation[inverse.clamp_min(0)], inverse)
    raw = raw_log_probs(logits, counts, inverse)
    other = raw_log_probs(logits, counts, renamed)
    torch.testing.assert_close(raw, other[..., permutation], atol=0, rtol=0)
    torch.testing.assert_close(raw[0, 0].exp(), torch.full((4,), 0.25, dtype=torch.float64))
    torch.testing.assert_close(raw.exp().sum(-1), torch.ones(2, 3, dtype=torch.float64))
    assert torch.autograd.gradcheck(lambda x: raw_log_probs(x, counts, inverse), (logits,))


def test_data_numpy_adapter_and_neural_canonical_renaming_agree():
    tokens = np.array([[3, 3, 1, 0, 3, 2], [2, 0, 0, 2, 1, 3]])
    ranks, counts, inverse = canonicalize(tokens)
    model = make_model("gru", canonical=True).double()
    logits = model(torch.from_numpy(ranks))
    actual = raw_log_probs(logits, torch.from_numpy(counts), torch.from_numpy(inverse))
    expected = canonical_raw_probs(logits.detach().numpy(), counts, inverse)
    np.testing.assert_allclose(actual.exp().detach().numpy(), expected, atol=1e-14, rtol=1e-13)
    for permutation in itertools.permutations(range(4)):
        permutation = np.array(permutation)
        renamed_ranks, renamed_counts, renamed_inverse = canonicalize(permutation[tokens])
        renamed_logits = model(torch.from_numpy(renamed_ranks))
        renamed = raw_log_probs(renamed_logits, torch.from_numpy(renamed_counts),
                                torch.from_numpy(renamed_inverse))
        torch.testing.assert_close(actual, renamed[..., torch.from_numpy(permutation)], atol=0, rtol=0)


@pytest.mark.parametrize("counts,inverse", [
    ([5], [[0, 1, 2, 3]]), ([-1], [[-1, -1, -1, -1]]),
    ([2], [[1, 1, -1, -1]]), ([1], [[0, 2, -1, -1]]),
    ([2], [[0, -1, -1, -1]]), ([2], [[0, 4, -1, -1]]),
])
def test_malformed_canonical_maps_fail(counts, inverse):
    with pytest.raises(ValueError):
        raw_log_probs(torch.zeros(1, 5), torch.tensor(counts), torch.tensor(inverse))


def make_hmm():
    return EdgeHMM(np.array([[[3., 1.], [1., 2.]], [[1., 2.], [4., 1.]]]), [2., 1.])


def test_edge_hmm_manual_path_sum_normalization_and_filter():
    model = make_hmm()
    np.testing.assert_allclose(model.edge.sum(axis=(0, 2)), np.ones(2), atol=1e-15)
    tokens = [1, 0, 1]
    path_sum = 0.0
    end_masses = np.zeros(2)
    for path in itertools.product(range(2), repeat=4):
        probability = model.prior[path[0]]
        for index, symbol in enumerate(tokens):
            probability *= model.edge[symbol, path[index], path[index + 1]]
        path_sum += probability
        end_masses[path[-1]] += probability
    assert model.log_likelihood(tokens) == pytest.approx(np.log(path_sum), abs=1e-14)
    np.testing.assert_allclose(model.filter(tokens), end_masses / path_sum)
    assert model.log_likelihood([]) == 0
    np.testing.assert_array_equal(model.filter([]), model.prior)
    belief = model.filter(tokens[:2])
    np.testing.assert_allclose(model.update(belief, tokens[2]), model.filter(tokens))
    assert model.score_continuation(belief, tokens[2:]) == pytest.approx(
        model.log_likelihood(tokens) - model.log_likelihood(tokens[:2]))
    for horizon in range(5):
        joint = model.joint_probs(belief, horizon)
        assert joint.shape == (2,) * horizon
        assert joint.sum() == pytest.approx(1., abs=1e-13)
        for continuation in itertools.product(range(2), repeat=horizon):
            assert joint[continuation] == pytest.approx(np.exp(model.score_continuation(belief, continuation)))
    np.testing.assert_allclose(model.joint_probs(belief, 1), model.next_probs(belief))


def test_edge_hmm_state_permutation_does_not_change_observations():
    model = make_hmm()
    permutation = [1, 0]
    renamed = EdgeHMM(model.edge[:, permutation][:, :, permutation], model.prior[permutation])
    tokens = [0, 1, 1, 0, 1]
    assert model.log_likelihood(tokens) == pytest.approx(renamed.log_likelihood(tokens), abs=1e-14)
    np.testing.assert_allclose(model.joint_probs(model.filter(tokens), 3),
                               renamed.joint_probs(renamed.filter(tokens), 3), atol=1e-14)


def test_visible_em_k1_matches_smoothed_counts_and_fit_is_reproducible():
    train = np.array([[0, 0, 1, 0, 2, 0, 1, 0], [1, 0, 0, 0, 3, 0, 0, 0]])
    validation = np.array([[0, 1, 0], [0, 0, 1]])
    first, report = fit_edge_hmm(train, validation, states=(1, 2), restarts=2, iterations=5, seed=8)
    second, other = fit_edge_hmm(train, validation, states=(1, 2), restarts=2, iterations=5, seed=8)
    assert report == other
    np.testing.assert_array_equal(first.edge, second.edge)
    assert report["selected_states"] == 1
    expected = np.bincount(train.reshape(-1), minlength=4) + 0.01
    expected /= expected.sum()
    np.testing.assert_allclose(first.next_probs(), expected, atol=1e-14)
    assert len(report["candidates"]) == 4
    assert report["refit_on_validation"] is False
    assert report["condition_validation_on_paired_train_prefix"] is True
    json.dumps(report, allow_nan=False)
    # Validation affects candidate selection only, never the fitted candidate's EM trace.
    _, changed = fit_edge_hmm(train, (validation + 1) % 4, states=(1, 2), restarts=2, iterations=5, seed=8)
    for original, modified in zip(report["candidates"], changed["candidates"]):
        assert original["train_log_likelihood_trace"] == modified["train_log_likelihood_trace"]


def test_conditional_suffix_validation_uses_training_filter():
    train = np.array([[0, 0, 1, 1, 0, 0]])
    suffix = np.array([[1, 1, 0]])
    model, report = fit_edge_hmm(train, suffix, alphabet=2, states=(2,), restarts=1, iterations=2, seed=55)
    expected = model.log_likelihood(suffix[0], initial=model.filter(train[0]))
    assert report["candidates"][0]["validation_log_likelihood"] == pytest.approx(expected)


@pytest.mark.parametrize("change", [
    {"kind": "mamba"}, {"alphabet": 1}, {"canonical": "false"}, {"width": True},
    {"layers": 0}, {"heads": 0}, {"ff_width": -1}, {"context": 0},
    {"width": 15}, {"dropout": float("nan")}, {"dropout": 1},
])
def test_model_spec_rejects_malformed_configuration(change):
    with pytest.raises(ValueError):
        ModelSpec(**change)


@pytest.mark.parametrize("kind", ["transformer", "gru", "signed_delta"])
@pytest.mark.parametrize("tokens", [torch.tensor([[4]]), torch.tensor([[-1]]),
                                    torch.ones(1, 2), torch.empty(1, 0, dtype=torch.long)])
def test_models_reject_invalid_input(kind, tokens):
    with pytest.raises(ValueError):
        make_model(kind)(tokens)


def test_edge_hmm_rejects_invalid_parameters_and_beliefs():
    with pytest.raises(ValueError):
        EdgeHMM(np.zeros((2, 1, 1)), [1.])
    with pytest.raises(ValueError):
        EdgeHMM(np.ones((2, 1, 1)), [0.])
    model = make_hmm()
    for belief in ([0.1, 0.1], [-0.1, 1.1], [float("nan"), 0.], [1.]):
        with pytest.raises(ValueError):
            model.next_probs(belief)
    with pytest.raises(ValueError):
        model.log_likelihood([0, 2])
    with pytest.raises(ValueError):
        model.joint_probs(horizon=21)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable in current environment")
@pytest.mark.parametrize("kind", ["transformer", "gru", "signed_delta"])
def test_mps_reference_matches_cpu(kind):
    cpu = make_model(kind)
    mps = make_model(kind, device="mps")
    mps.load_state_dict(cpu.state_dict())
    tokens = torch.tensor([[0, 1, 0, 2, 3]])
    with torch.no_grad():
        torch.testing.assert_close(cpu(tokens), mps(tokens.to("mps")).cpu(), atol=3e-5, rtol=3e-4)
        if kind != "transformer":
            state = mps.initial_state(1)
            outputs = []
            for symbol in tokens.to("mps").unbind(1):
                logits, state = mps.step(symbol, state)
                outputs.append(logits)
            torch.testing.assert_close(mps(tokens.to("mps")), torch.stack(outputs, 1),
                                       atol=3e-5, rtol=3e-4)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable in current environment")
def test_mps_canonical_adapter_forward_and_gradient_match_cpu():
    logits = torch.tensor([[.1, .2, .3, .4, .5], [.5, .4, .3, .2, .1]], requires_grad=True)
    counts = torch.tensor([2, 4])
    inverse = torch.tensor([[2, 0, -1, -1], [1, 3, 0, 2]])
    expected = raw_log_probs(logits, counts, inverse)
    (-expected[:, 0].sum()).backward()
    mps_logits = logits.detach().to("mps").requires_grad_(True)
    actual = raw_log_probs(mps_logits, counts.to("mps"), inverse.to("mps"))
    (-actual[:, 0].sum()).backward()
    torch.testing.assert_close(expected, actual.cpu(), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(logits.grad, mps_logits.grad.cpu(), atol=2e-6, rtol=2e-5)
