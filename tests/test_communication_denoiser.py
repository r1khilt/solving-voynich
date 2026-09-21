from dataclasses import asdict
import io

import pytest
import torch

from voynich.communication.denoiser import (
    MASK, PAD, ConditionalDenoiser, DenoiserConfig, masked_loss, sample,
)


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


def make_model(**changes):
    torch.manual_seed(31)
    config = {
        "vocab_size": 9, "condition_vocab_size": 12,
        "max_latent_length": 16, "max_condition_length": 12,
        "width": 16, "layers": 2, "heads": 2,
    }
    config.update(changes)
    return ConditionalDenoiser(DenoiserConfig(**config))


def rng(seed=123):
    return torch.Generator(device="cpu").manual_seed(seed)


def test_forward_shapes_condition_and_noise_influence():
    model = make_model().eval()
    noisy = torch.tensor([[MASK, 3, MASK], [4, MASK, 5]])
    condition = torch.tensor([[2, 3, 4], [5, 6, 7]])
    noise = torch.tensor([0.7, 0.3])
    logits, hidden = model(noisy, condition, noise, return_hidden=True)
    assert logits.shape == (2, 3, 9)
    assert hidden.shape == (2, 3, 16)
    assert torch.isfinite(logits).all()
    changed = model(noisy, condition.flip(0), noise)
    assert not torch.allclose(logits, changed)
    assert not torch.allclose(logits, model(noisy, condition, 1 - noise))
    assert not torch.allclose(logits, model(noisy, condition, noise, latent_types=torch.ones_like(noisy)))


def test_masked_loss_never_passes_the_clean_masked_target_to_model():
    model = make_model()
    calls = []

    def record_inputs(_module, args, kwargs):
        calls.append((args[0].detach().clone(), args[1].detach().clone()))

    outputs = []
    input_hook = model.register_forward_pre_hook(record_inputs, with_kwargs=True)
    output_hook = model.register_forward_hook(lambda _m, _a, output: outputs.append(output.detach().clone()))
    condition = torch.tensor([[2, 3], [4, 5]])
    try:
        # Each row has exactly one target; corruption must hide it completely.
        masked_loss(model, torch.tensor([[2], [3]]), condition, generator=rng())
        masked_loss(model, torch.tensor([[7], [8]]), condition, generator=rng())
    finally:
        input_hook.remove()
        output_hook.remove()
    assert len(calls) == 2
    for noisy, observed in calls:
        assert torch.equal(noisy, torch.full((2, 1), MASK))
        assert torch.equal(observed, condition)
    # Changing only hidden truth cannot change model predictions.
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


def test_loss_has_finite_gradients_and_ignores_padding():
    model = make_model()
    clean = torch.tensor([[2, 3, 4, PAD], [PAD, PAD, PAD, PAD], [5, 6, 7, 8]])
    condition = torch.tensor([[3, 4, PAD], [PAD, PAD, PAD], [7, 8, 9]])
    saved_clean, saved_condition = clean.clone(), condition.clone()
    loss, metrics = masked_loss(model, clean, condition, generator=rng())
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert metrics["active_tokens"] == 7
    assert 2 <= metrics["masked_tokens"] <= 7
    assert 0 <= metrics["masked_accuracy"] <= 1
    assert all(type(value) in (int, float) for value in metrics.values())
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.condition_embedding.weight.grad[2:].abs().sum() > 0
    assert model.head.weight.grad.abs().sum() > 0
    assert model.latent_embedding.weight.grad[PAD].count_nonzero() == 0
    assert model.condition_embedding.weight.grad[PAD].count_nonzero() == 0
    assert torch.equal(clean, saved_clean)
    assert torch.equal(condition, saved_condition)


def test_padding_does_not_change_predictions_or_loss_and_all_pad_is_finite():
    model = make_model().eval()
    noisy = torch.tensor([[MASK, 3, MASK]])
    condition = torch.tensor([[2, 3]])
    noise = torch.tensor([0.5])
    base = model(noisy, condition, noise)
    padded = torch.tensor([[MASK, 3, MASK, PAD, PAD]])
    padded_condition = torch.tensor([[2, 3, PAD, PAD]])
    with torch.no_grad():
        model.latent_embedding.weight[PAD].fill_(1000)
        model.condition_embedding.weight[PAD].fill_(2000)
    logits, hidden = model(padded, padded_condition, noise, return_hidden=True)
    torch.testing.assert_close(base, logits[:, :3], rtol=1e-5, atol=5e-7)
    assert logits[:, 3:].count_nonzero() == 0
    assert hidden[:, 3:].count_nonzero() == 0
    empty, empty_hidden = model(
        torch.zeros((2, 4), dtype=torch.long), torch.zeros((2, 3), dtype=torch.long),
        torch.zeros(2), return_hidden=True,
    )
    assert torch.isfinite(empty).all() and torch.isfinite(empty_hidden).all()
    assert empty.count_nonzero() == 0 and empty_hidden.count_nonzero() == 0
    # Empty and all-padding conditions both represent absent evidence.
    empty_condition = model(noisy, condition[:, :0], noise)
    all_pad_condition = model(noisy, torch.zeros_like(condition), noise)
    torch.testing.assert_close(empty_condition, all_pad_condition, rtol=1e-5, atol=5e-7)
    clean = torch.tensor([[2, 3, 4]])
    loss, metrics = masked_loss(model, clean, condition, generator=rng())
    padded_loss, padded_metrics = masked_loss(
        model, torch.tensor([[2, 3, 4, PAD, PAD]]), padded_condition, generator=rng(),
    )
    torch.testing.assert_close(loss, padded_loss, rtol=1e-5, atol=5e-7)
    assert metrics["masked_tokens"] == padded_metrics["masked_tokens"]


def test_seeded_masking_and_sampling_are_reproducible():
    model = make_model()
    condition = torch.tensor([[2, 3], [4, 5]])
    clean = torch.tensor([[2, 3, 4, 5], [5, 6, 7, 8]])
    loss1, metrics1 = masked_loss(model, clean, condition, generator=rng())
    loss2, metrics2 = masked_loss(model, clean, condition, generator=rng())
    torch.testing.assert_close(loss1, loss2, rtol=0, atol=0)
    assert metrics1 == metrics2
    model = make_model(dropout=0.6)
    output1, trace1 = sample(model, condition, length=8, steps=5, generator=rng(), return_trace=True)
    output2, trace2 = sample(model, condition, length=8, steps=5, generator=rng(), return_trace=True)
    assert torch.equal(output1, output2)
    assert model.training
    for left, right in zip(trace1, trace2, strict=True):
        for key in ("tokens", "noisy_tokens", "remasked", "noise_level"):
            assert torch.equal(left[key], right[key])
    output3 = sample(model, condition, length=8, steps=5, generator=rng(222))
    assert not torch.equal(output1, output3)


def test_fixed_padding_and_allowed_support_are_preserved_at_every_step():
    model = make_model()
    condition = torch.tensor([[2, 3], [4, 5]])
    fixed = torch.tensor([[2, -1, -1, PAD], [-1, 5, PAD, PAD]])
    allowed = torch.zeros((2, 4, 9), dtype=torch.bool)
    allowed[..., 2] = True
    allowed[0, 1, 3] = True
    allowed[0, 2, 4] = True
    allowed[1, 1, 5] = True
    allowed[fixed == PAD] = False  # Padding needs no ordinary category support.
    saved = (condition.clone(), fixed.clone(), allowed.clone())
    tokens, trace = sample(
        model, condition, length=4, steps=6, generator=rng(), fixed=fixed,
        allowed=allowed, return_trace=True,
    )
    for entry in trace:
        concrete = entry["tokens"]
        assert torch.equal(concrete[fixed >= 0], fixed[fixed >= 0])
        assert not (concrete == MASK).any()
        assert not entry["remasked"][fixed >= 0].any()
        support = allowed.gather(-1, concrete[..., None]).squeeze(-1)
        assert support[concrete != PAD].all()
        for key in ("tokens", "noisy_tokens", "remasked", "noise_level"):
            assert entry[key].device.type == "cpu" and not entry[key].requires_grad
    assert torch.equal(tokens, trace[-1]["tokens"])
    tokens.fill_(8)
    assert not torch.equal(tokens, trace[-1]["tokens"])
    for actual, expected in zip((condition, fixed, allowed), saved, strict=True):
        assert torch.equal(actual, expected)


def test_later_steps_really_revise_committed_values():
    model = make_model()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.weight[2, 0] = 1
        model.head.weight[3, 0] = -1
    calls = []

    def patch(step, hidden):
        calls.append(step)
        changed = torch.zeros_like(hidden)
        changed[..., 0] = 1000 if step == 0 else -1000
        return changed

    _, trace = sample(
        model, torch.tensor([[2, 3]]), length=8, steps=3, generator=rng(),
        hidden_patch=patch, return_trace=True,
    )
    assert calls == [0, 1, 2]
    assert (trace[0]["tokens"] == 2).all()
    revisited = trace[1]["remasked"]
    assert revisited.any()
    assert (trace[0]["tokens"][revisited] == 2).all()
    assert (trace[1]["noisy_tokens"][revisited] == MASK).all()
    assert (trace[1]["tokens"][revisited] == 3).all()
    assert torch.equal(trace[0]["tokens"][~revisited], trace[1]["tokens"][~revisited])


def test_initial_guesses_are_context_and_initial_padding_is_immutable():
    model = make_model()
    initial = torch.tensor([[2, MASK, 3, MASK, PAD], [4, 5, 6, 7, PAD]])
    original = initial.clone()
    _, trace = sample(
        model, torch.tensor([[2, 3], [4, 5]]), length=5, steps=4, initial=initial,
        generator=rng(), return_trace=True,
    )
    assert torch.equal(trace[0]["remasked"][0], initial[0] == MASK)
    assert trace[0]["remasked"][1].sum() == 2
    assert torch.equal(trace[0]["noisy_tokens"][0, [0, 2]], initial[0, [0, 2]])
    for entry in trace:
        assert (entry["tokens"][:, -1] == PAD).all()
        assert not entry["remasked"][:, -1].any()
    assert torch.equal(initial, original)


def test_loss_support_is_hard_and_clean_categories_are_validated():
    model = make_model()
    clean = torch.tensor([[2, 3, PAD]])
    allowed = torch.zeros((1, 3, 9), dtype=torch.bool)
    allowed[0, 0, 2] = True
    allowed[0, 1, 3] = True
    loss, metrics = masked_loss(model, clean, torch.tensor([[2]]), allowed=allowed, generator=rng())
    assert loss.item() == 0
    assert metrics["masked_accuracy"] == 1
    allowed[0, 1, 3] = False
    with pytest.raises(ValueError, match="nonempty category support"):
        masked_loss(model, clean, torch.tensor([[2]]), allowed=allowed)
    allowed[0, 1, 4] = True
    with pytest.raises(ValueError, match="outside allowed support"):
        masked_loss(model, clean, torch.tensor([[2]]), allowed=allowed)


def test_patch_identity_and_model_modes_restored_on_patch_error():
    model = make_model()
    noisy, condition, noise = torch.tensor([[MASK, 2, PAD]]), torch.tensor([[2, 3]]), torch.tensor([0.5])
    baseline = model(noisy, condition, noise)
    patched, hidden = model(noisy, condition, noise, hidden_patch=lambda h: h, return_hidden=True)
    torch.testing.assert_close(baseline, patched, rtol=0, atol=0)
    torch.testing.assert_close(model.head(hidden)[:, :2], patched[:, :2])
    for patch in (lambda h: h.double(), lambda h: h[:, :1], lambda h: None):
        with pytest.raises(ValueError, match="preserve shape, device, and dtype"):
            model(noisy, condition, noise, hidden_patch=patch)
    with pytest.raises(ValueError, match="finite values"):
        model(noisy, condition, noise, hidden_patch=lambda h: h * float("nan"))
    model.layers[0].eval()
    before = [module.training for module in model.modules()]

    def broken(_step, _hidden):
        raise RuntimeError("deliberate patch failure")

    with pytest.raises(RuntimeError, match="deliberate patch failure"):
        sample(model, condition, length=3, hidden_patch=broken)
    assert before == [module.training for module in model.modules()]
    output = sample(model, condition, length=3, steps=2, hidden_patch=lambda h: h, generator=rng())
    assert output.shape == (1, 3)
    assert before == [module.training for module in model.modules()]


def test_serialization_round_trip_preserves_outputs_and_sampling():
    model = make_model().eval()
    storage = io.BytesIO()
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, storage)
    storage.seek(0)
    saved = torch.load(storage, weights_only=True)
    restored = ConditionalDenoiser(DenoiserConfig(**saved["config"])).eval()
    restored.load_state_dict(saved["state_dict"], strict=True)
    assert saved["config"] == model.config.to_dict()
    noisy, condition, noise = torch.tensor([[MASK, 3, MASK]]), torch.tensor([[2, 3]]), torch.tensor([0.5])
    torch.testing.assert_close(model(noisy, condition, noise), restored(noisy, condition, noise), rtol=0, atol=0)
    assert torch.equal(
        sample(model, condition, length=6, steps=4, generator=rng()),
        sample(restored, condition, length=6, steps=4, generator=rng()),
    )


@pytest.mark.parametrize("changes", [
    {"vocab_size": 2}, {"vocab_size": True}, {"condition_vocab_size": 0}, {"width": 15},
    {"heads": 0}, {"layers": -1}, {"max_latent_length": 0}, {"max_condition_length": "4"},
    {"type_vocab_size": 0}, {"dropout": 1}, {"dropout": -0.1}, {"dropout": float("nan")},
    {"dropout": float("inf")}, {"dropout": "0.1"}, {"dropout": True},
])
def test_invalid_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        make_model(**changes)


@pytest.mark.parametrize("noisy,condition,noise,extras", [
    (torch.tensor([1, 2]), torch.tensor([[2]]), torch.tensor([0.5]), {}),
    (torch.tensor([[1., 2.]]), torch.tensor([[2]]), torch.tensor([0.5]), {}),
    (torch.tensor([[-1, 2]]), torch.tensor([[2]]), torch.tensor([0.5]), {}),
    (torch.tensor([[1, 9]]), torch.tensor([[2]]), torch.tensor([0.5]), {}),
    (torch.ones((1, 17), dtype=torch.long), torch.tensor([[2]]), torch.tensor([0.5]), {}),
    (torch.ones((1, 0), dtype=torch.long), torch.tensor([[2]]), torch.tensor([0.5]), {}),
    (torch.tensor([[1, 2]]), torch.tensor([[12]]), torch.tensor([0.5]), {}),
    (torch.tensor([[1, 2]]), torch.tensor([[2.]]), torch.tensor([0.5]), {}),
    (torch.tensor([[1, 2]]), torch.ones((1, 13), dtype=torch.long), torch.tensor([0.5]), {}),
    (torch.tensor([[1, 2]]), torch.tensor([[2], [3]]), torch.tensor([0.5]), {}),
    (torch.tensor([[1, 2]]), torch.tensor([[2]]), torch.tensor([[0.5]]), {}),
    (torch.tensor([[1, 2]]), torch.tensor([[2]]), torch.tensor([1]), {}),
    (torch.tensor([[1, 2]]), torch.tensor([[2]]), torch.tensor([1.1]), {}),
    (torch.tensor([[1, 2]]), torch.tensor([[2]]), torch.tensor([float("nan")]), {}),
    (torch.tensor([[1, 2]]), torch.tensor([[2]]), torch.tensor([0.5]), {"latent_types": torch.tensor([[8, 0]])}),
    (torch.tensor([[1, 2]]), torch.tensor([[2]]), torch.tensor([0.5]), {"latent_types": torch.tensor([[0]])}),
    (torch.tensor([[1, 2]]), torch.tensor([[2]]), torch.tensor([0.5]), {"return_hidden": "yes"}),
])
def test_forward_invalid_shapes_and_domains_are_rejected(noisy, condition, noise, extras):
    with pytest.raises(ValueError):
        make_model()(noisy, condition, noise, **extras)


@pytest.mark.parametrize("kwargs", [
    {"steps": 0}, {"steps": True}, {"steps": 1.5}, {"length": 0}, {"length": 17},
    {"generator": 3}, {"fixed": torch.tensor([[MASK, -1]])},
    {"fixed": torch.tensor([[-2, -1]])}, {"initial": torch.tensor([[9, MASK]])},
    {"initial": torch.tensor([[2, MASK]]), "fixed": torch.tensor([[3, -1]])},
    {"initial": torch.tensor([[PAD, MASK]]), "fixed": torch.tensor([[3, -1]])},
    {"allowed": torch.zeros((1, 2, 9), dtype=torch.bool)},
    {"allowed": torch.ones((1, 2, 9), dtype=torch.long)},
    {"allowed": torch.ones((1, 2, 8), dtype=torch.bool)},
    {"return_trace": "yes"}, {"hidden_patch": lambda: None},
])
def test_sampling_invalid_inputs_are_rejected(kwargs):
    options = {"length": 2, "steps": 2, **kwargs}
    with pytest.raises(ValueError):
        sample(make_model(), torch.tensor([[2, 3]]), **options)


def test_sampling_rejects_unsupported_initial_or_fixed_category():
    allowed = torch.zeros((1, 2, 9), dtype=torch.bool)
    allowed[..., 2] = True
    for kwargs in ({"initial": torch.tensor([[3, MASK]])}, {"fixed": torch.tensor([[3, -1]])}):
        with pytest.raises(ValueError, match="outside allowed support"):
            sample(make_model(), torch.tensor([[2, 3]]), length=2, allowed=allowed, **kwargs)


def test_fixed_only_and_all_pad_sampling_have_no_masks_and_finite_traces():
    fixed = torch.tensor([[2, 3, PAD], [PAD, PAD, PAD]])
    output, trace = sample(
        make_model(), torch.zeros((2, 0), dtype=torch.long), length=3, steps=2,
        fixed=fixed, return_trace=True,
    )
    assert torch.equal(output, fixed)
    for entry in trace:
        assert not entry["remasked"].any()
        assert torch.isfinite(entry["noise_level"]).all()
        assert (entry["noise_level"] == 0).all()


@pytest.mark.parametrize("clean", [torch.tensor([[MASK, 2]]), torch.tensor([[PAD, PAD]])])
def test_loss_rejects_corrupted_truth_or_no_targets(clean):
    with pytest.raises(ValueError):
        masked_loss(make_model(), clean, torch.tensor([[2, 3]]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_uses_cpu_generator_for_reproducible_loss_and_sampling():
    model = make_model().to("mps")
    clean = torch.tensor([[2, 3, 4]], device="mps")
    condition = torch.tensor([[2, 3]], device="mps")
    first_loss, first_metrics = masked_loss(model, clean, condition, generator=rng())
    second_loss, second_metrics = masked_loss(model, clean, condition, generator=rng())
    torch.testing.assert_close(first_loss, second_loss, rtol=0, atol=0)
    assert first_metrics == second_metrics
    first_loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    first = sample(model, condition, length=5, steps=3, generator=rng())
    second = sample(model, condition, length=5, steps=3, generator=rng())
    assert torch.equal(first, second)
    assert first.device.type == "mps" and not (first == MASK).any()
