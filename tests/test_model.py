"""Behavioral guarantees needed before training or interpreting a predictor."""

import itertools

import pytest
import torch
import torch.nn.functional as F

from voynich.model import ModelConfig, VoynichTransformer


VARIANTS = list(itertools.product([False, True], repeat=4))


@pytest.fixture(scope="module", autouse=True)
def limit_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_model(**changes):
    options = dict(
        vocab_size=17,
        pad_id=0,
        d_model=24,
        n_layers=2,
        n_heads=3,
        d_ff=32,
        context_length=16,
        dropout=0.2,
        auxiliary_horizons=(2, 3),
    )
    options.update(changes)
    torch.manual_seed(173)
    return VoynichTransformer(ModelConfig(**options)).eval()


def assert_same_predictions(first, second):
    torch.testing.assert_close(first.logits, second.logits, rtol=1e-5, atol=2e-7)
    assert first.auxiliary_logits.keys() == second.auxiliary_logits.keys()
    for horizon in first.auxiliary_logits:
        torch.testing.assert_close(
            first.auxiliary_logits[horizon], second.auxiliary_logits[horizon], rtol=1e-5, atol=2e-7
        )


@pytest.mark.parametrize("qk_norm,attention_gate,attention_only,tie_embeddings", VARIANTS)
@pytest.mark.parametrize("explicit", [False, True], ids=["sdpa", "explicit"])
def test_future_tokens_cannot_change_prefix_or_auxiliary_predictions(
    qk_norm, attention_gate, attention_only, tie_embeddings, explicit
):
    model = make_model(
        qk_norm=qk_norm,
        attention_gate=attention_gate,
        attention_only=attention_only,
        tie_embeddings=tie_embeddings,
    )
    clean = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [8, 7, 6, 5, 4, 3, 2]])
    changed = clean.clone()
    changed[:, 4:] = torch.tensor([[11, 12, 13], [14, 15, 16]])
    cache_names = ["*"] if explicit else None
    with torch.no_grad():
        first = model(clean, cache_names=cache_names)
        second = model(changed, cache_names=cache_names)
        truncated = model(clean[:, :4], cache_names=cache_names)
    torch.testing.assert_close(first.logits[:, :4], second.logits[:, :4], rtol=1e-5, atol=2e-7)
    torch.testing.assert_close(first.logits[:, :4], truncated.logits, rtol=1e-5, atol=2e-7)
    for horizon in (2, 3):
        torch.testing.assert_close(
            first.auxiliary_logits[horizon][:, :4],
            second.auxiliary_logits[horizon][:, :4],
            rtol=1e-5,
            atol=2e-7,
        )
        torch.testing.assert_close(
            first.auxiliary_logits[horizon][:, :4],
            truncated.auxiliary_logits[horizon],
            rtol=1e-5,
            atol=2e-7,
        )


@pytest.mark.parametrize("qk_norm,attention_gate,attention_only,tie_embeddings", VARIANTS)
def test_analysis_cache_and_identity_interventions_preserve_fast_predictions(
    qk_norm, attention_gate, attention_only, tie_embeddings
):
    model = make_model(
        qk_norm=qk_norm,
        attention_gate=attention_gate,
        attention_only=attention_only,
        tie_embeddings=tie_embeddings,
    )
    ids = torch.tensor([[1, 2, 3, 4, 0, 0], [6, 7, 8, 9, 10, 11]])
    with torch.no_grad():
        fast = model(ids)
        explicit = model(ids, cache_names=["*"])
        identity = model(ids, interventions={name: lambda x: x for name in explicit.cache})
    assert_same_predictions(fast, explicit)
    assert_same_predictions(fast, identity)
    assert fast.cache == identity.cache == {}
    assert all(not value.requires_grad for value in explicit.cache.values())
    assert model(ids).cache == {}  # Captures and interventions cannot leak into later calls.


@pytest.mark.parametrize("attention_gate", [False, True])
def test_head_contributions_sum_to_attention_update_and_preserve_mask(attention_gate):
    model = make_model(qk_norm=True, attention_gate=attention_gate)
    ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
    output = model(ids, cache_names=["*"])
    forbidden_future = torch.ones(5, 5, dtype=torch.bool).triu(diagonal=1)
    for layer in range(model.config.n_layers):
        base = f"blocks.{layer}.attn"
        result = output.cache[base + ".result"]
        assert result.shape == (2, 5, model.config.n_heads, model.config.d_model)
        torch.testing.assert_close(result.sum(dim=2), output.cache[base + ".out"])
        pattern = output.cache[base + ".pattern"]
        assert torch.count_nonzero(pattern.masked_select(forbidden_future[None, None])) == 0
        assert torch.count_nonzero(pattern[0, :, :, 3:]) == 0
        torch.testing.assert_close(pattern.sum(dim=-1), torch.ones_like(pattern[..., 0]))
        assert all(torch.isfinite(value).all() for value in output.cache.values())


@pytest.mark.parametrize("site", ["embed", "blocks.0.resid_post", "blocks.1.resid_pre", "final_norm"])
def test_whole_residual_replacement_recovers_clean_predictions(site):
    model = make_model(qk_norm=True, attention_gate=True)
    clean = torch.tensor([[1, 2, 3, 4, 5], [7, 8, 9, 0, 0]])
    altered = torch.tensor([[10, 11, 12, 13, 14], [15, 16, 1, 0, 0]])
    with torch.no_grad():
        clean_output = model(clean, cache_names=[site])
        altered_output = model(altered)
        patched = model(
            altered,
            interventions={site: lambda _: clean_output.cache[site].clone()},
            cache_names=[site],
        )
    assert not torch.allclose(clean_output.logits, altered_output.logits)
    assert_same_predictions(clean_output, patched)
    torch.testing.assert_close(patched.cache[site], clean_output.cache[site])


def test_head_ablation_changes_downstream_predictions_without_changing_earlier_positions():
    model = make_model(attention_gate=True)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    site = "blocks.0.attn.result"

    def remove_one_head_after_prefix(value):
        changed = value.clone()
        changed[:, 3:, 1, :] = 0
        return changed

    with torch.no_grad():
        original = model(ids, cache_names=[site])
        ablated = model(ids, cache_names=[site], interventions={site: remove_one_head_after_prefix})
    torch.testing.assert_close(original.logits[:, :3], ablated.logits[:, :3], rtol=1e-5, atol=2e-7)
    assert torch.max((original.logits[:, 3:] - ablated.logits[:, 3:]).abs()) > 1e-5
    assert torch.count_nonzero(ablated.cache[site][:, 3:, 1, :]) == 0
    torch.testing.assert_close(ablated.cache[site][:, :, 0, :], original.cache[site][:, :, 0, :])


@pytest.mark.parametrize("cache_names,interventions", [
    (["blocks.99.resid_pre"], None),
    (None, {"blocks.0.attn.typo": lambda x: x}),
    (["blocks.0.attn.gate"], None),
    (["*", "not_a_site"], None),
])
def test_unknown_or_inactive_hooks_fail_explicitly(cache_names, interventions):
    model = make_model(attention_gate=False)
    with pytest.raises(ValueError, match="Unknown or inactive activation sites"):
        model(torch.tensor([[1, 2, 3]]), cache_names=cache_names, interventions=interventions)


def test_attention_only_model_rejects_mlp_hooks():
    with pytest.raises(ValueError, match="Unknown or inactive activation sites"):
        make_model(attention_only=True)(torch.tensor([[1, 2, 3]]), cache_names=["blocks.0.mlp.act"])


@pytest.mark.parametrize("change", [lambda x: x[:, :-1], lambda x: x.double()])
def test_interventions_must_preserve_tensor_contract(change):
    with pytest.raises(ValueError, match="preserve shape, device, and dtype"):
        make_model()(torch.tensor([[1, 2, 3]]), interventions={"embed": change})


@pytest.mark.parametrize("explicit", [False, True])
def test_right_padding_and_pad_embedding_do_not_affect_valid_predictions(explicit):
    model = make_model(qk_norm=True, attention_gate=True)
    unpadded = torch.tensor([[1, 2, 3, 4]])
    padded = torch.tensor([[1, 2, 3, 4, 0, 0, 0]])
    cache_names = ["*"] if explicit else None
    with torch.no_grad():
        original = model(unpadded, cache_names=cache_names)
        padded_output = model(padded, cache_names=cache_names)
        model.embedding.weight[0].fill_(1000)
        changed_pad = model(padded, cache_names=cache_names)
    for output in (padded_output, changed_pad):
        torch.testing.assert_close(original.logits, output.logits[:, :4], rtol=1e-5, atol=2e-7)
        assert torch.isfinite(output.logits).all()
        for horizon in (2, 3):
            torch.testing.assert_close(
                original.auxiliary_logits[horizon], output.auxiliary_logits[horizon][:, :4],
                rtol=1e-5, atol=2e-7,
            )


def test_valid_target_losses_train_gates_auxiliary_heads_and_backbone():
    model = make_model(qk_norm=True, attention_gate=True, dropout=0.0)
    model.train()
    ids = torch.tensor([[1, 2, 3, 4, 5, 0], [6, 7, 8, 9, 10, 11]])
    output = model(ids)
    losses = []
    for horizon, logits in [(1, output.logits), *output.auxiliary_logits.items()]:
        targets = ids[:, horizon:]
        losses.append(F.cross_entropy(
            logits[:, :-horizon].reshape(-1, model.config.vocab_size),
            targets.reshape(-1), ignore_index=model.config.pad_id,
        ))
    sum(losses).backward()
    required = [model.embedding.weight, model.unembedding.weight]
    required.extend(head.weight for head in model.auxiliary_heads.values())
    for block in model.blocks:
        required.extend([block.attn.gate.weight, block.attn.q_norm.weight, block.attn.k_norm.weight])
        required.extend([block.attn.q.weight, block.attn.k.weight, block.attn.v.weight, block.down.weight])
    for parameter in required:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert torch.count_nonzero(model.embedding.weight.grad[model.config.pad_id]) == 0


@pytest.mark.parametrize("changes", [
    {"d_model": 0}, {"d_model": 23}, {"n_heads": 0}, {"n_heads": 4, "d_model": 12},
    {"n_layers": 0}, {"d_ff": -1}, {"context_length": 0},
    {"dropout": -0.1}, {"dropout": 1.0}, {"rope_base": 0},
    {"auxiliary_horizons": (1,)}, {"auxiliary_horizons": (17,)},
    {"auxiliary_horizons": (2, 2)}, {"vocab_size": 1}, {"pad_id": -1}, {"pad_id": 17},
])
def test_invalid_model_config_is_rejected(changes):
    with pytest.raises(ValueError):
        make_model(**changes)


@pytest.mark.parametrize("name", [
    "vocab_size", "pad_id", "d_model", "n_layers", "n_heads", "d_ff", "context_length",
])
@pytest.mark.parametrize("value", [2.5, True, "2"])
def test_config_rejects_noninteger_dimensions_before_module_construction(name, value):
    with pytest.raises(ValueError, match=f"{name} must be an integer"):
        ModelConfig(**{name: value})


@pytest.mark.parametrize("name", ["qk_norm", "attention_gate", "attention_only", "tie_embeddings"])
def test_config_rejects_truthy_strings_instead_of_silently_enabling_features(name):
    with pytest.raises(ValueError, match=f"{name} must be a boolean"):
        ModelConfig(**{name: "false"})


@pytest.mark.parametrize("name", ["dropout", "rope_base"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), "0.1", True])
def test_config_rejects_nonfinite_or_nonnumeric_scalars(name, value):
    with pytest.raises(ValueError, match=f"{name} must be a finite number"):
        ModelConfig(**{name: value})


@pytest.mark.parametrize("horizons", [(2.5,), (True,), ("2",), None, 2])
def test_auxiliary_horizons_require_an_integer_sequence(horizons):
    with pytest.raises(ValueError, match="auxiliary horizons"):
        ModelConfig(auxiliary_horizons=horizons)


def test_config_accepts_json_horizon_lists_and_unbound_default_vocabulary():
    config = ModelConfig(auxiliary_horizons=[2, 3])
    assert config.auxiliary_horizons == (2, 3)
    assert config.vocab_size == 0
    with pytest.raises(ValueError, match="valid vocabulary"):
        VoynichTransformer(config)


@pytest.mark.parametrize("ids,error", [
    (torch.tensor([1, 2, 3]), "nonempty"),
    (torch.empty((1, 0), dtype=torch.long), "nonempty"),
    (torch.empty((0, 3), dtype=torch.long), "nonempty"),
    (torch.ones((1, 17), dtype=torch.long), "exceeds"),
    (torch.tensor([[0, 1, 2]]), "non-padding"),
    (torch.tensor([[0, 0, 0]]), "non-padding"),
    (torch.tensor([[1, 0, 2]]), "right padding"),
])
def test_invalid_input_layout_is_rejected(ids, error):
    with pytest.raises(ValueError, match=error):
        make_model()(ids)


def test_checkpoint_reload_and_parameter_tying_preserve_outputs():
    model = make_model(tie_embeddings=True, qk_norm=True, attention_gate=True)
    assert model.embedding.weight is model.unembedding.weight
    restored = make_model(tie_embeddings=True, qk_norm=True, attention_gate=True)
    restored.load_state_dict(model.state_dict(), strict=True)
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        assert_same_predictions(model(ids), restored(ids))
    assert model.parameter_count == sum(parameter.numel() for parameter in model.parameters())
