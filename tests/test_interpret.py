"""Checks that reported causal effects match the interventions actually applied."""

import json

import pytest
import torch

from voynich.interpret import ablate_head, patch_activation, patch_report, target_log_probability
from voynich.model import ModelConfig, VoynichTransformer


@pytest.fixture(scope="module", autouse=True)
def limit_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def model():
    torch.manual_seed(911)
    return VoynichTransformer(ModelConfig(
        vocab_size=17, d_model=24, n_layers=2, n_heads=3, d_ff=32,
        context_length=16, dropout=0.5, qk_norm=True, attention_gate=True,
    )).eval()


@pytest.fixture
def contexts():
    clean = torch.tensor([[1, 2, 3, 4, 5, 6], [1, 7, 8, 9, 10, 11]])
    corrupted = torch.tensor([[1, 12, 13, 14, 15, 16], [1, 5, 4, 3, 2, 7]])
    return clean, corrupted


@pytest.mark.parametrize("positions", [None, [0, 2], [-1]])
def test_identity_patch_keeps_values_and_does_not_alias_input(positions):
    value = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    original = value.clone()
    patched = patch_activation(value, positions=positions)(value)
    torch.testing.assert_close(patched, original, rtol=0, atol=0)
    assert patched.data_ptr() != value.data_ptr()
    patched.zero_()
    torch.testing.assert_close(value, original, rtol=0, atol=0)


def test_head_and_position_patch_changes_only_selected_values():
    recipient = torch.arange(120, dtype=torch.float32).reshape(2, 5, 3, 4)
    original = recipient.clone()
    donor = (recipient + 1000).double()
    patched = patch_activation(donor, positions=[1, -1], head=2)(recipient)
    selected = torch.zeros_like(recipient, dtype=torch.bool)
    selected[:, [1, -1], 2, :] = True
    torch.testing.assert_close(patched[selected], donor.float()[selected], rtol=0, atol=0)
    torch.testing.assert_close(patched[~selected], recipient[~selected], rtol=0, atol=0)
    torch.testing.assert_close(recipient, original, rtol=0, atol=0)
    assert patched.dtype == recipient.dtype
    assert patched.device == recipient.device
    torch.testing.assert_close(donor, (original + 1000).double(), rtol=0, atol=0)


@pytest.mark.parametrize("donor_shape", [(1, 5, 4), (2, 4, 4), (2, 5, 3), (2, 5, 1, 4)])
def test_donor_recipient_shape_mismatch_fails(donor_shape):
    with pytest.raises(ValueError, match="shapes must match exactly"):
        patch_activation(torch.zeros(donor_shape))(torch.zeros(2, 5, 4))


@pytest.mark.parametrize("head", [-1, 3])
def test_invalid_head_index_fails_for_patch_and_ablation(head):
    value = torch.ones(2, 5, 3, 4)
    with pytest.raises(ValueError, match="valid head"):
        patch_activation(value, head=head)(value)
    with pytest.raises(ValueError, match="valid head"):
        ablate_head(head)(value)


def test_head_operations_reject_residual_tensors():
    value = torch.ones(2, 5, 4)
    with pytest.raises(ValueError, match="Head patch requires"):
        patch_activation(value, head=1)(value)
    with pytest.raises(ValueError, match="Head ablation requires"):
        ablate_head(1)(value)


def test_ablation_removes_exactly_one_head_without_mutating_recipient():
    value = torch.arange(1, 121, dtype=torch.float32).reshape(2, 5, 3, 4)
    original = value.clone()
    ablated = ablate_head(1)(value)
    assert torch.count_nonzero(ablated[:, :, 1]) == 0
    torch.testing.assert_close(ablated[:, :, [0, 2]], original[:, :, [0, 2]], rtol=0, atol=0)
    torch.testing.assert_close(value, original, rtol=0, atol=0)


def test_score_uses_last_position_and_mean_of_log_probabilities():
    logits = torch.tensor([[[500., -500.], [0., 2.]], [[-500., 500.], [1., 0.]]])
    expected = (torch.log(torch.sigmoid(torch.tensor(2.))) +
                torch.log(torch.sigmoid(torch.tensor(-1.)))) / 2
    assert target_log_probability(logits, 1) == pytest.approx(float(expected), abs=1e-7)


@pytest.mark.parametrize("site", ["blocks.1.resid_post", "final_norm"])
def test_full_final_residual_patch_recovers_clean_score_and_reverse_recovers_corruption(model, contexts, site):
    clean, corrupted = contexts
    report = patch_report(model, clean, corrupted, 3, site)
    assert report["positions"] == "all"
    assert abs(report["clean_corrupted_gap"]) > 1e-4
    assert report["patched"] == pytest.approx(report["clean"], abs=1e-7)
    assert report["reverse_patch"] == pytest.approx(report["corrupted"], abs=1e-7)
    assert report["identity_patch"] == pytest.approx(report["corrupted"], abs=1e-7)
    assert report["identity_max_logit_difference"] < 1e-7
    assert report["normalized_recovery"] == pytest.approx(1.0, abs=1e-6)
    assert report["absolute_patch_effect"] == pytest.approx(report["clean_corrupted_gap"], abs=1e-7)
    json.dumps(report, allow_nan=False)


def test_reported_head_patch_score_equals_direct_selective_intervention(model, contexts):
    clean, corrupted = contexts
    site = "blocks.0.attn.result"
    report = patch_report(model, clean, corrupted, 4, site, positions=[2, -1], head=1)
    with torch.no_grad():
        donor = model(clean, cache_names=[site]).cache[site]
        direct = model(corrupted, interventions={site: patch_activation(donor, positions=[2, -1], head=1)})
    assert report["patched"] == pytest.approx(target_log_probability(direct.logits, 4), abs=1e-7)
    assert report["positions"] == [2, -1]
    assert report["head"] == 1
    assert report["identity_max_logit_difference"] < 1e-7


def test_zero_gap_does_not_report_normalized_recovery(model, contexts):
    clean, _ = contexts
    report = patch_report(model, clean, clean.clone(), 3, "blocks.0.resid_pre")
    assert report["clean_corrupted_gap"] == 0
    assert report["absolute_patch_effect"] == 0
    assert report["normalized_recovery"] is None
    json.dumps(report, allow_nan=False)


def test_small_nonzero_gap_does_not_amplify_normalized_recovery(model, contexts):
    clean, corrupted = contexts
    with torch.no_grad():
        model.unembedding.weight.mul_(1e-4)
    report = patch_report(model, clean, corrupted, 3, "blocks.1.resid_post")
    assert 0 < abs(report["clean_corrupted_gap"]) < 1e-4
    assert report["normalized_recovery"] is None
    assert report["patched"] == pytest.approx(report["clean"], abs=1e-7)


def test_shuffled_control_is_seeded_preserves_first_token_and_records_actual_score(model, contexts):
    clean, corrupted = contexts
    state_before = torch.get_rng_state().clone()
    first = patch_report(model, clean, corrupted, 3, "final_norm", seed=13)
    second = patch_report(model, clean, corrupted, 3, "final_norm", seed=13)
    other_seed = patch_report(model, clean, corrupted, 3, "final_norm", seed=23)
    assert first == second
    assert first["shuffled_donor_ids"] != other_seed["shuffled_donor_ids"]
    torch.testing.assert_close(torch.get_rng_state(), state_before, rtol=0, atol=0)
    shuffled = torch.tensor(first["shuffled_donor_ids"])
    torch.testing.assert_close(shuffled[:, 0], clean[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(shuffled[:, 1:].sort().values, clean[:, 1:].sort().values, rtol=0, atol=0)
    with torch.no_grad():
        expected = target_log_probability(model(shuffled).logits, 3)
    assert first["shuffled_donor_patch"] == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("was_training", [False, True])
def test_report_restores_training_mode_and_disables_dropout_during_analysis(model, contexts, was_training):
    clean, corrupted = contexts
    model.train(was_training)
    first = patch_report(model, clean, corrupted, 3, "blocks.0.attn.result", head=0)
    assert model.training is was_training
    assert all(module.training is was_training for module in model.modules())
    second = patch_report(model, clean, corrupted, 3, "blocks.0.attn.result", head=0)
    assert first == second
    assert model.training is was_training
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("was_training", [False, True])
def test_report_restores_mode_after_failed_hook_lookup(model, contexts, was_training):
    clean, corrupted = contexts
    model.train(was_training)
    with pytest.raises(ValueError, match="Unknown or inactive"):
        patch_report(model, clean, corrupted, 3, "blocks.99.resid_pre")
    assert model.training is was_training
    assert all(module.training is was_training for module in model.modules())


def test_report_rejects_misaligned_contexts(model, contexts):
    clean, corrupted = contexts
    with pytest.raises(ValueError, match="identical shape"):
        patch_report(model, clean, corrupted[:, :-1], 3, "final_norm")


@pytest.mark.parametrize("target", [-1, 17])
def test_report_rejects_invalid_target_ids(model, contexts, target):
    clean, corrupted = contexts
    with pytest.raises(ValueError, match="valid target"):
        patch_report(model, clean, corrupted, target, "final_norm")


@pytest.mark.parametrize("pad_clean", [False, True])
def test_report_rejects_padding_in_either_context(model, contexts, pad_clean):
    clean, corrupted = contexts
    padded = clean if pad_clean else corrupted
    padded[0, -1] = model.config.pad_id
    with pytest.raises(ValueError, match="unpadded"):
        patch_report(model, clean, corrupted, 3, "final_norm")


@pytest.mark.parametrize("site", ["blocks.0.attn.pattern", "blocks.0.attn.scores"])
@pytest.mark.parametrize("head", [None, 1])
def test_report_rejects_attention_matrix_axes_instead_of_mislabeling_intervention(model, contexts, site, head):
    clean, corrupted = contexts
    model.train()
    with pytest.raises(ValueError, match="Unsupported activation layout"):
        patch_report(model, clean, corrupted, 3, site, positions=[2], head=head)
    assert model.training


@pytest.mark.parametrize("site", ["final_norm", "blocks.0.resid_pre", "blocks.0.attn.gate"])
def test_report_does_not_treat_feature_or_scalar_gate_axis_as_vector_heads(model, contexts, site):
    clean, corrupted = contexts
    with pytest.raises(ValueError, match="vector-head activation site"):
        patch_report(model, clean, corrupted, 3, site, head=1)


@pytest.mark.parametrize("head", [-1, 3, 1.5, True])
def test_report_requires_a_valid_integer_head_before_forward_pass(model, contexts, head):
    clean, corrupted = contexts
    with pytest.raises(ValueError, match="valid integer head"):
        patch_report(model, clean, corrupted, 3, "blocks.0.attn.result", head=head)


@pytest.mark.parametrize("site", ["blocks.0.attn.q", "blocks.0.attn.z_gated"])
def test_report_supports_documented_additional_time_axis_head_sites(model, contexts, site):
    clean, corrupted = contexts
    report = patch_report(model, clean, corrupted, 3, site, positions=[-1], head=1)
    with torch.no_grad():
        donor = model(clean, cache_names=[site]).cache[site]
        direct = model(corrupted, interventions={site: patch_activation(donor, positions=[-1], head=1)})
    assert report["patched"] == pytest.approx(target_log_probability(direct.logits, 3), abs=1e-7)


def test_report_can_patch_all_scalar_gates_at_a_position_without_head_selection(model, contexts):
    clean, corrupted = contexts
    report = patch_report(model, clean, corrupted, 3, "blocks.0.attn.gate", positions=[-1])
    assert report["head"] is None
    assert report["identity_max_logit_difference"] < 1e-7
