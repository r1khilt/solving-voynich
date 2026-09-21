import json
import math

import pytest
import torch

from voynich.communication.denoiser import ConditionalDenoiser, DenoiserConfig
from voynich.communication.interpretation import (
    TransitionConflictError, TransitionProgram, causal_report, compare_distributions,
    evaluate_transition_program, extract_transition_program, intervention_controls, patch_hidden,
)


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def hidden_pair():
    base = torch.arange(24, dtype=torch.float).reshape(2, 3, 4)
    return base, base + 2


def test_identity_full_and_selected_patches_are_exact_without_input_mutation():
    base, donor = hidden_pair()
    saved_base, saved_donor = base.clone(), donor.clone()
    assert torch.equal(patch_hidden(base, base, [0, 2]), base)
    assert torch.equal(patch_hidden(base, donor, range(3)), donor)
    assert torch.equal(patch_hidden(base, donor, []), base)
    assert torch.equal(patch_hidden(base, donor, [0], dimensions=[]), base)
    targeted = patch_hidden(base, donor, [0, 2], dimensions=[1, 3])
    expected = base.clone()
    expected[:, [0, 2], 1] = donor[:, [0, 2], 1]
    expected[:, [0, 2], 3] = donor[:, [0, 2], 3]
    assert torch.equal(targeted, expected)
    assert torch.equal(base, saved_base) and torch.equal(donor, saved_donor)


def test_boolean_per_example_selection_and_gradient_routing():
    base = torch.randn(2, 3, 4, requires_grad=True)
    donor = torch.randn(2, 3, 4, requires_grad=True)
    positions = torch.tensor([[True, False, False], [False, True, False]])
    dimensions = torch.tensor([False, True, False, True])
    mask = positions[..., None] & dimensions
    patched = patch_hidden(base, donor, positions, dimensions)
    patched.sum().backward()
    assert torch.equal(base.grad, (~mask).float())
    assert torch.equal(donor.grad, mask.float())


def test_random_control_preserves_support_and_matches_delta_norm_per_example():
    base, donor = hidden_pair()
    rng_before = torch.random.get_rng_state().clone()
    controls = intervention_controls(base, donor, [1], dimensions=[0, 2], seed=19)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert set(controls) == {"identity", "targeted", "full_donor", "norm_matched_random"}
    assert torch.equal(controls["identity"], base)
    assert torch.equal(controls["full_donor"], donor)
    random_delta, target_delta = controls["norm_matched_random"] - base, controls["targeted"] - base
    torch.testing.assert_close(random_delta.flatten(1).norm(dim=1), target_delta.flatten(1).norm(dim=1))
    assert (random_delta[:, [0, 2]] == 0).all() and (random_delta[..., [1, 3]] == 0).all()
    assert not torch.equal(controls["norm_matched_random"], controls["targeted"])
    repeat = intervention_controls(base, donor, [1], dimensions=[0, 2], seed=19)
    assert torch.equal(controls["norm_matched_random"], repeat["norm_matched_random"])
    zero = intervention_controls(base, base, [1], seed=19)
    assert torch.equal(zero["norm_matched_random"], base)


def test_distribution_identity_and_analytic_disjoint_support():
    p = torch.tensor([[0.25, 0.75]], dtype=torch.double)
    same = compare_distributions(p, p)
    assert same["mean_kl_base_to_changed"] == 0
    assert same["mean_js_divergence"] == 0
    assert same["mean_total_variation"] == 0
    assert same["argmax_agreement"] == 1
    extremes = compare_distributions(torch.tensor([1., 0.]), torch.tensor([0., 1.]))
    assert extremes["mean_kl_base_to_changed"] == math.inf
    assert extremes["mean_kl_changed_to_base"] == math.inf
    assert extremes["mean_js_divergence"] == pytest.approx(math.log(2))
    assert extremes["mean_total_variation"] == 1
    assert extremes["argmax_agreement"] == 0
    assert not any(math.isnan(value) for value in extremes.values())


def test_logits_and_probabilities_agree_with_extreme_and_zero_logits_supported():
    base = torch.tensor([[10000., 10001.], [0., -math.inf]], dtype=torch.double)
    changed = torch.tensor([[10001., 10000.], [0., -math.inf]], dtype=torch.double)
    logits_report = compare_distributions(base, changed, input_kind="logits")
    probs_report = compare_distributions(base.softmax(-1), changed.softmax(-1))
    assert logits_report == pytest.approx(probs_report)


def test_causal_report_identity_full_donor_controls_and_mode_restoration():
    torch.manual_seed(42)
    model = ConditionalDenoiser(DenoiserConfig(8, 10, 4, 4, width=16, layers=1, heads=2))
    model.train()
    model.layers[0].eval()
    modes = [module.training for module in model.modules()]
    noisy = torch.tensor([[1, 1, 1, 0]])
    allowed = torch.zeros((1, 4, 8), dtype=torch.bool)
    allowed[:, :3, 2:] = True
    report = causal_report(
        model, noisy, torch.tensor([[2, 3]]), torch.tensor([1.]),
        donor_condition=torch.tensor([[6, 7]]), positions=[0], dimensions=[0, 1], allowed=allowed,
    )
    identity = report["controls"]["identity"]
    full = report["controls"]["full_donor"]
    assert identity["against_base"]["mean_total_variation"] == 0
    assert full["against_donor"]["mean_total_variation"] == 0
    assert report["base_to_donor"]["mean_total_variation"] > 0
    assert report["controls"]["targeted"]["other_positions"]["mean_total_variation"] == 0
    assert report["active_positions"] == 3 and report["selected_coordinates"] == 2
    assert [module.training for module in model.modules()] == modes
    json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError, match="nonempty boolean output support"):
        causal_report(model, noisy, torch.tensor([[2, 3]]), torch.tensor([1.]), positions=[0],
                      allowed=torch.zeros_like(allowed))
    assert [module.training for module in model.modules()] == modes


def test_transition_extraction_and_held_out_coverage_keep_unknown_explicit():
    program = extract_transition_program([("s0", "go", "s1"), ("s1", "go", "s2"), ("s0", "go", "s1")])
    assert program.training_examples == 3 and len(program.rules) == 2
    assert program.predict("s0", "go") == {"known": True, "next_state": "s1"}
    assert program.predict("s9", "go") == {"known": False, "next_state": None}
    metrics = evaluate_transition_program(program, [("s0", "go", "s1"), ("s1", "go", "wrong"), ("s9", "go", "s0")])
    assert metrics["examples"] == 3 and metrics["known"] == 2 and metrics["unknown"] == 1
    assert metrics["correct"] == 1 and metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["accuracy_on_known"] == 0.5 and metrics["accuracy_overall"] == pytest.approx(1 / 3)
    assert metrics["predictions"][-1]["status"] == "unknown"
    assert metrics["predictions"][-1]["correct"] is None
    wholly_unknown = evaluate_transition_program(program, [("unseen", "new", "truth")])
    assert wholly_unknown["accuracy_on_known"] is None
    assert wholly_unknown["coverage"] == 0 and wholly_unknown["accuracy_overall"] == 0


def test_conflicting_transitions_raise_or_are_excluded_in_report_mode():
    triples = [(0, "a", 1), (0, "a", 2), (0, "a", 1), (1, "a", 2)]
    with pytest.raises(TransitionConflictError) as raised:
        extract_transition_program(triples)
    conflict = raised.value.conflicts[0]
    assert conflict == {"state": 0, "action": "a", "outcomes": [{"next_state": 1, "count": 2},
                                                                 {"next_state": 2, "count": 1}]}
    reported = extract_transition_program(triples, on_conflict="report")
    assert len(reported.rules) == 1 and len(reported.conflicts) == 1
    assert reported.predict(0, "a")["known"] is False
    assert reported.predict(1, "a") == {"known": True, "next_state": 2}


def test_json_transition_states_round_trip_null_outcomes_and_immutable_records():
    source = {"objects": [1, 2]}
    program = extract_transition_program([(source, {"move": 1}, {"objects": [2, 1]}), ("ending", "stop", None)])
    source["objects"][0] = 99
    assert program.predict({"objects": [1, 2]}, {"move": 1}) == {"known": True, "next_state": {"objects": [2, 1]}}
    assert program.predict("ending", "stop") == {"known": True, "next_state": None}
    restored = TransitionProgram.from_dict(json.loads(program.to_json()))
    assert restored.to_dict() == program.to_dict()
    json.dumps(restored.to_dict(), allow_nan=False)
    conflicting = extract_transition_program([(0, 1, 2), (0, 1, 3)], on_conflict="report")
    assert TransitionProgram.from_dict(conflicting.to_dict()).to_dict() == conflicting.to_dict()


@pytest.mark.parametrize("positions,dimensions", [([-1], None), ([3], None), ([True], None), ([0, 0], None),
                                                     ([0], [-1]), ([0], [4]), ([0], [1, 1]),
                                                     (torch.ones((1, 3), dtype=torch.bool), None),
                                                     ([0], torch.ones((2, 4), dtype=torch.bool))])
def test_patch_invalid_selections_are_rejected(positions, dimensions):
    base, donor = hidden_pair()
    with pytest.raises(ValueError):
        patch_hidden(base, donor, positions, dimensions)


@pytest.mark.parametrize("base,changed,kind", [
    (torch.tensor([1, 0]), torch.tensor([0, 1]), "probabilities"),
    (torch.tensor([0.2, 0.2]), torch.tensor([0.5, 0.5]), "probabilities"),
    (torch.tensor([-0.1, 1.1]), torch.tensor([0.5, 0.5]), "probabilities"),
    (torch.tensor([float("nan"), 0.]), torch.tensor([0.5, 0.5]), "probabilities"),
    (torch.tensor([1., 0.]), torch.tensor([0., 1., 0.]), "probabilities"),
    (torch.tensor([-math.inf, -math.inf]), torch.tensor([0., 1.]), "logits"),
    (torch.tensor([math.inf, 1.]), torch.tensor([0., 1.]), "logits"),
    (torch.tensor([0., 1.]), torch.tensor([0., 1.]), "unknown"),
])
def test_invalid_distribution_inputs_are_rejected(base, changed, kind):
    with pytest.raises(ValueError):
        compare_distributions(base, changed, kind)


@pytest.mark.parametrize("triples,kwargs", [([], {}), ([(0, 1)], {}), ([(float("nan"), 1, 2)], {}),
                                           ([(0, 1, 2)], {"on_conflict": "majority"})])
def test_invalid_extraction_is_rejected(triples, kwargs):
    with pytest.raises(ValueError):
        extract_transition_program(triples, **kwargs)


def test_transition_import_cannot_execute_conflicted_or_duplicate_rules():
    program = extract_transition_program([(0, 1, 2)]).to_dict()
    program["rules"].append({"state": 0, "action": 1, "next_state": 3})
    with pytest.raises(ValueError, match="duplicate or conflicting"):
        TransitionProgram.from_dict(program)
    program = extract_transition_program([(0, 1, 2)], on_conflict="report").to_dict()
    program["conflicts"] = [{"state": 0, "action": 1,
                              "outcomes": [{"next_state": 2, "count": 1}, {"next_state": 3, "count": 1}]}]
    with pytest.raises(ValueError, match="cannot have executable rules"):
        TransitionProgram.from_dict(program)


@pytest.mark.parametrize("mutate", [
    lambda record: record["conflicts"][0]["outcomes"][0].update(count=0),
    lambda record: record["conflicts"][0]["outcomes"][0].update(count=True),
    lambda record: record["conflicts"][0]["outcomes"][0].update(count=10),
    lambda record: record["conflicts"][0]["outcomes"][0].update(next_state=3),
    lambda record: record["conflicts"].append(record["conflicts"][0]),
])
def test_malformed_conflict_records_are_rejected(mutate):
    record = extract_transition_program([(0, 1, 2), (0, 1, 3)], on_conflict="report").to_dict()
    mutate(record)
    with pytest.raises(ValueError):
        TransitionProgram.from_dict(record)
