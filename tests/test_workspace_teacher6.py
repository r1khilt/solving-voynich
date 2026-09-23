import torch

from voynich.workspace.teacher6_geometry import (
    _map_states,
    _procrustes,
    decide,
    key_centroid_basis,
    nearest_centroid_accuracy,
    score_predictions,
)


def test_teacher6_centroid_span_recovers_synthetic_classes():
    states = torch.tensor([
        [2.0, 0.0, 0.0, 0.0], [2.1, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0], [0.0, 2.1, 0.0, 0.0],
        [-2.0, -2.0, 0.0, 0.0], [-2.1, -2.1, 0.0, 0.0],
    ])
    labels = torch.tensor([21, 21, 22, 22, 23, 23])
    geometry = key_centroid_basis(states, labels)
    accuracy, predictions = nearest_centroid_accuracy(
        states, labels, geometry["centroids"], geometry["keys"])
    assert geometry["rank"] == 2
    assert accuracy == 1.0
    assert torch.equal(predictions, labels)
    projector = geometry["basis"] @ geometry["basis"].T
    assert torch.allclose(projector @ projector, projector, atol=1e-6)


def test_teacher6_twelve_centered_centroids_cannot_report_rank_twelve():
    torch.manual_seed(19)
    centroids = torch.randn(12, 128, dtype=torch.float32)
    states = torch.repeat_interleave(centroids, 2, dim=0)
    labels = torch.repeat_interleave(torch.arange(21, 33), 2)
    geometry = key_centroid_basis(states, labels)
    assert geometry["rank"] <= 11


def test_teacher6_procrustes_recovers_rotated_states():
    torch.manual_seed(4)
    source = torch.randn(40, 8)
    rotation = torch.linalg.qr(torch.randn(8, 8)).Q
    target = source @ rotation + 3.0
    source_mean, target_mean, fitted_rotation, residual = _procrustes(source, target)
    mapped = _map_states(source, source_mean, target_mean, fitted_rotation)
    assert residual < 1e-5
    assert torch.allclose(mapped, target, atol=1e-5)


def test_teacher6_scoring_and_frozen_decisions():
    donor = torch.tensor([36, 37, 38, 39, 40, 41])
    base = torch.tensor([33, 34, 35, 33, 34, 35])
    score = score_predictions(donor, donor, base, groups=2)
    assert score["donor_items"]["accuracy"] == 1.0
    assert score["donor_groups"]["accuracy"] == 1.0
    metrics = {}
    for rep in ("0", "1"):
        metrics[rep] = {
            "rank": 11,
            "nearest_centroid_confirmation_accuracy": 1.0,
            "random_subspace_p95": .2,
            "rotated_top_rank_p95": .3,
            "scores": {
                "clean": {"base_items": {"accuracy": 1.0}},
                "full": {"donor_items": {"accuracy": 1.0}},
                "key_span": {"donor_items": {"accuracy": .95},
                             "donor_groups": {"accuracy": .9}},
                "complement": {"base_items": {"accuracy": .95},
                               "donor_items": {"accuracy": .05}},
            },
            "native_neuron_scores": {"11": {"donor_items": {"accuracy": .9}}},
        }
    cross = {
        "0_to_1": {"donor_accuracy": .9, "advantage": .6},
        "1_to_0": {"donor_accuracy": .9, "advantage": .6},
    }
    verdict = decide(metrics, cross)
    assert verdict["causal_subspace"] == "causal_subspace_supported"
    assert verdict["cross_seed_alignment"] == "cross_seed_causal_alignment"
    assert verdict["native_axis"] == "native_axis_concentrated"
