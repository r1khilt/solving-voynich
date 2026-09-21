"""Independently check the blind interface, synthetic filter and frozen inference."""

import inspect
import itertools
import json
import time

import numpy as np
import pytest
import torch

from voynich.blind_recovery import (
    ALPHABET, CLUSTER_COUNTS, Partition, adjusted_rand, emission_table, features,
    checked_visible, fit_partition, keyed_hmm, model_config, observed_losses, prepare, read_visible,
    sample_process, surface_baselines, train_predictor,
)
from voynich.model import ModelConfig, VoynichTransformer
from voynich.runtime import digest


@pytest.mark.parametrize("family", ["cycle_null", "branch_null", "rrxor"])
def test_edge_process_normalization_stationarity_and_key_equivariance(family):
    permutation = np.random.default_rng(1).permutation(ALPHABET)
    edge, prior = keyed_hmm(family, np.arange(ALPHABET))
    encoded, encoded_prior = keyed_hmm(family, permutation)
    np.testing.assert_allclose(edge.sum(axis=(0, 2)), 1)
    np.testing.assert_allclose(prior @ edge.sum(axis=0), prior)
    np.testing.assert_allclose(encoded[permutation], edge)
    np.testing.assert_allclose(encoded_prior, prior)


def test_oracle_filter_matches_exhaustive_hidden_path_posterior():
    permutation = np.arange(ALPHABET)
    observed, truth = sample_process("cycle_null", permutation, 1, 6, np.random.default_rng(4), truth_at=2)
    edge, prior = keyed_hmm("cycle_null", permutation)
    posterior = np.zeros(len(prior))
    for path in itertools.product(range(len(prior)), repeat=4):
        weight = prior[path[0]]
        for t in range(3):
            weight *= edge[observed[0, t], path[t], path[t + 1]]
        posterior[path[-1]] += weight
    posterior /= posterior.sum()
    forecast = posterior @ edge.sum(axis=2).T
    np.testing.assert_allclose(truth["oracle"][0, 0], forecast)
    assert truth["map_label"][0] == posterior.argmax()


@pytest.mark.parametrize("family", ["cycle_null", "branch_null", "rrxor", "copy_lag", "iid"])
def test_truth_flag_cannot_change_observed_text_and_forecasts_are_probabilities(family):
    plain, _ = sample_process(family, np.arange(ALPHABET), 12, 35, np.random.default_rng(8))
    labeled, truth = sample_process(family, np.arange(ALPHABET), 12, 35, np.random.default_rng(8), truth_at=31)
    np.testing.assert_array_equal(plain, labeled)
    np.testing.assert_allclose(truth["oracle"].sum(axis=-1), 1)
    assert truth["oracle"].shape == (12, 3, 12)
    if family == "copy_lag":
        assert truth["oracle"][0, 0].argmax() == labeled[0, 24]
        assert truth["label_kind"].endswith("not_complete_hidden_state")


def test_rrxor_constraint_is_not_trivial_next_symbol_copy():
    data, _ = sample_process("rrxor", np.arange(ALPHABET), 100, 63, np.random.default_rng(70))
    bits = data // 6
    # Unknown stationary phase is fixed within a string. One alignment must have
    # every complete triplet satisfying parity, even though individual bits vary.
    assert all(any(np.all((row[offset:-2:3] ^ row[offset + 1:-1:3]) == row[offset + 2::3])
                       for offset in range(3) if len(row[offset:-2:3]) == len(row[offset + 2::3]))
               for row in bits)


def test_blind_partition_uses_only_observed_targets_and_unknown_cluster_count():
    signature = set(inspect.signature(fit_partition).parameters)
    assert signature == {"fit_features", "fit_future", "validation_features", "validation_future", "seed"}
    assert CLUSTER_COUNTS == (1, 2, 4, 8, 16, 32)
    rng = np.random.default_rng(3)
    # Two separated predictive groups. Their hidden labels are never passed.
    x = np.repeat([[-2., -2.], [2., 2.]], 80, axis=0) + rng.normal(0, 0.01, (160, 2))
    future = np.repeat([[1, 2, 3], [8, 9, 10]], 80, axis=0)
    partition, decision = fit_partition(x, future, x + 0.001, future, 100)
    assert decision["selected_count"] == 2
    assert observed_losses(partition.predict(x), future).mean() < 0.3
    scrambled, _ = fit_partition(x, future[rng.permutation(len(future))], x, future, 100)
    assert observed_losses(scrambled.predict(x), future).mean() > 0.8


def test_partition_serialization_and_test_queries_do_not_mutate_fitted_parameters(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(70, 6))
    future = rng.integers(12, size=(70, 3))
    partition, _ = fit_partition(x, future, x[:20], future[:20], 10)
    path = tmp_path / "partition.npz"
    partition.save(path)
    before = path.read_bytes()
    with np.load(path) as saved:
        restored = Partition(**{name: saved[name] for name in saved.files})
    np.testing.assert_array_equal(restored.predict(x), partition.predict(x))
    restored.predict(x * 100)
    assert path.read_bytes() == before


def test_adjusted_rand_is_label_permutation_invariant_and_chance_corrected():
    assert adjusted_rand([0, 0, 1, 1], [9, 9, 4, 4]) == 1
    assert adjusted_rand([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(-0.5)
    assert adjusted_rand([0, 0, 0, 0], [0, 1, 0, 1]) == 0


def test_read_visible_rejects_latent_labels(tmp_path):
    path = tmp_path / "poisoned.npz"
    np.savez(path, x=np.ones((2, 5)), future=np.ones((2, 3)), state=np.ones(2))
    with pytest.raises(ValueError, match="unexpected fields"):
        read_visible(path)


def test_fresh_data_splits_files_and_family_key_holdout(tmp_path):
    root = tmp_path / "synthetic"
    config = {"data_seed": 8201, "train_per_key": 2, "lm_validation_per_key": 2, "context": 40,
              "analysis_context": 32, "fit_contexts": 2, "validation_contexts": 2, "test_contexts": 2}
    manifest = prepare(root, config)
    assert len(manifest["datasets"]) == 20
    assert len([d for d in manifest["datasets"] if d["transfer"] == "novel_family"]) == 4
    assert len([d for d in manifest["datasets"] if d["transfer"] == "unseen_key"]) == 4
    assert len([d for d in manifest["datasets"] if d["transfer"] == "fresh_stream_control"]) == 8
    assert np.load(root / "train_visible.npy").shape == (64, 41)
    for dataset in manifest["datasets"]:
        assert "SEALED-TRUTH" in dataset["pools"]["sealed_truth"]["file"]
        assert "fit" in dataset["pools"] and "validation" in dataset["pools"]
        assert set(read_visible(root / dataset["pools"]["fit"]["file"])) == {"x", "future"}
    assert json.loads((root / "manifest.json").read_text())["config"] == config
    with pytest.raises(FileExistsError):
        prepare(root, config)


def test_feature_extraction_cannot_see_future_and_preserves_model():
    torch.manual_seed(4)
    cfg = ModelConfig(vocab_size=13, pad_id=0, d_model=16, n_layers=1, n_heads=2,
                      d_ff=24, context_length=16, dropout=0)
    model = VoynichTransformer(cfg)
    x = np.random.default_rng(1).integers(12, size=(3, 12), dtype=np.uint8)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    result = features(model, x, "cpu", batch=2)
    logits = model(torch.tensor(x.astype(np.int64) + 1)).logits[:, -1, 1:]
    np.testing.assert_allclose(result["forecast"], logits.softmax(-1).detach().numpy(), atol=1e-7)
    assert result["residual"].shape == (3, 16)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_surface_baselines_are_normalized_and_finite_on_unseen_suffixes():
    rng = np.random.default_rng(33)
    x = rng.integers(12, size=(50, 20))
    future = rng.integers(12, size=(50, 3))
    baseline = surface_baselines(x, future, rng.integers(12, size=(20, 20)))
    for value in baseline.values():
        np.testing.assert_allclose(value.sum(axis=-1), 1)
        assert np.isfinite(value).all() and (value > 0).all()
    np.testing.assert_allclose(emission_table(np.zeros(50, dtype=int), future, 1).sum(axis=-1), 1)


def test_parameter_scaling_is_substantial_without_changing_tokenization():
    small = VoynichTransformer(model_config("compact", 256))
    large = VoynichTransformer(model_config("large", 256))
    assert large.parameter_count > 7 * small.parameter_count
    assert large.config.vocab_size == small.config.vocab_size == 13


def test_checkpoint_roundtrip_and_tiny_train_extract_pipeline(tmp_path):
    rng = np.random.default_rng(77)
    visible = rng.integers(12, size=(8, 9), dtype=np.uint8)
    cfg = ModelConfig(vocab_size=13, pad_id=0, d_model=16, n_layers=1, n_heads=2,
                      d_ff=24, context_length=8, dropout=0)
    destination = tmp_path / "tiny"
    summary = train_predictor(visible, visible[:4], cfg,
                              {"learning_rate": 0.001, "updates": 2, "batch_size": 2, "eval_every": 1},
                              3, destination, "cpu", time.monotonic() + 30)
    checkpoint = torch.load(destination / "best.pt", map_location="cpu", weights_only=True)
    restored = VoynichTransformer(ModelConfig(**checkpoint["config"]))
    restored.load_state_dict(checkpoint["state_dict"])
    assert digest(destination / "best.pt") == summary["best_sha256"]
    extracted = features(restored, visible[:, :6], "cpu", batch=2)
    partition, _ = fit_partition(extracted["residual"], visible[:, 6:9],
                                 extracted["residual"][:4], visible[:4, 6:9], 5)
    np.testing.assert_allclose(partition.predict(extracted["residual"]).sum(axis=-1), 1)


def test_registered_visible_file_digest_detects_drift(tmp_path):
    path = tmp_path / "visible.npz"
    np.savez(path, x=np.ones((3, 5)), future=np.ones((3, 3)))
    record = {"file": path.name, "sha256": digest(path)}
    assert set(checked_visible(tmp_path, record)) == {"x", "future"}
    np.savez(path, x=np.zeros((3, 5)), future=np.ones((3, 3)))
    with pytest.raises(AssertionError, match="changed"):
        checked_visible(tmp_path, record)
