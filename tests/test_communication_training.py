import json

import pytest
import torch

from voynich.communication.training import (
    load_checkpoint,
    read_configuration,
    train,
    validation_loss,
    TrainingConfig,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def configuration(path):
    data = {
        "schema_version": 1,
        "layout": {"alphabet_size": 24, "max_observation": 96, "max_anchors": 2},
        "model": {"width": 16, "layers": 1, "heads": 2, "dropout": 0.0},
        "training": {
            "steps": 4,
            "batch_size": 2,
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "eval_interval": 2,
            "validation_examples": 2,
            "seed": 7301,
            "max_seconds": 30,
            "threads": 1,
            "anchor_count": 2,
        },
    }
    path.write_text(json.dumps(data))
    return path


def test_checkpoint_roundtrip_resume_matches_uninterrupted(tmp_path):
    config = configuration(tmp_path / "config.json")
    full = tmp_path / "full"
    partial = tmp_path / "partial"
    train(config, full, device="cpu")
    a = train(config, partial, device="cpu", stop_after=2)
    assert a["status"] == "explicit_checkpoint_stop"
    b = train(config, partial, device="cpu", resume=partial / "last.pt")
    assert b["steps"] == 4
    m1, l1, c1 = load_checkpoint(full / "last.pt")
    m2, l2, c2 = load_checkpoint(partial / "last.pt")
    assert l1 == l2
    for key, value in m1.state_dict().items():
        assert torch.equal(value, m2.state_dict()[key]), key
    assert c1["best_validation_loss"] == c2["best_validation_loss"]
    assert torch.equal(c1["sampler_rng"], c2["sampler_rng"])
    assert c1["initial_validation"] == c2["initial_validation"]


def test_overwrite_and_unfrozen_resume_rejected(tmp_path):
    config = configuration(tmp_path / "config.json")
    folder = tmp_path / "run"
    train(config, folder, device="cpu", stop_after=2)
    with pytest.raises(ValueError):
        train(config, folder, device="cpu")
    raw = json.loads(config.read_text())
    raw["training"]["seed"] = 999
    config.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="identical"):
        train(config, folder, device="cpu", resume=folder / "last.pt")


def test_final_test_is_not_a_validation_option(tmp_path):
    config = configuration(tmp_path / "config.json")
    folder = tmp_path / "run"
    train(config, folder, device="cpu", stop_after=2)
    model, layout, _ = load_checkpoint(folder / "last.pt")
    with pytest.raises(ValueError, match="Final synthetic"):
        validation_loss(model, layout, TrainingConfig(), "cpu", split="test")


def test_nonfinite_or_unbounded_configuration_refused(tmp_path):
    for kwargs in (
        {"steps": 0},
        {"max_seconds": float("inf")},
        {"learning_rate": float("nan")},
        {"batch_size": False},
        {"steps": 10**9},
    ):
        with pytest.raises(ValueError):
            TrainingConfig(**kwargs)
    path = configuration(tmp_path / "config.json")
    raw = json.loads(path.read_text())
    raw["model"]["unsupported"] = True
    path.write_text(json.dumps(raw))
    with pytest.raises(TypeError):
        read_configuration(path)


def test_configuration_rejects_boolean_numbers_and_incompatible_slot_capacity(tmp_path):
    for keyword in ("learning_rate", "max_seconds", "gradient_clip", "weight_decay"):
        with pytest.raises(ValueError):
            TrainingConfig(**{keyword: True})
    with pytest.raises(ValueError):
        TrainingConfig(weight_decay="bad")
    for section, key, value in [("model", "type_vocab_size", 5), ("layout", "alphabet_size", 18)]:
        path = configuration(tmp_path / "bad.json")
        raw = json.loads(path.read_text())
        raw[section][key] = value
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="requires"):
            read_configuration(path)
