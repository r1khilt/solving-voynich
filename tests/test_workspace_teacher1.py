"""Generator, split and numerical checks for TEACH-0001 (no training run)."""

import json
import random

import pytest

from voynich.workspace.teacher1_tasks import (
    OBJECTS, evaluation_suite, family_signatures, sample_episode, sample_factorial,
    symbolic_oracle, training_batch,
)
from voynich.workspace.teacher1_train import (
    TeacherConfig, _model, _provenance, _record_progress, decide, numerical_qualification, wilson_95,
)


def test_streaming_partitions_oracle_and_determinism():
    first = training_batch(99001, 256)
    second = training_batch(99001, 256)
    assert first == second
    assert all(ep.f_partition == ep.g_partition == "train" for ep in first[0])
    assert all(symbolic_oracle(ep.tokens) == ep.answer for ep in first[0])
    null, labels = training_batch(99001, 256, null_labels=True)
    assert null == first[0]
    assert all(label in OBJECTS for label in labels)
    assert any(label != ep.answer for ep, label in zip(null, labels, strict=True)
               if ep.task == "composed")
    assert all(label == ep.answer for ep, label in zip(null, labels, strict=True)
               if ep.task != "composed")


@pytest.mark.parametrize("f_part,g_part", [
    ("train", "train"), ("holdout", "train"),
    ("train", "holdout"), ("holdout", "holdout"),
])
def test_all_crossed_partitions_are_reachable(f_part, g_part):
    episode = sample_episode(random.Random(177), f_partition=f_part, g_partition=g_part,
                             task="composed")
    assert (episode.f_partition, episode.g_partition) == (f_part, g_part)
    assert symbolic_oracle(episode.tokens) == episode.answer


def test_factorial_requires_two_tables_and_keeps_family_holdout():
    quartet = sample_factorial(random.Random(401), partition="holdout")
    assert len(quartet) == 4
    assert all(ep.f_partition == ep.g_partition == "holdout" for ep in quartet)
    assert len({ep.answer for ep in quartet}) == 4
    assert quartet[0].tokens[6:] == quartet[1].tokens[6:]
    assert quartet[2].tokens[6:] == quartet[3].tokens[6:]
    assert quartet[0].tokens[:6] == quartet[2].tokens[:6]
    assert quartet[1].tokens[:6] == quartet[3].tokens[:6]
    assert all(symbolic_oracle(ep.tokens) == ep.answer for ep in quartet)


def test_eval_suite_and_registered_thresholds():
    suite = evaluation_suite(61103, 16)
    assert len(suite["factorial"]) == 32
    assert all(len(rows) == 16 for name, rows in suite.items() if name != "factorial")
    assert all(symbolic_oracle(ep.tokens) == ep.answer for rows in suite.values() for ep in rows)
    train_f, train_g = (set(), set())
    for seed in range(99001, 99033):
        episodes, _ = training_batch(seed, 64)
        for ep in episodes:
            f_family, g_family = family_signatures(ep)
            train_f.add(f_family)
            train_g.add(g_family)
    heldout = suite["composed_holdout_holdout"] + suite["factorial"]
    assert all(family_signatures(ep)[0] not in train_f for ep in heldout)
    assert all(family_signatures(ep)[1] not in train_g for ep in heldout)
    p = {name: {"accuracy": 0.99} for name in suite}
    p["factorial"]["quartet_accuracy"] = 0.75
    n = {name: {"accuracy": 0.125} for name in suite}
    assert decide(p, n)["verdict"] == "qualified"
    p["composed_holdout_holdout"]["accuracy"] = 0.80
    assert decide(p, n)["verdict"] == "not_qualified"


def test_cpu_numerical_gate_and_resource_ceiling():
    assert _model().parameter_count == 664_704
    result = numerical_qualification("cpu")
    assert result["finite_gradients"]
    assert result["max_cached_logit_error"] <= 0.002
    TeacherConfig().validate()
    with pytest.raises(ValueError):
        TeacherConfig(max_seconds=3601).validate()
    assert wilson_95(0, 256)[0] == 0.0
    assert wilson_95(256, 256)[1] == 1.0


def test_progress_and_source_provenance(tmp_path):
    provenance = _provenance(TeacherConfig())
    assert len(provenance["source_git_head"]) == 40
    assert len(provenance["registration_sha256"]) == 64
    assert len(provenance["config_sha256"]) == 64
    status = {"status": "running", **provenance}
    resources = {"peak_sampled_mps_allocated_bytes": 1234}
    status_path = tmp_path / "status.json"
    _record_progress(status, status_path, resources, "primary", 250, 12.5, 0.5, 0.6, 1234)
    written = json.loads(status_path.read_text())
    assert written["progress"] == {
        "arm": "primary", "step": 250, "elapsed_seconds": 12.5,
        "latest_loss": 0.5, "last_100_loss": 0.6,
        "peak_sampled_mps_allocated_bytes": 1234,
    }
    assert written["source_git_head"] == provenance["source_git_head"]
