"""Sampled CPU replay numerics and strict prior-screen entry behavior."""

import json
from pathlib import Path

import pytest

from scripts.teacher0015_finite_replay import _compare
from scripts.teacher0016_cross_replay import ABS_TOLERANCE
from scripts.teacher0016_cross_run import _entry, _pack_surface, _surface
from voynich.workspace.teacher14_train import Config, new_model
from voynich.workspace.teacher16_tasks import generate_split


def test_sampled_cpu_crossed_surface_recomputes_logits_and_vectors():
    groups = generate_split("discovery", 4)
    group_index, donor_index = next((a, b) for a in range(4)
                                    for b in range(4) if a != b and
                                    groups[a].key1 != groups[b].key1)
    controls = tuple(donor_index for _ in groups)
    model, _ = new_model(Config(), "latent_rows_answer", 0, "cpu")
    archived = _surface(model, groups, group_index, 3, controls,
                        device="cpu", full_logits=True)
    packed, vectors = _pack_surface(archived)
    fresh = _surface(model, groups, group_index, 3, controls,
                     device="cpu", full_logits=True)
    for attempt, (reference, current) in enumerate(zip(
            archived, fresh, strict=True)):
        assert packed[attempt] == {key: value for key, value in reference.items()
                                   if key != "replacement_vectors"}
        for key, value in current.items():
            if key.endswith("_logits"):
                assert _compare(value, packed[attempt][key], label=key) == 0
        for index, name in enumerate(reference["replacement_vectors"]):
            assert _compare(current["replacement_vectors"][name],
                            vectors[attempt, index].tolist(), label=name) == 0
    tampered = packed[0]["transfer_logits"].copy()
    tampered[64] += ABS_TOLERANCE * 10
    with pytest.raises(ValueError, match="replay differs"):
        _compare(fresh[0]["transfer_logits"], tampered, label="transfer")


def test_causal_entry_refuses_incomplete_prior_screen(tmp_path: Path):
    primary = tmp_path / "primary"
    primary.mkdir()
    (primary / "report.json").write_text(json.dumps({"status": "complete"}))
    with pytest.raises(ValueError, match="prior clean screen incomplete"):
        _entry(primary, tmp_path / "suite", tmp_path / "finite",
               tmp_path / "screen", tmp_path)
