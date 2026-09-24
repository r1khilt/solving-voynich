"""Random-weight interface checks; these tests do not score trained models."""

import pytest
import torch

from voynich.workspace.teacher14_train import Config, new_model
from voynich.workspace.teacher15_intervene import capture, patch
from voynich.workspace.teacher16_intervene import (
    VECTOR_NAMES, _cell, evaluate_cross_surface,
)
from voynich.workspace.teacher16_tasks import generate_split


def _groups():
    groups = generate_split("discovery", 4)
    pair = next((left, right) for left in groups for right in groups
                if left.group_id != right.group_id and left.key1 != right.key1)
    return pair


def test_crossed_surface_full_grid_and_numerical_patch_identity():
    group, distinct = _groups()
    model, _ = new_model(Config(), "latent_rows_answer", 0, "cpu")
    rows = evaluate_cross_surface(
        model, group, distinct, distractor=0, marked=True,
        source_order=0, device="cpu", full_logits=True)
    assert [(row["source_g"], row["recipient_g"]) for row in rows] == [
        (a, b) for a in range(3) for b in range(3)]
    assert all(row["recipient_order"] == 1 for row in rows)
    assert all(row["identity_max_abs_logit_error"] < 1e-5 for row in rows)
    assert all(row["final_donor_max_abs_logit_error"] < 1e-5
               for row in rows)
    assert all(row["final_donor_prediction"] == row["source_prediction"]
               for row in rows)
    for index in (0, 4, 8):
        row = rows[index]
        a, b = row["source_g"], row["recipient_g"]
        base = _cell(group, 0, b, 0, True, 1)
        target = _cell(group, 1, b, 0, True, 1)
        donor = _cell(group, 1, a, 0, True, 0)
        same = _cell(group, 1, b, 0, True, 0)
        wrong = _cell(group, 0, a, 0, True, 0)
        base_state = capture(model, base, "cpu")
        donor_state = capture(model, donor, "cpu")
        assert row["base_render_id"] == base.render_id
        assert row["source_render_id"] == donor.render_id
        assert row["same_key_render_id"] == same.render_id
        assert row["wrong_render_id"] == wrong.render_id
        assert row["target_render_id"] == target.render_id
        for name, episode, site in (
            ("identity", base, "query.1"),
            ("transfer", base, "query.1"),
            ("same_key", target, "query.1"),
            ("wrong_key", base, "query.1"),
            ("distinct", base, "query.1"),
            ("random", base, "query.1"),
            ("reverse", target, "query.1"),
            ("final_donor", base, "query.2"),
        ):
            vector_name = {
                "identity": "base_native", "transfer": "source_native",
                "same_key": "same_key_native",
                "wrong_key": "wrong_replacement",
                "distinct": "distinct_replacement",
                "random": "random_replacement",
                "reverse": "wrong_source_native",
                "final_donor": "source_final_native",
            }[name]
            state = torch.tensor(row["replacement_vectors"][vector_name])[None]
            logits = patch(model, episode, state, site=site, device="cpu")
            assert row[f"{name}_logits"] == pytest.approx(
                logits[0].tolist(), abs=1e-5)
        assert row["source_logits"] == pytest.approx(
            donor_state.logits[0].tolist(), abs=1e-5)
        assert row["base_logits"] == pytest.approx(
            base_state.logits[0].tolist(), abs=1e-5)
        vectors = row["replacement_vectors"]
        assert set(vectors) == set(VECTOR_NAMES)
        assert all(len(vector) == 512 for vector in vectors.values())
        base_vector = torch.tensor(vectors["base_native"])
        desired_norm = row["source_delta_norm"]
        assert desired_norm > 0
        for name in ("random_replacement", "wrong_replacement",
                     "distinct_replacement"):
            assert abs((torch.tensor(vectors[name]) - base_vector).norm().item()
                       - desired_norm) < 1e-4
    repeated = evaluate_cross_surface(
        model, group, distinct, distractor=0, marked=True,
        source_order=0, device="cpu", full_logits=False)
    for full, compact in zip(rows, repeated, strict=True):
        assert compact == {key: value for key, value in full.items()
                           if not key.endswith("_logits")}


def test_crossed_surface_rejects_invalid_controls_and_oracle_model():
    group, distinct = _groups()
    model, _ = new_model(Config(), "latent_rows_answer", 0, "cpu")
    with pytest.raises(ValueError, match="surface/control"):
        evaluate_cross_surface(model, group, group, distractor=0,
                               marked=True, source_order=0, device="cpu")
    with pytest.raises(ValueError, match="surface/control"):
        evaluate_cross_surface(model, group, distinct, distractor=0,
                               marked=1, source_order=0, device="cpu")
    oracle, _ = new_model(Config(), "oracle_rows_workspace", 0, "cpu")
    with pytest.raises(ValueError, match="raw candidate-edge"):
        evaluate_cross_surface(oracle, group, distinct, distractor=0,
                               marked=True, source_order=0, device="cpu")
