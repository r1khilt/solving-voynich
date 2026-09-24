"""Independent archive checks on a random-weight crossed surface."""

from copy import deepcopy

import numpy as np
import pytest

from scripts.teacher0016_cross_audit import VECTOR_NAMES, audit_cross_surface
from voynich.workspace.teacher14_train import Config, new_model
from voynich.workspace.teacher16_intervene import evaluate_cross_surface
from voynich.workspace.teacher16_tasks import generate_split, split_manifest


def _fixture():
    groups = generate_split("discovery", 4)
    first, second = next((left, right) for left in groups for right in groups
                         if left.group_id != right.group_id and
                         left.key1 != right.key1)
    manifest = split_manifest([first, second], "discovery")
    model, _ = new_model(Config(), "latent_rows_answer", 0, "cpu")
    complete = evaluate_cross_surface(
        model, first, second, distractor=0, marked=True,
        source_order=0, device="cpu", full_logits=True)
    vectors = np.asarray([
        [row["replacement_vectors"][name] for name in VECTOR_NAMES]
        for row in complete], dtype=np.float32)
    rows = [{key: value for key, value in row.items()
             if key != "replacement_vectors"} for row in complete]
    return manifest["groups"], rows, vectors


def test_crossed_auditor_accepts_actual_interface_and_rejects_tampering():
    groups, rows, vectors = _fixture()
    assert audit_cross_surface(
        groups[0], groups[1], rows, vectors, distractor=0,
        marked=True, source_order=0, full_logits=True)["audit"] == "pass"
    bad_rows = deepcopy(rows)
    bad_rows[4]["source_slot"] += 1
    with pytest.raises(ValueError, match="visible"):
        audit_cross_surface(groups[0], groups[1], bad_rows, vectors,
                            distractor=0, marked=True, source_order=0,
                            full_logits=True)
    bad_vectors = vectors.copy()
    bad_vectors[4, VECTOR_NAMES.index("wrong_replacement"), 0] += 1
    with pytest.raises(ValueError, match="wrong_replacement"):
        audit_cross_surface(groups[0], groups[1], rows, bad_vectors,
                            distractor=0, marked=True, source_order=0,
                            full_logits=True)
    bad_rows = deepcopy(rows)
    bad_rows[4]["transfer_prediction"] = 16
    with pytest.raises(ValueError, match="argmax/logits"):
        audit_cross_surface(groups[0], groups[1], bad_rows, vectors,
                            distractor=0, marked=True, source_order=0,
                            full_logits=True)


def test_crossed_auditor_rejects_wrong_archive_shape():
    groups, rows, vectors = _fixture()
    with pytest.raises(ValueError, match="grid"):
        audit_cross_surface(groups[0], groups[1], rows, vectors[:8],
                            distractor=0, marked=True, source_order=0,
                            full_logits=True)
