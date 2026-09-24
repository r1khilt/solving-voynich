"""Adversarial checks on the independent TEACH-0014 suite auditor."""

import copy
import importlib.util
from pathlib import Path

import pytest

from voynich.workspace.teacher14_tasks import evaluation_suite, suite_manifest


def _load_auditor():
    path = Path(__file__).resolve().parents[1] / "scripts/teacher0014_suite_audit.py"
    spec = importlib.util.spec_from_file_location("teacher0014_suite_audit_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest():
    suite = evaluation_suite(74111, size=2)
    return suite_manifest(suite, seed=74111, group_count=2)


def test_independent_audit_accepts_complete_fresh_grid():
    audit = _load_auditor().audit_manifest(_manifest())
    assert audit["audit"] == "pass"
    assert audit["items"] == 74
    assert len(audit["panels"]) == 19
    assert audit["max_rows"] == 32
    assert len(audit["manifest_sha256"]) == 64


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda m: m["panels"]["factorial"].pop(), "wrong item count"),
        (lambda m: m["panels"]["factorial"].__setitem__(
            1, copy.deepcopy(m["panels"]["factorial"][0])), "Factorial answers"),
        (lambda m: m["panels"]["composed_confirm_confirm"][0].__setitem__(
            "answer", 16), "oracle answer"),
        (lambda m: m["panels"]["composed_confirm_confirm"][0][
            "serialized_rows"][0].__setitem__(1, 16), "row/position"),
        (lambda m: m["panels"]["hop_4"][0].__setitem__(
            "graph_partition", "train"), "Split or identity"),
        (lambda m: m["panels"].__setitem__("surprise", []), "panel set"),
    ],
)
def test_independent_audit_rejects_label_grid_and_provenance_tampering(
        mutation, error):
    manifest = _manifest()
    mutation(manifest)
    with pytest.raises(ValueError, match=error):
        _load_auditor().audit_manifest(manifest)
