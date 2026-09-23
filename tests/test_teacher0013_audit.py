"""Development-size cross-checks for the independent TEACH-0013 suite auditor."""

import importlib.util
from pathlib import Path


def _load(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preparer = _load("teacher0013_prepare")
auditor = _load("teacher0013_audit")


def test_independent_auditor_validates_native_development_suite():
    payload = preparer.suite_payload(seed=73161, discovery_groups=2, confirmation_groups=2)
    result = auditor.audit_payload(payload)
    assert result == {"groups": 4,
                      "split_counts": {"discovery": 2, "confirmation": 2},
                      "unique_semantic_labels": len(payload["semantic_label_order"])}


def test_independent_auditor_rejects_oracle_and_split_corruption():
    payload = preparer.suite_payload(seed=73162, discovery_groups=1, confirmation_groups=1)
    payload["splits"]["discovery"][0]["base"][0]["answer"] += 1
    try:
        auditor.audit_payload(payload)
    except auditor.AuditError as exc:
        assert "oracle" in str(exc).lower()
    else:
        raise AssertionError("Corrupted answer passed the independent audit")

    payload = preparer.suite_payload(seed=73163, discovery_groups=1, confirmation_groups=1)
    group = payload["splits"]["discovery"].pop()
    group["split"] = "confirmation"
    payload["splits"]["confirmation"].append(group)
    try:
        auditor.audit_payload(payload)
    except auditor.AuditError as exc:
        assert "split" in str(exc).lower()
    else:
        raise AssertionError("Corrupted split passed the independent audit")
