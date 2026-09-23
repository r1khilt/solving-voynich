"""Tests for deterministic TEACH-0013 suite materialization."""

import hashlib
import importlib.util
import json
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "teacher0013_prepare", Path(__file__).parents[1] / "scripts/teacher0013_prepare.py")
assert _SPEC is not None and _SPEC.loader is not None
preparer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(preparer)


def test_suite_payload_is_deterministic_and_has_exhaustive_semantic_maps():
    first = preparer.suite_payload(seed=73, discovery_groups=2, confirmation_groups=2)
    second = preparer.suite_payload(seed=73, discovery_groups=2, confirmation_groups=2)
    assert hashlib.sha256(preparer.stable_json(first)).digest() == \
        hashlib.sha256(preparer.stable_json(second)).digest()
    assert {name: len(groups) for name, groups in first["splits"].items()} == {
        "discovery": 2, "confirmation": 2,
    }
    for groups in first["splits"].values():
        for group in groups:
            assert set(group["semantic_layouts"]) == set(preparer.VARIANTS)
            for name, layouts in group["semantic_layouts"].items():
                episodes = group[name] if isinstance(group[name], tuple) else (group[name],)
                assert len(layouts) == len(episodes)
                for layout, episode in zip(layouts, episodes, strict=True):
                    assert len(layout["roles"]) == len(episode["tokens"])
                    assert len(set(layout["labels"])) == len(layout["labels"])


def test_stable_payload_contains_no_nonfinite_or_unserializable_values():
    payload = preparer.suite_payload(seed=74, discovery_groups=1, confirmation_groups=1)
    parsed = json.loads(preparer.stable_json(payload))
    assert parsed["experiment"] == "TEACH-0013"
    assert parsed["namespace"] == "TEACH-0013-v1"
