"""A pure physical pointer passes transfer without a key-symbol state."""

from scripts.teacher0015_row_pointer_audit import expected_split
from scripts.teacher0015_row_pointer_counterexample import construct_split
from voynich.workspace.teacher15_tasks import generate_split, split_manifest


def test_full_visible_pointer_counterexample_has_independent_equivalent_score():
    groups = generate_split("discovery")
    rows, structure = construct_split(groups, "discovery")
    independent = expected_split(split_manifest(
        groups, "discovery")["groups"], "discovery")
    assert len(rows) == 3072
    assert structure["same_slot_surfaces"] == 1024
    assert structure["prediction_sha256"] == independent["prediction_sha256"]
    assert independent["scores"]["transfer"]["correct"] == 3072
    assert independent["scores"]["triples"]["correct"] == 1024
    assert independent["scores"]["changed_g_non_injection"][
        "correct"] == 2048
    assert independent["passes_behavioral_thresholds_only"]
