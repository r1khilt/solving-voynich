"""Independent parser scorer fixtures on a structurally valid development suite."""

from copy import deepcopy

import pytest

from scripts.teacher0014_parser_audit import RAW_ARMS, audit_parser
from scripts.teacher0014_suite_audit import SYMBOL_START, audit_manifest
from voynich.workspace.teacher14_tasks import evaluation_suite, suite_manifest


@pytest.fixture(scope="module")
def perfect_fixture():
    manifest = suite_manifest(evaluation_suite(74117, 1), seed=74117,
                              group_count=1)
    sha = audit_manifest(manifest)["manifest_sha256"]
    panels = {}
    for name, episodes in manifest["panels"].items():
        panels[name] = []
        for episode in episodes:
            ordinary = [value for value in episode["tokens"][1:-3]
                        if value >= SYMBOL_START]
            panels[name].append({"render_id": episode["render_id"],
                                 "logits": [1.0 if index % 2 == 0 else -1.0
                                            for index in range(len(ordinary) - 1)]})
    archive = {"experiment": "TEACH-0014", "manifest_sha256": sha,
               "threshold_logit": 0.0,
               "runs": {arm: {"0": deepcopy(panels), "1": deepcopy(panels)}
                        for arm in RAW_ARMS}}
    return manifest, archive


def test_perfect_visible_edges_pass_symbolic_solver(perfect_fixture):
    manifest, archive = perfect_fixture
    result = audit_parser(manifest, archive)
    assert result["answer_only_parser_qualified_both_seeds"]
    row = result["runs"]["latent_rows_answer"]["0"]
    assert row["rates"]["marked"]["complete_table_exact"]["accuracy"] == 1
    assert row["rates"]["marker_free"]["complete_table_exact"]["accuracy"] == 1
    assert row["rates"]["all"]["symbolic_factorial_groups"]["accuracy"] == 1


def test_wrong_edges_fail_and_archive_tampering_is_rejected(perfect_fixture):
    manifest, archive = perfect_fixture
    wrong = deepcopy(archive)
    for panel in wrong["runs"]["latent_rows_answer"]["0"].values():
        for row in panel:
            row["logits"] = [-1.0] * len(row["logits"])
    result = audit_parser(manifest, wrong)
    assert not result["answer_only_parser_qualified_both_seeds"]
    assert result["runs"]["latent_rows_answer"]["0"]["rates"]["all"][
        "signal_row_recall"]["accuracy"] == 0
    missing = deepcopy(archive)
    missing["runs"]["latent_rows_answer"]["0"]["factorial"].pop()
    with pytest.raises(ValueError, match="item coverage"):
        audit_parser(manifest, missing)
    altered = deepcopy(archive)
    altered["runs"]["latent_rows_answer"]["0"]["factorial"][0][
        "logits"][0] = float("nan")
    with pytest.raises(ValueError, match="invalid gate logits"):
        audit_parser(manifest, altered)
