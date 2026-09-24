"""Outcome-blind checks for the registered fresh rank-shift suite."""

from copy import deepcopy

import pytest

from scripts.teacher0017_suite_audit import _attempt_sets, audit_splits
from voynich.workspace.teacher17_tasks import generate_split, split_manifest


@pytest.fixture(scope="module")
def discovery():
    return split_manifest(generate_split("discovery"), "discovery")


@pytest.fixture(scope="module")
def confirmation():
    return split_manifest(generate_split("confirmation"), "confirmation")


def test_fresh_cross_distractor_suites_shift_relative_rank(
        discovery, confirmation):
    for manifest in (discovery, confirmation):
        checked = _attempt_sets(manifest)["summary"]
        assert checked["groups"] == 128
        assert checked["episodes"] == 8448
        assert checked["cross_distractor_pairs"] == 1024
        assert checked["shifted_rank_fraction"] >= .60
        assert checked["shifted_rank_offdiag_attempts"] == (
            checked["shifted_rank_pairs"] * 6)
        assert checked["relative_rank_null_shifted_hits"] == 0


def test_rank_suite_audit_rejects_changed_visible_program(discovery):
    altered = deepcopy(discovery)
    altered["groups"][0]["cells"][0]["episode"]["tokens"][1] += 1
    with pytest.raises(ValueError):
        _attempt_sets(altered)


def test_rank_suite_refuses_missing_prior_exposure(discovery, confirmation):
    with pytest.raises(ValueError, match="prior exposure"):
        audit_splits(discovery, confirmation, (), (), ())
