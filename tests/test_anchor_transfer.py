"""Finite controls for the ANCHOR-0001 registered readout."""

from voynich.anchor_transfer import edit_distance, rank_fraction, similarity


def test_edit_distance_and_similarity() -> None:
    assert edit_distance("koldarod", "koldarod") == 0
    assert edit_distance("koldarod", "koldaro") == 1
    assert edit_distance("abc", "xyz") == 3
    assert similarity("abc", {"abc", "xyz"}) == (1.0, "abc")


def test_rank_fraction_ties_and_direction() -> None:
    assert rank_fraction([1.0, 0.5, 0.0], 0) == (1.0, 1.0)
    assert rank_fraction([1.0, 0.5, 0.0], 2) == (0.0, 3.0)
    assert rank_fraction([0.5, 0.5, 0.5], 1) == (0.5, 2.0)
