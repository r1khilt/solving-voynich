"""Lossless page splits must preserve duplicate and nonstandard-header bytes."""
import pytest

from scripts.audit_borg_page_review003 import complete_intervals


def test_complete_split_preserves_unsorted_intervals():
    complete_intervals([(22, 40), (10, 22)], 10, 40)


@pytest.mark.parametrize("parts", [
    [(10, 21), (22, 40)],
    [(10, 23), (22, 40)],
    [(10, 40), (10, 40)],
    [(10, 39)],
    [(9, 40)],
    [(10, 41)],
    [(10, 10), (10, 40)],
    [],
])
def test_malformed_split_is_rejected(parts):
    with pytest.raises(ValueError):
        complete_intervals(parts, 10, 40)
