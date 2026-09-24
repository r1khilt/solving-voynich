"""Frozen outcome-blind wrong/deranged donor assignment for TEACH-0015."""

from .teacher14_tasks import digest
from .teacher15_tasks import TransferGroup


def control_permutations(
        groups: list[TransferGroup]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Two distinct, key-mismatched bijections of the frozen group order."""
    if len(groups) != 128:
        raise ValueError("TEACH-0015 finite control requires 128 groups")
    if (len({group.group_id for group in groups}) != 128 or
            len({group.split for group in groups}) != 1):
        raise ValueError("TEACH-0015 finite control split mismatch")
    valid = [offset for offset in range(1, 128) if all(
        groups[(index + offset) % 128].key1 != groups[index].key1
        for index in range(128))]
    if len(valid) < 2:
        raise ValueError("Two global wrong-key donor permutations unavailable")
    deranged_offset = valid[0]
    first = 1 + int(digest([
        "TEACH-0015-wrong", groups[0].split,
        [group.group_id for group in groups]]), 16) % 127
    wrong_offset = next(
        offset for turn in range(127)
        if (offset := 1 + ((first + turn - 1) % 127)) in valid
        and offset != deranged_offset)
    return (
        tuple((index + wrong_offset) % 128 for index in range(128)),
        tuple((index + deranged_offset) % 128 for index in range(128)),
    )


def control_indices(groups: list[TransferGroup], index: int) -> tuple[int, int]:
    if not 0 <= index < len(groups):
        raise ValueError("TEACH-0015 finite control index outside grid")
    wrong, deranged = control_permutations(groups)
    return wrong[index], deranged[index]
