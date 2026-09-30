"""Complete, deterministic one-move dictionary neighborhood; no answer access."""
from __future__ import annotations

import itertools


def one_move_bank(parent, glyphs, max_unit_length=2):
    parent, glyphs = tuple(parent), tuple(glyphs)
    if (not parent or not glyphs or len(set(glyphs)) != len(glyphs)
            or any(not isinstance(g, str) or len(g) != 1 for g in glyphs)
            or type(max_unit_length) is not int or not 1 <= max_unit_length <= 2
            or any(not isinstance(u, str) or not 1 <= len(u) <= max_unit_length
                   or set(u) - set(glyphs) for u in parent)):
        raise ValueError('Invalid literal key family')
    pool = tuple(''.join(x) for length in range(1, max_unit_length + 1)
                 for x in itertools.product(glyphs, repeat=length))
    keys, seen = [parent], {parent}

    def add(key):
        key = tuple(key)
        if key not in seen:
            seen.add(key)
            keys.append(key)

    for left in range(len(parent)):
        for right in range(left + 1, len(parent)):
            changed = list(parent)
            changed[left], changed[right] = changed[right], changed[left]
            add(changed)
    for row in range(len(parent)):
        for unit in pool:
            changed = list(parent)
            changed[row] = unit
            add(changed)
    return tuple(keys)
