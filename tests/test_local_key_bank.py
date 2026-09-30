import itertools

import pytest

from voynich.local_key_bank import one_move_bank


@pytest.mark.parametrize('parent', [('x', 'y', 'xx'), ('x', 'x', 'xx'), ('xx',)])
def test_bank_equals_literal_complete_one_move_set(parent):
    pool = tuple(''.join(x) for n in (1, 2) for x in itertools.product('xy', repeat=n))
    expected = {parent}
    for key in itertools.product(pool, repeat=len(parent)):
        changed = [i for i, (a, b) in enumerate(zip(key, parent)) if a != b]
        if len(changed) == 1 or (len(changed) == 2
                                and key[changed[0]] == parent[changed[1]]
                                and key[changed[1]] == parent[changed[0]]):
            expected.add(key)
    bank = one_move_bank(parent, 'xy')
    assert bank[0] == parent and len(bank) == len(set(bank))
    assert set(bank) == expected
    assert bank == one_move_bank(parent, 'xy')


def test_all_23_rows_and_42_units_are_eligible_without_gold_filters():
    units = tuple('ABCDEF') + tuple(''.join(p) for p in itertools.product('ABCDEF', repeat=2))[:17]
    bank = one_move_bank(units, 'ABCDEF')
    assert len(bank) == 1 + 23 * 22 // 2 + 23 * 41 == 1197
    for row in range(23):
        for unit in ('F', 'FF', 'AA'):
            key = list(units)
            key[row] = unit
            assert tuple(key) in bank


@pytest.mark.parametrize('parent,glyphs,length', [
    ((), 'xy', 2), (('x',), 'xx', 2), (('z',), 'xy', 2),
    (('xxx',), 'xy', 2), (('x',), 'xy', True), (('x',), 'xy', 3),
])
def test_invalid_families_fail(parent, glyphs, length):
    with pytest.raises(ValueError):
        one_move_bank(parent, glyphs, length)
