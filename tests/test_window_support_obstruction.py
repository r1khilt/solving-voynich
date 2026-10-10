"""A constructive support obstruction, not an empirical recovery experiment."""
from fractions import Fraction as F
from itertools import permutations
from types import SimpleNamespace

import numpy as np

from scripts.audit_window_reading_admit001 import direct_family
from scripts.audit_reading_label_landscape001 import independent_point
from voynich.reading_pair_rewrite import rewrite_pair
from voynich.reading_window import observed_window, window_choices, replace_window
from voynich.source_action_proposal import ReadingEnvironment, ReadingState


def obstruction():
    # Twenty-three distinct pair codes; each has another occurrence in the other record.
    pool = ReadingEnvironment(((0,),), rows=23, glyphs=6).pool
    pairs = tuple(i for i, unit in enumerate(pool) if len(unit) == 2)[:23]
    record = tuple(g for code in pairs for g in pool[code])
    env = ReadingEnvironment((record, record), rows=23, glyphs=6)
    locked = ReadingState(pairs, (len(record), len(record)), (tuple(range(23)),)*2)
    singles = ReadingState(tuple(range(6))+(-1,)*17, locked.offsets, env.records)
    env.validate(locked)
    env.validate(singles)
    return env, locked, singles


def test_all_original_shape_windows_are_identity_despite_better_supported_reading():
    env, locked, singles = obstruction()
    assert len(set(locked.key)) == env.rows == 23
    windows = sum(2*len(r)-1 for r in env.records)
    valid = invalid = 0
    for rank in range(windows):
        window = observed_window(env, rank)
        actual = window_choices(env, locked, *window)
        alternate = direct_family(env, locked, *window)
        if actual is None:
            assert not alternate
            invalid += 1
        else:
            assert actual.replacements == tuple(alternate) and len(alternate) == 1
            for replacement, state in alternate.items():
                assert state == locked == replace_window(env, locked, *window, replacement)
            valid += 1
    assert (windows, valid, invalid) == (182, 46, 136)
    # A positive artificial source favors the reachable-in-principle alternative.
    p = np.array([[1.]*6+[1e-12]*17], dtype=np.float64)
    p /= p.sum()
    t = np.zeros((1, 23), dtype=np.uint32)
    p.flags.writeable = t.flags.writeable = False
    source = SimpleNamespace(probabilities=p, transitions=t, state=lambda _: 0)
    old, better = (F(*independent_point(source, env, state, lambda: None)) for state in (locked, singles))
    assert 0 < old < better


def test_global_distinct_triplet_pair_rewrite_cannot_escape_all_pair_codes():
    env, locked, _ = obstruction()
    for a, b, c in permutations(range(env.rows), 3):
        result = rewrite_pair(env, locked, a, b, c)
        assert result.state == locked and result.branch == 'identity' and result.replacements == 0
