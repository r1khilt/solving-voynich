"""Independent integer-root quantiles for bounded radical Bernoulli decisions."""
from fractions import Fraction as F

import numpy as np
import pytest

from voynich.exact_radical import (RadicalRatio, exact_radical_bernoulli,
                                  swap_ratio, tempered_ratio)
from voynich.reading_regrowth import exact_bernoulli


def integer_root(number, degree, upper):
    lo, hi = 0, upper+1
    while lo+1 < hi:
        middle = (lo+hi)//2
        if middle**degree <= number:
            lo = middle
        else:
            hi = middle
    assert lo**degree <= number < (lo+1)**degree
    return lo


def quantile_decision(ratio, rng, maximum_blocks=16):
    """Binary-search floor(alpha*2**bits), not production's interval inequalities."""
    prefix = 0
    for blocks in range(1, maximum_blocks+1):
        prefix = prefix*2**64+int(rng.bit_generator.random_raw())
        if ratio.numerator >= ratio.denominator:
            return True, blocks
        scale = 2**(64*blocks)
        scaled_n = ratio.numerator*scale**ratio.degree
        floor = integer_root(scaled_n//ratio.denominator, ratio.degree, scale)
        equality = floor**ratio.degree*ratio.denominator == scaled_n
        if prefix < floor:
            return True, blocks
        if prefix > floor or equality:
            return False, blocks
    raise RuntimeError('Independent quantile undecided at fixed cap')


def qualification_decisions():
    count = boundary = 0
    for n, d in ((1, 2), (2, 1), (1, 1), (1, 2**160), (37, 81), (9, 16)):
        for k in (1, 2, 4, 16):
            ratio = RadicalRatio(n, d, k)
            for seed in (96021, 96029):
                a, b = np.random.default_rng(seed), np.random.default_rng(seed)
                for _ in range(128):
                    assert exact_radical_bernoulli(ratio, a) == quantile_decision(ratio, b)
                    assert a.bit_generator.state == b.bit_generator.state
                    count += 1
    # Construct a root strictly INSIDE the first sampled raw64 interval. The
    # second block matters; a one-block cap must abort, not accept/reject.
    for seed in (96031, 96037, 96041, 96043):
        v = int(np.random.default_rng(seed).bit_generator.random_raw())
        for k in (1, 2, 4, 16):
            ratio = RadicalRatio((2*v+1)**k+(k > 1), (2**65)**k, k)
            a, b = np.random.default_rng(seed), np.random.default_rng(seed)
            decision = exact_radical_bernoulli(ratio, a)
            assert decision == quantile_decision(ratio, b) and decision[1] == 2
            assert a.bit_generator.state == b.bit_generator.state
            with pytest.raises(RuntimeError):
                exact_radical_bernoulli(ratio, np.random.default_rng(seed), maximum_blocks=1)
            boundary += 1
    return {'independent_radical_decisions': count, 'two_block_boundary_and_abort_witnesses': boundary}


def test_independent_quantiles_and_boundary_abort():
    assert qualification_decisions() == {'independent_radical_decisions': 6144,
                                         'two_block_boundary_and_abort_witnesses': 16}


def test_degree_one_bit_identity_with_existing_rational_sampler():
    for n, d in ((1, 2), (37, 81), (1, 2**160), (19, 19), (2, 1)):
        a, b = np.random.default_rng(96051), np.random.default_rng(96051)
        for _ in range(50):
            assert exact_radical_bernoulli(RadicalRatio(n, d, 1), a) == exact_bernoulli(n, d, b)
            assert a.bit_generator.state == b.bit_generator.state


def test_corrections_outside_root_and_swap_orientation():
    ratio = tempered_ratio(F(7, 19), F(3, 5), 4)
    assert F(ratio.numerator, ratio.denominator) == F(7, 19)*F(3, 5)**4
    assert F(ratio.numerator, ratio.denominator) != F(7, 19)*F(3, 5)
    swap = swap_ratio(F(16), 1, 4)
    assert swap.degree == 4 and F(swap.numerator, swap.denominator) == 16**3
    reverse = swap_ratio(F(1, 16), 1, 4)
    assert F(reverse.numerator, reverse.denominator) == F(1, 16**3)


def test_inputs_and_integer_work_caps(monkeypatch):
    for args in ((0, 1, 1), (1, 0, 2), (False, 1, 2), (1, 1, 0), (1, 1, 257)):
        with pytest.raises(ValueError):
            RadicalRatio(*args)
    with pytest.raises(ValueError):
        exact_radical_bernoulli(RadicalRatio(1, 2, 2), np.random.Generator(np.random.MT19937(1)))
    with pytest.raises(ValueError):
        swap_ratio(F(2), 4, 1)
    import voynich.exact_radical as module
    monkeypatch.setattr(module, 'MAX_INTEGER_BITS', 128)
    with pytest.raises(RuntimeError):
        tempered_ratio(F(1), F(2**64), 4)
    with pytest.raises(RuntimeError):
        exact_radical_bernoulli(RadicalRatio(1, 2, 16), np.random.default_rng(1))
