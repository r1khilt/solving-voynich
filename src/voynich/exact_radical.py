"""Bounded exact MH decisions for positive rational radicals, without floats."""
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

MAX_DEGREE, MAX_INTEGER_BITS = 256, 4_194_304


@dataclass(frozen=True)
class RadicalRatio:
    numerator: int
    denominator: int
    degree: int

    def __post_init__(self):
        if (any(type(x) is not int or x <= 0 for x in (self.numerator, self.denominator, self.degree))
                or self.degree > MAX_DEGREE
                or max(self.numerator.bit_length(), self.denominator.bit_length()) > MAX_INTEGER_BITS):
            raise ValueError('Positive bounded rational radicand and integer degree required')


def _power(integer, exponent):
    if integer.bit_length()*exponent > MAX_INTEGER_BITS:
        raise RuntimeError('Exact radical integer-work cap; no approximation')
    return integer**exponent


def tempered_ratio(target_ratio, correction, degree):
    """(target_ratio**(1/degree))*correction; correction stays OUTSIDE the root."""
    if (not isinstance(target_ratio, Fraction) or not isinstance(correction, Fraction)
            or target_ratio <= 0 or correction <= 0 or type(degree) is not int
            or not 1 <= degree <= MAX_DEGREE):
        raise ValueError('Positive rational target/proposal factors and bounded degree required')
    if max(target_ratio.numerator.bit_length()+correction.numerator.bit_length()*degree,
           target_ratio.denominator.bit_length()+correction.denominator.bit_length()*degree) > MAX_INTEGER_BITS:
        raise RuntimeError('Exact radical product-work cap; no approximation')
    n = target_ratio.numerator*_power(correction.numerator, degree)
    d = target_ratio.denominator*_power(correction.denominator, degree)
    common = math.gcd(n, d)
    return RadicalRatio(n//common, d//common, degree)


def swap_ratio(warm_over_cold_target, cold_degree, warm_degree):
    if (not isinstance(warm_over_cold_target, Fraction) or warm_over_cold_target <= 0
            or any(type(k) is not int for k in (cold_degree, warm_degree))
            or not 1 <= cold_degree < warm_degree <= MAX_DEGREE):
        raise ValueError('Positive target ratio and increasing bounded inverse temperatures required')
    exponent = Fraction(1, cold_degree)-Fraction(1, warm_degree)
    return RadicalRatio(_power(warm_over_cold_target.numerator, exponent.numerator),
                        _power(warm_over_cold_target.denominator, exponent.numerator), exponent.denominator)


def exact_radical_bernoulli(ratio, rng, *, maximum_blocks=16):
    """Refine a dyadic uniform interval; undecided resource caps ABORT the run."""
    if (not isinstance(ratio, RadicalRatio) or type(maximum_blocks) is not int
            or not 1 <= maximum_blocks <= 16 or not isinstance(rng, np.random.Generator)
            or not isinstance(rng.bit_generator, np.random.PCG64)):
        raise ValueError('Bounded radical and full64-bit PCG64 generator required')
    base = 0
    for blocks in range(1, maximum_blocks+1):
        raw = int(rng.bit_generator.random_raw())
        if ratio.numerator >= ratio.denominator:
            return True, blocks
        base = (base << 64)+raw
        bits, k = 64*blocks, ratio.degree
        if max(ratio.numerator.bit_length()+bits*k,
               ratio.denominator.bit_length()+(bits+1)*k) > MAX_INTEGER_BITS:
            raise RuntimeError('Exact radical comparison-work cap; no approximation')
        threshold = ratio.numerator << (bits*k)
        if (base+1)**k*ratio.denominator <= threshold:
            return True, blocks
        if base**k*ratio.denominator >= threshold:
            return False, blocks
    raise RuntimeError('Exact radical raw64-block cap; no approximate decision')
