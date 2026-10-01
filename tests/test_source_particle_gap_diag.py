import math

import pytest

from scripts.source_particle_gap_diag import joint_bank_coverage, reading_mass


class SingleRowSource:
    alphabet = ("a",)

    def row(self, state):
        return (1.,)

    def step(self, state, letter):
        return 0


def test_exact_unused_integrated_singleton_and_digram_marginals():
    source = SingleRowSource()
    single = reading_mass(((0,), (0,)), ((0,), (0,)), source, glyphs=1, rho=.25)
    double = reading_mass(((0,), (0,)), ((0, 0), (0, 0)), source, glyphs=1, rho=.25)
    longer = reading_mass(((0, 0), (0, 0)), ((0, 0), (0, 0)), source, glyphs=1, rho=.25)
    assert single["solver"]["complete"] and single["solver"]["solution_count_lower_bound"] == 1
    assert math.exp(single["reading_log_mass"]) == pytest.approx(9/512)
    assert double["reading_log_mass"] == single["reading_log_mass"]
    assert math.exp(longer["reading_log_mass"]-double["reading_log_mass"]) == pytest.approx(9/16)


def test_impossible_shared_key_reading_zero_mass():
    r = reading_mass(((0,), (0,)), ((0,), (1,)), SingleRowSource(), glyphs=2, rho=.25)
    assert r["solver"]["complete"] and r["reading_log_mass"] == -math.inf


def test_joint_coverage_uses_disjoint_leaf_and_stable_large_gap():
    a = joint_bank_coverage(math.log(.3), math.log(.2), True)
    assert math.exp(a["joint_bank_log_coverage_upper"]) == pytest.approx(.4)
    assert a["joint_tv_lower"] == pytest.approx(.6)
    assert joint_bank_coverage(-10, -1000, True)["joint_bank_log_coverage_upper"] == -990
    assert joint_bank_coverage(-10, -1000, False) == {
        "joint_bank_log_coverage_upper": 0., "joint_tv_lower": 0.}
