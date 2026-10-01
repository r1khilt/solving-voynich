"""Exhaustive full-key/source strings, not a lattice-state reimplementation."""
import itertools
import math
from collections import defaultdict
from fractions import Fraction as F

import numpy as np
import pytest

from voynich.source_state_lattice import (
    State, remaining_log_bound, search_state_lattice, selected_record,
)


def full_reference(cipher, iid=False, zero=False):
    pool = tuple((g,) for g in range(2))+tuple(itertools.product(range(2), repeat=2))
    rows = [(F(1, 3), F(2, 3)), (F(3, 4), F(1, 4)), (F(1, 5), F(4, 5))]
    if zero:
        rows = [(F(1), F(0))]*3
    if iid:
        rows = [rows[0]]*3
    probability = {}
    for n in range(1, max(map(len, cipher))+1):
        for text in itertools.product(range(2), repeat=n):
            weight, context = F(1, 4)*F(3, 4)**n, 0
            for row in text:
                weight *= rows[context][row]
                context = row+1
            probability[text] = weight
    masses, maxima = defaultdict(F), defaultdict(F)
    for indices in itertools.product(range(6), repeat=2):
        key = tuple(pool[index] for index in indices)
        choices = [[text for text in probability if tuple(g for row in text for g in key[row]) == observed] for observed in cipher]
        for texts in itertools.product(*choices):
            used = {row for text in texts for row in text}
            partial = tuple(indices[row] if row in used else -1 for row in range(2))
            # Sum full-key priors; maximize using the USED-key prior, matching
            # the lattice's independently integrated unused rows.
            likelihood = math.prod(probability[text] for text in texts)
            masses[partial] += F(1, 36)*likelihood
            maxima[partial] = max(maxima[partial], likelihood/F(6)**len(used))
    return {k:v for k,v in masses.items() if v}, maxima


def source(iid=False, zero=False):
    probabilities = np.array([[1/3, 2/3], [3/4, 1/4], [1/5, 4/5]], dtype=np.float64)
    if iid:
        probabilities[:] = probabilities[0]
    if zero:
        probabilities[:] = [1., 0.]
    return probabilities, np.tile(np.array([1, 2], dtype=np.uint32), (3, 1))


@pytest.mark.parametrize("schedule", ["sequential", "balanced"])
@pytest.mark.parametrize("merge", [False, True])
def test_all36_short_cipher_pairs_match_full_key_reference(schedule, merge):
    observations = [(g,) for g in range(2)]+list(itertools.product(range(2), repeat=2))
    for cipher in itertools.product(observations, repeat=2):
        expected, maxima = full_reference(cipher)
        result = search_state_lattice(cipher, *source(), glyphs=2, rho=.25, width=100_000,
            schedule=schedule, merge=merge)
        assert result["complete_search"] and not result["terminal_output_truncated"]
        actual = {tuple(t["used_key"]): t for t in result["terminals"]}
        assert actual.keys() == expected.keys()
        assert math.exp(result["found_log_mass"]) == pytest.approx(float(sum(expected.values())), abs=1e-13)
        for key, mass in expected.items():
            assert math.exp(actual[key]["log_mass"]) == pytest.approx(float(mass), abs=1e-13)
            assert math.exp(actual[key]["best_leaf_log_mass"]) == pytest.approx(float(maxima[key]), abs=1e-13)


def test_iid_histories_different_letter_lengths_merge_before_final_layer():
    cipher = ((0,)*5,)
    probabilities = np.array([[1/3, 2/3]], dtype=np.float64)
    transitions = np.zeros((1, 2), dtype=np.uint32)
    expected, maxima = full_reference(cipher, iid=True)
    merged = search_state_lattice(cipher, probabilities, transitions, glyphs=2, rho=.25, width=100_000)
    unmerged = search_state_lattice(cipher, probabilities, transitions, glyphs=2, rho=.25, width=100_000, merge=False)
    assert merged["merged_arrivals"] > 0 and merged["expanded"] < unmerged["expanded"]
    assert merged["found_log_mass"] == pytest.approx(unmerged["found_log_mass"], abs=1e-13)
    actual = {tuple(t["used_key"]): t for t in merged["terminals"]}
    for key, mass in expected.items():
        assert math.exp(actual[key]["log_mass"]) == pytest.approx(float(mass), abs=1e-13)
        assert math.exp(actual[key]["best_leaf_log_mass"]) == pytest.approx(float(maxima[key]), abs=1e-13)
    # Same partial dictionary supports both three- and four-letter sources.
    assert actual[(0, 2)]["log_mass"] > actual[(0, 2)]["best_leaf_log_mass"]


@pytest.mark.parametrize("guidance", ["none", "iid"])
def test_pruning_and_whole_layer_caps_have_valid_evidence_bounds(guidance):
    cipher = ((0, 1, 0), (1, 0, 1))
    expected, _ = full_reference(cipher)
    evidence = float(sum(expected.values()))
    for limits in ({"width": 2}, {"width": 1000, "max_expanded": 1},
                   {"width": 1000, "max_generated": 1}):
        result = search_state_lattice(cipher, *source(), glyphs=2, rho=.25, guidance=guidance, **limits)
        assert not result["complete_search"]
        assert math.exp(result["found_log_mass"]) <= evidence+1e-13
        assert math.exp(result["evidence_log_upper"]) >= evidence-1e-13
        for terminal in result["terminals"]:
            key = tuple(terminal["used_key"])
            assert math.exp(terminal["log_mass"]) <= float(expected[key])+1e-13


def test_guide_ordering_retains_exact_law_when_unpruned_and_zero_null():
    cipher = ((0, 0), (0, 1))
    expected, _ = full_reference(cipher)
    result = search_state_lattice(cipher, *source(), glyphs=2, rho=.25, guidance="iid", width=100_000)
    assert result["complete_search"] and result["guide_tables_built"] > 0
    assert math.exp(result["found_log_mass"]) == pytest.approx(float(sum(expected.values())), abs=1e-13)
    null = search_state_lattice(((0, 1, 0),), *source(zero=True), glyphs=2, rho=.25, width=100_000)
    assert null["complete_search"] and not null["terminals"]
    assert null["found_log_mass"] == null["evidence_log_upper"] == -math.inf


def test_output_truncation_keeps_evidence_and_allocation_cap_is_not_success():
    result = search_state_lattice(((0, 0),), *source(), glyphs=2, rho=.25, width=100_000, max_terminals=1)
    assert result["complete_search"] and result["terminal_output_truncated"]
    assert result["returned_terminal_log_mass"] < result["found_log_mass"]
    with pytest.raises(MemoryError):
        search_state_lattice(((0, 0),), *source(), glyphs=2, rho=.25, max_active=1)


def test_scheduler_integer_ties_and_remaining_geometric_bound():
    assert selected_record((1, 2), (3, 6), "balanced") == 0
    assert selected_record((3, 2), (3, 6), "balanced") == 1
    assert selected_record((3, 6), (3, 6), "balanced") is None
    state = State((1, 2), (0, 0), (-1, -1))
    expected = sum(.25*.75**n for n in range(1, 3))*sum(.25*.75**n for n in range(2, 5))
    assert math.exp(remaining_log_bound(state, (3, 6), .25)) == pytest.approx(expected)
