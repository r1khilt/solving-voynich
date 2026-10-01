"""Compiled sums versus independent full-key/string and Python-state laws."""
import itertools
import math

import numpy as np
import pytest

from tests.test_source_state_lattice import full_reference, source
from voynich.native_source_state import NativeStateLattice, build_source_state_native
from voynich.source_state_lattice import search_state_lattice


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    return build_source_state_native(tmp_path_factory.mktemp("source-state-build"))


@pytest.mark.parametrize("schedule", ["balanced", "sequential"])
@pytest.mark.parametrize("merge", [False, True])
def test_all36_native_pairs_match_full_reference_and_python(build, schedule, merge):
    native = NativeStateLattice(*source(), build)
    observations = [(g,) for g in range(2)]+list(itertools.product(range(2), repeat=2))
    for cipher in itertools.product(observations, repeat=2):
        expected, maxima = full_reference(cipher)
        actual = native.search(cipher, glyphs=2, rho=.25, width=100_000, merge=merge, schedule=schedule)
        reference = search_state_lattice(cipher, *source(), glyphs=2, rho=.25, width=100_000, merge=merge, schedule=schedule)
        assert actual["complete_search"] and not actual["terminal_output_truncated"]
        terminals = {t["used_key"]: t for t in actual["terminals"]}
        assert terminals.keys() == expected.keys()
        for key, mass in expected.items():
            assert math.exp(terminals[key]["log_mass"]) == pytest.approx(float(mass), abs=1e-13)
            assert math.exp(terminals[key]["best_leaf_log_mass"]) == pytest.approx(float(maxima[key]), abs=1e-13)
        for field in ("found_log_mass", "expanded", "generated", "merged_arrivals", "maximum_active_states", "maximum_unpruned_layer"):
            assert actual[field] == pytest.approx(reference[field], abs=1e-13)


@pytest.mark.parametrize("guidance", ["none", "iid"])
def test_native_beam_and_whole_layer_caps_bound_exact_evidence(build, guidance):
    native = NativeStateLattice(*source(), build)
    cipher = ((0, 1, 0), (1, 0, 1))
    expected, _ = full_reference(cipher)
    evidence = float(sum(expected.values()))
    for config in ({"width": 2}, {"width": 1000, "max_expanded": 1},
                   {"width": 1000, "max_generated": 1}, {"width": 1000, "max_seconds": 1e-9}):
        result = native.search(cipher, glyphs=2, rho=.25, guidance=guidance, **config)
        assert not result["complete_search"]
        assert math.exp(result["found_log_mass"]) <= evidence+1e-13
        assert math.exp(result["evidence_log_upper"]) >= evidence-1e-13
        for terminal in result["terminals"]:
            assert math.exp(terminal["log_mass"]) <= float(expected[terminal["used_key"]])+1e-13


def test_native_iid_guide_changes_order_only_without_pruning(build):
    native = NativeStateLattice(*source(), build)
    cipher = ((0, 0), (0, 1))
    expected, _ = full_reference(cipher)
    result = native.search(cipher, glyphs=2, rho=.25, guidance="iid", width=100_000)
    assert result["complete_search"] and result["guide_tables_built"] > 0
    assert math.exp(result["found_log_mass"]) == pytest.approx(float(sum(expected.values())), abs=1e-13)


def test_native_iid_histories_and_different_letter_length_paths(build):
    probabilities, transitions = np.array([[1/3, 2/3]]), np.zeros((1, 2), dtype=np.uint32)
    native = NativeStateLattice(probabilities, transitions, build)
    cipher = ((0,)*5,)
    expected, maxima = full_reference(cipher, iid=True)
    result = native.search(cipher, glyphs=2, rho=.25, width=100_000)
    assert result["complete_search"] and result["merged_arrivals"] > 0
    terminals = {t["used_key"]: t for t in result["terminals"]}
    for key, mass in expected.items():
        assert math.exp(terminals[key]["log_mass"]) == pytest.approx(float(mass), abs=1e-13)
        assert math.exp(terminals[key]["best_leaf_log_mass"]) == pytest.approx(float(maxima[key]), abs=1e-13)


def test_native_zero_support_output_caps_and_allocation_failure(build):
    null = NativeStateLattice(*source(zero=True), build).search(((0, 1, 0),), glyphs=2, rho=.25, width=100_000)
    assert null["complete_search"] and null["found_log_mass"] == null["evidence_log_upper"] == -math.inf
    native = NativeStateLattice(*source(), build)
    result = native.search(((0, 0),), glyphs=2, rho=.25, width=100_000, max_terminals=1)
    assert result["complete_search"] and result["terminal_output_truncated"]
    assert result["returned_terminal_log_mass"] < result["found_log_mass"]
    with pytest.raises(MemoryError):
        native.search(((0, 0),), glyphs=2, rho=.25, max_active=1)
    with pytest.raises(RuntimeError, match="guide table"):
        native.search(((0, 0),), glyphs=2, rho=.25, guidance="iid", guide_max_tables=1)


def test_native_pins_dimensions_and_configuration(build):
    with pytest.raises(ValueError, match="changed"):
        NativeStateLattice(*source(), dict(build, library_sha256="0"*64))
    native = NativeStateLattice(*source(), build)
    for observed in ((), ((),), ((True,),), ((2,),)):
        with pytest.raises(ValueError):
            native.search(observed, glyphs=2)
    for config in ({"max_seconds": math.inf}, {"width": True}, {"guide_prewidth": 0}, {"merge": 1}):
        with pytest.raises(ValueError):
            native.search(((0,),), glyphs=2, **config)
