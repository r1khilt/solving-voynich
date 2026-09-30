import math

import pytest

from voynich.finite_state_channel_fit import CodingContext
from voynich.higher_order_unit_channel import estimate_source
from voynich.higher_order_unit_refine import refine_units
from voynich import higher_order_unit_refine as module


def fixture():
    source = estimate_source(["aaaabaaab"], ("a", "b"), 3, 1.)
    context = CodingContext(("a", "b"), ("X", "Y"), 8, 2, 2, 3, stop_probability=.2)
    return source, context


def independent_score(source, units, records, context):
    import itertools
    from scripts.audit_blind_channel_dev004 import literal_model_bits
    masses = []
    for record in records:
        total = 0.
        for length in range(len(record) + 1):
            for chars in itertools.product(source.alphabet, repeat=length):
                if "".join(units[source.alphabet.index(char)] for char in chars) != record:
                    continue
                probability, history = .2, ""
                for char in chars:
                    probability *= .8 * source.probabilities[history][char]
                    history = (history + char)[-source.order:]
                total += probability
        if not total:
            return None
        masses.append(math.log(total))
    from dataclasses import asdict
    return literal_model_bits(tuple(units), asdict(context)) - sum(masses) / math.log(2)


def test_every_neighbor_and_selection_matches_exhaustive_literal_scores():
    source, context = fixture()
    records = ["XXX", "XY"]
    result = refine_units(source, records, context, ("Y", "X"), max_sweeps=10, max_seconds=10)
    all_scores = [(result["initial_units"], independent_score(source, result["initial_units"], records, context))]
    for sweep in result["trace"]:
        assert sweep["complete"]
        neighbors = sweep["neighbors"]
        assert len(neighbors) == sweep["expected_neighbors"]
        assert len({tuple(row["units"]) for row in neighbors}) == len(neighbors)
        for row in neighbors:
            expected = independent_score(source, row["units"], records, context)
            assert (row["score"] is None) == (expected is None)
            if expected is not None:
                assert row["score"]["total_bits"] == pytest.approx(expected, abs=1e-12)
                all_scores.append((row["units"], expected))
    assert result["score"]["total_bits"] == pytest.approx(min(value for _, value in all_scores), abs=1e-12)
    assert result["best_is_certified_local_optimum"]
    assert result["units"] == result["trace"][-1]["parent_units"]


def test_deadline_keeps_finished_improvement_without_false_certificate(monkeypatch):
    source, context = fixture()
    ticks = iter([0., .01, .02, 2., 2.])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    result = refine_units(source, ["XXX"], context, ("Y", "X"), max_seconds=1.)
    assert result["evaluated_neighbors"] == 1
    assert result["units"] == ["X", "Y"]
    assert result["stop_reason"] == "time_limit"
    assert not result["best_is_certified_local_optimum"]


def test_zero_sweeps_and_invalid_or_unsupported_inputs():
    source, context = fixture()
    result = refine_units(source, ["XXX"], context, ("X", "Y"), max_sweeps=0)
    assert result["units"] == ["X", "Y"] and not result["trace"]
    assert not result["best_is_certified_local_optimum"]
    for arguments in ({"max_seconds": 0}, {"max_sweeps": -1}, {"tolerance": math.nan}):
        with pytest.raises(ValueError):
            refine_units(source, ["X"], context, ("X", "Y"), **arguments)
    with pytest.raises(ValueError, match="support"):
        refine_units(source, ["Y"], context, ("X", "XX"))
