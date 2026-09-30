"""Artificial controls only; no corpus, pilot, source artifact, or answer access."""
import hashlib
import itertools
import json
import math
import random
from collections import Counter
from fractions import Fraction

import pytest

from voynich.finite_state_channel import SourceModel, forward_log_probability
from voynich.unit_channel_decision import (
    _BackwardLattice,
    edit_distance,
    posterior_sample_plaintexts,
    sample_mbr_decode,
)


def source(order=1):
    rows = {"": {"a": .25, "b": .75}}
    if order:
        rows.update(a={"a": .5, "b": .5}, b={"a": .75, "b": .25})
    return SourceModel(("a", "b"), order, rows)


def enumerate_paths(model, units, record, rho):
    results = {}

    def extend(text, output, probability, context):
        if output == record:
            results[text] = probability * rho
            return
        for letter, unit in zip(model.alphabet, units, strict=True):
            emitted = output + unit
            if record.startswith(emitted):
                weight = Fraction(model.probabilities[context][letter])
                if weight:
                    extend(text + letter, emitted, probability * (1 - rho) * weight,
                           letter if model.order else "")

    extend("", "", Fraction(1), "")
    return results


def scalar_edit(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        row = [i]
        for j, b in enumerate(right, 1):
            row.append(min(previous[j] + 1, row[-1] + 1, previous[j - 1] + (a != b)))
        previous = row
    return previous[-1]


@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("units,record", [(("x", "xx"), "xxxx"), (("x", "x"), "xxx"),
                                         (("x\0", "x"), "x\0x"), (("α", "😀α"), "😀αα"),
                                         (("xx", "xxxx"), "xxxxxx"), (("x", "xx"), "")])
def test_every_sampling_path_probability_matches_fraction_posterior(order, units, record):
    model, rho = source(order), Fraction(1, 2)
    exact = enumerate_paths(model, units, record, rho)
    normalizer = sum(exact.values())
    lattice = _BackwardLattice(model, units, record, float(rho))
    assert lattice.log_likelihood == pytest.approx(math.log(normalizer), abs=3e-14)
    assert lattice.log_likelihood == pytest.approx(
        forward_log_probability(model, lattice.channel(), record), abs=3e-14)
    for text, joint in exact.items():
        offset = context = 0
        conditional = 1.
        for char in text:
            choices = lattice.choices(offset, context)
            assert math.fsum(probability for _, _, probability in choices) == pytest.approx(1., abs=2e-15)
            letter = model.alphabet.index(char)
            _, offset, probability = next(edge for edge in choices if edge[0] == letter)
            conditional *= probability
            context = int(lattice.next_context[letter])
        assert offset == len(record)
        assert conditional == pytest.approx(float(joint / normalizer), rel=2e-14, abs=1e-16)


def test_seeded_samples_follow_enumerated_probabilities():
    model, units, record = source(), ("x", "xx"), "xxxx"
    exact = enumerate_paths(model, units, record, Fraction(1, 2))
    total = sum(exact.values())
    bank = posterior_sample_plaintexts(model, units, record, .5, draws=20_000, seed=81917)
    counts = Counter(bank.samples)
    assert set(counts) == set(exact)
    for text, weight in exact.items():
        probability = float(weight / total)
        standard_error = math.sqrt(probability * (1 - probability) / len(bank.samples))
        assert abs(counts[text] / len(bank.samples) - probability) < 6 * standard_error + .001


@pytest.mark.parametrize("record", ["z", "xxx"])
def test_impossible_has_no_conditional_samples_and_serializes(record):
    model = SourceModel(("a", "b"), 0, {"": {"a": 1., "b": 0.}})
    sample = posterior_sample_plaintexts(model, ("xx", "x"), record, .5, draws=10, seed=1)
    assert sample.samples == () and sample.log_likelihood == -math.inf
    result = sample_mbr_decode(model, ("xx", "x"), record, .5).to_dict()
    assert result["plaintext"] is result["map_plaintext"] is None
    assert result["log_likelihood"] is None
    assert result["candidate_samples"] == result["risk_samples"] == result["candidate_risks"] == []
    json.dumps(result, allow_nan=False)


def test_empty_record_includes_stop_and_has_one_reading():
    result = sample_mbr_decode(source(), ("x", "xx"), "", .125).to_dict()
    assert result["plaintext"] == result["map_plaintext"] == ""
    assert result["log_likelihood"] == result["map_log_probability"] == math.log(.125)
    assert result["candidate_samples"] == [""] * 32
    assert result["risk_samples"] == [""] * 256
    assert result["candidate_risks"] == [{"plaintext": "", "total_edit_distance": 0, "mean_edit_distance": 0.}]
    json.dumps(result, allow_nan=False)


def test_long_low_probability_record_is_sampled_without_probability_underflow():
    model = SourceModel(("a",), 0, {"": {"a": 1.}})
    record = "x" * 2000
    result = posterior_sample_plaintexts(model, ("x",), record, .5, draws=2, seed=7)
    assert result.samples == ("a" * 2000,) * 2
    assert result.log_likelihood == pytest.approx(2001 * math.log(.5), abs=1e-9)


def test_rare_aliased_context_survives_until_unique_suffix():
    model = SourceModel(("a", "b", "c"), 1, {
        "": {"a": .5, "b": .5, "c": 0.},
        "a": {"a": 1., "b": 0., "c": 0.},
        "b": {"a": 0., "b": .5, "c": .5},
        "c": {"a": 0., "b": 0., "c": 1.},
    })
    result = posterior_sample_plaintexts(model, ("x", "x", "y"), "x" * 1100 + "y",
                                         .25, draws=3, seed=9)
    assert result.samples == ("b" * 1100 + "c",) * 3
    assert result.log_likelihood == pytest.approx(math.log(.25) + 1101 * math.log(.375), abs=1e-9)


def test_bitvector_matches_independent_scalar_edit_recurrence():
    texts = ["".join(chars) for n in range(5) for chars in itertools.product("ab", repeat=n)]
    texts += ["α😀\0a", "😀αa", "a" * 130 + "b", "b" + "a" * 130]
    for left, right in itertools.product(texts, repeat=2):
        assert edit_distance(left, right) == scalar_edit(left, right)
    rng = random.Random(7751)
    for _ in range(30):
        left = "".join(rng.choices("abc😀", k=rng.randrange(100, 250)))
        right = "".join(rng.choices("abc😀", k=rng.randrange(100, 250)))
        assert edit_distance(left, right) == scalar_edit(left, right)


def test_banks_independent_of_opposite_draw_count_and_hashes_replay():
    args = (source(), ("x", "xx"), "xxxxxx", .5)
    first = sample_mbr_decode(*args, candidate_draws=5, risk_draws=17, seed=91).to_dict()
    larger_candidates = sample_mbr_decode(*args, candidate_draws=11, risk_draws=17, seed=91).to_dict()
    larger_risk = sample_mbr_decode(*args, candidate_draws=5, risk_draws=29, seed=91).to_dict()
    assert first["risk_samples"] == larger_candidates["risk_samples"]
    assert first["candidate_samples"] == larger_risk["candidate_samples"]
    assert larger_candidates["candidate_samples"][:5] == first["candidate_samples"]
    assert larger_risk["risk_samples"][:17] == first["risk_samples"]
    assert first["candidate_seed"] != first["risk_seed"]
    for prefix, bank_name in (("candidate", "candidates"), ("risk", "risk")):
        stream = hashlib.sha256(f"unit-channel-decision/v1:{bank_name}:91".encode()).hexdigest()
        assert first[f"{prefix}_seed"] == stream
        samples = first[f"{prefix}_samples"]
        raw = json.dumps(samples, ensure_ascii=True, separators=(",", ":")).encode()
        assert first[f"{prefix}_bank_sha256"] == hashlib.sha256(raw).hexdigest()


def test_all_risks_recomputed_with_multiplicity_and_map_first_ties():
    result = sample_mbr_decode(source(), ("x", "xx"), "xxxxxx", .5, seed=292).to_dict()
    candidates = list(dict.fromkeys((result["map_plaintext"], *result["candidate_samples"])))
    assert candidates == [row["plaintext"] for row in result["candidate_risks"]]
    assert len(set(result["risk_samples"])) < len(result["risk_samples"])
    totals = [sum(scalar_edit(candidate, reference) for reference in result["risk_samples"])
              for candidate in candidates]
    assert totals == [row["total_edit_distance"] for row in result["candidate_risks"]]
    assert result["plaintext"] == candidates[min(range(len(totals)), key=totals.__getitem__)]
    assert result["estimated_risk"] <= result["map_estimated_risk"]
    assert result == sample_mbr_decode(source(), ("x", "xx"), "xxxxxx", .5, seed=292).to_dict()


def test_edit_risk_can_prefer_non_map_plaintext():
    # Conditional on xx: aa has mass .4; bb .35; bc .25. Exact risks are
    # aa=1.2, bb=1.05, bc=1.15, so the MAP is not an edit-risk minimizer.
    model = SourceModel(("a", "b", "c"), 1, {
        "": {"a": .4, "b": .6, "c": 0.},
        "a": {"a": 1., "b": 0., "c": 0.},
        "b": {"a": 0., "b": 7 / 12, "c": 5 / 12},
        "c": {"a": 0., "b": 0., "c": 1.},
    })
    result = sample_mbr_decode(model, ("x",) * 3, "xx", .5, risk_draws=4096, seed=77).to_dict()
    assert result["map_plaintext"] == "aa"
    assert result["plaintext"] == "bb"
    assert result["estimated_risk"] < result["map_estimated_risk"]


def test_zero_candidate_draws_retains_map_and_risk_bank():
    result = sample_mbr_decode(source(), ("x", "xx"), "xxxx", .5, candidate_draws=0).to_dict()
    assert result["plaintext"] == result["map_plaintext"]
    assert result["candidate_samples"] == [] and len(result["risk_samples"]) == 256
    assert len(result["candidate_risks"]) == 1


@pytest.mark.parametrize("kwargs", [{"candidate_draws": -1}, {"candidate_draws": True},
                                    {"risk_draws": 0}, {"risk_draws": 1.5}, {"seed": True}])
def test_invalid_budgets_rejected(kwargs):
    with pytest.raises(ValueError):
        sample_mbr_decode(source(), ("x", "xx"), "xx", .5, **kwargs)


@pytest.mark.parametrize("units,record,rho", [("xx", "x", .5), (("x",), "x", .5),
                                           (("", "x"), "x", .5), (("x", "x"), None, .5),
                                           (("x", "x"), "x", True), (("x", "x"), "x", 0),
                                           (("x", "x"), "x", 1), (("x", "x"), "x", math.nan)])
def test_invalid_channel_inputs_rejected(units, record, rho):
    with pytest.raises(ValueError):
        sample_mbr_decode(source(), units, record, rho)
