"""Independent exact posterior and raw edit-risk controls on artificial data."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from fractions import Fraction as F

import pytest

from voynich import unit_channel_decision as decision
from voynich.finite_state_channel import SourceModel


def words(alphabet, bound):
    return ("".join(chars) for length in range(bound + 1) for chars in itertools.product(alphabet, repeat=length))


def enumeration(source, units, observed, rho, start=""):
    paths = {}
    for text in words(source.alphabet, len(observed)):
        if "".join(units[source.alphabet.index(char)] for char in text) != observed:
            continue
        mass, history = F(rho), start if source.order else ""
        for char in text:
            mass *= (1 - F(rho)) * F(source.probabilities[history][char])
            history = char if source.order else ""
        if mass:
            paths[text] = mass
    return paths


def simple_distance(left, right):
    # Ordinary two-row scalar dynamic programming; no bit-vector operations.
    previous = list(range(len(right) + 1))
    for i, char in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def source_fixture(order):
    rows = {"": {"a": .75, "b": .25}}
    if order:
        rows.update(a={"a": .125, "b": .875}, b={"a": .625, "b": .375})
    return SourceModel(("a", "b"), order, rows)


@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("units", [("x", "xx"), ("x", "x"), ("xy", "yx"), ("y", "xx")])
def test_backward_values_edges_and_path_products_equal_independent_fraction_posteriors(order, units):
    source, rho = source_fixture(order), .25
    for record in words("xy", 4):
        lattice = decision._BackwardLattice(source, units, record, rho)
        for offset in range(len(record) + 1):
            for context_index, history in enumerate(lattice.contexts):
                paths = enumeration(source, units, record[offset:], rho, history)
                total = sum(paths.values(), F(0))
                expected = math.log(float(total)) if total else -math.inf
                if total:
                    assert lattice.beta[offset, context_index] == pytest.approx(expected, rel=0, abs=1e-14)
                else:
                    assert lattice.beta[offset, context_index] == -math.inf
                if offset == len(record) or not paths:
                    continue
                expected_edges = {letter: sum((mass for text, mass in paths.items() if text[0] == letter), F(0)) / total
                                  for letter in source.alphabet}
                edges = lattice.choices(offset, context_index)
                assert {source.alphabet[a] for a, _, _ in edges} == {a for a, p in expected_edges.items() if p}
                assert math.fsum(p for _, _, p in edges) == pytest.approx(1., rel=0, abs=3e-16)
                for a, end, probability in edges:
                    assert end == offset + len(units[a])
                    assert probability == pytest.approx(float(expected_edges[source.alphabet[a]]), rel=0, abs=3e-16)
        paths = enumeration(source, units, record, rho)
        total = sum(paths.values(), F(0))
        for text, mass in paths.items():
            offset = context = 0
            product = 1.
            for char in text:
                a, end, probability = next(edge for edge in lattice.choices(offset, context)
                                           if source.alphabet[edge[0]] == char)
                product *= probability
                offset = end
                context = source.alphabet.index(char) + 1 if order else 0
            assert offset == len(record)
            assert product == pytest.approx(float(mass / total), rel=0, abs=5e-16)


def test_categorical_boundaries_use_the_correct_posterior_intervals():
    source = SourceModel(("a", "b"), 0, {"": {"a": .5, "b": .5}})
    lattice = decision._BackwardLattice(source, ("x", "x"), "x", .5)

    class Quantiles:
        values = iter((0., .499999, .5, .999999))

        def random(self):
            return next(self.values)

    assert lattice.sample(4, Quantiles()) == ("a", "a", "b", "b")


def test_actual_samples_are_supported_reproducible_and_banks_do_not_share_rng_consumption():
    source, units, record = source_fixture(1), ("x", "xx"), "xxxx"
    posterior = enumeration(source, units, record, .25)
    samples = decision.posterior_sample_plaintexts(source, units, record, .25, draws=400, seed=17)
    assert len(samples.samples) == 400 and set(samples.samples) <= set(posterior)
    assert samples == decision.posterior_sample_plaintexts(source, units, record, .25, draws=400, seed=17)
    first = decision.sample_mbr_decode(source, units, record, .25, candidate_draws=7, risk_draws=19, seed=23)
    more_candidates = decision.sample_mbr_decode(source, units, record, .25, candidate_draws=13, risk_draws=19, seed=23)
    more_risk = decision.sample_mbr_decode(source, units, record, .25, candidate_draws=7, risk_draws=31, seed=23)
    assert first.risk_samples == more_candidates.risk_samples
    assert first.candidate_samples == more_risk.candidate_samples
    assert first.candidate_samples == more_candidates.candidate_samples[:7]
    assert first.risk_samples == more_risk.risk_samples[:19]
    assert first.candidate_seed != first.risk_seed
    for bank, expected in (("candidates", first.candidate_seed), ("risk", first.risk_seed)):
        assert expected == hashlib.sha256(f"unit-channel-decision/v1:{bank}:23".encode()).hexdigest()
    serialized = first.to_dict()
    for name in ("candidate", "risk"):
        raw = json.dumps(serialized[name + "_samples"], ensure_ascii=True, separators=(",", ":")).encode()
        assert serialized[name + "_bank_sha256"] == hashlib.sha256(raw).hexdigest()
    json.dumps(serialized, allow_nan=False)


def test_every_reported_risk_uses_raw_edit_distance_and_reference_multiplicity():
    source, units, observed = source_fixture(1), ("x", "xx"), "xxxxxx"
    result = decision.sample_mbr_decode(source, units, observed, .25, candidate_draws=32, risk_draws=64, seed=4)
    candidates = list(dict.fromkeys((result.map_plaintext, *result.candidate_samples)))
    assert [row["plaintext"] for row in result.candidate_risks] == candidates
    totals = []
    for row in result.candidate_risks:
        total = sum(simple_distance(row["plaintext"], text) for text in result.risk_samples)
        assert row["total_edit_distance"] == total
        assert row["mean_edit_distance"] == total / 64
        totals.append(total)
    assert result.plaintext == candidates[min(range(len(totals)), key=totals.__getitem__)]
    paths = enumeration(source, units, observed, .25)
    assert paths[result.map_plaintext] == max(paths.values())
    assert result.to_dict()["estimated_risk"] <= result.to_dict()["map_estimated_risk"]


@pytest.mark.parametrize("candidates,references,expected", [
    (("bb", "ab", "bb"), ("bb", "bb", "bb", "aa"), "bb"),
    (("bb", "ab"), ("aa", "bb"), "aa"),
    (("bb", "ab"), ("bb", "ab"), "bb"),
])
def test_finite_candidate_decision_and_ties_are_exact_without_claiming_true_error_gain(monkeypatch, candidates, references, expected):
    source = SourceModel(("a", "b"), 0, {"": {"a": .5, "b": .5}})
    banks = iter((candidates, references))
    monkeypatch.setattr(decision._BackwardLattice, "sample", lambda _self, _draws, _rng: next(banks))
    result = decision.sample_mbr_decode(source, ("x", "x"), "xx", .5,
                                       candidate_draws=len(candidates), risk_draws=len(references), seed=0)
    assert result.map_plaintext == "aa" and result.plaintext == expected
    for row in result.candidate_risks:
        assert row["total_edit_distance"] == sum(simple_distance(row["plaintext"], text) for text in references)
    if expected == "bb":
        # Empirical risk may prefer bb even when aa is the actual plaintext.
        assert simple_distance(result.plaintext, "aa") > simple_distance(result.map_plaintext, "aa")


def test_empty_impossible_zero_draw_and_rare_path_cases_remain_explicit():
    source = source_fixture(1)
    empty = decision.sample_mbr_decode(source, ("x", "xx"), "", .25, candidate_draws=0, risk_draws=4)
    assert empty.plaintext == empty.map_plaintext == "" and empty.risk_samples == ("",) * 4
    assert empty.to_dict()["estimated_risk"] == 0.
    impossible = decision.sample_mbr_decode(source, ("x", "xx"), "y", .25)
    assert impossible.plaintext is impossible.map_plaintext is None
    assert impossible.candidate_samples == impossible.risk_samples == impossible.candidate_risks == ()
    assert impossible.to_dict()["status"] == "impossible"
    no_draws = decision.posterior_sample_plaintexts(source, ("x", "xx"), "xxx", .25, draws=0, seed=0)
    assert no_draws.samples == () and math.isfinite(no_draws.log_likelihood)
    rare = SourceModel(("a", "b"), 1, {"": {"a": 1e-200, "b": 1.},
                                       "a": {"a": 1., "b": 0.}, "b": {"a": 0., "b": 1.}})
    result = decision.posterior_sample_plaintexts(rare, ("x", "y"), "x" * 800, .25, draws=3, seed=1)
    assert result.samples == ("a" * 800,) * 3
    assert result.log_likelihood == pytest.approx(math.log(1e-200) + 800 * math.log(.75) + math.log(.25), rel=0, abs=1e-10)
    json.dumps(impossible.to_dict(), allow_nan=False)


def test_bitvector_distance_matches_scalar_recurrence_including_machine_word_boundaries():
    short = tuple(words(("a", "雪", "\0"), 3))
    for left, right in itertools.product(short, repeat=2):
        assert decision.edit_distance(left, right) == simple_distance(left, right)
    for length in (63, 64, 65, 127, 128, 129):
        pairs = (("a" * length, "a" * (length - 1) + "b"),
                 (("ab" * length)[:length], ("ba" * length)[:length]),
                 ("雪" * length, "\0" * (length + 2)))
        for left, right in pairs:
            assert decision.edit_distance(left, right) == simple_distance(left, right)
            assert decision.edit_distance(right, left) == simple_distance(left, right)
