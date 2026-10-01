"""Independent exhaustive tuple scores and bounds, no trained models/data."""

import itertools
import math

import numpy as np
import pytest

from voynich.recurrent_shared_prefix import decode_shared_prefix


def distribution(history):
    # Full-history dependence deliberately cannot be merged by offset or suffix.
    a = 0.07 if history.startswith("b") else 0.91 if history == "aa" else 0.61
    return (a, 1 - a)


class Provider:
    alphabet = "ab"

    def __init__(self, probabilities=distribution):
        self.probabilities = probabilities
        self.histories = []

    def advance(self, tokens, states):
        histories = []
        for token, state in zip(tokens, states, strict=True):
            if state is None:
                assert token == 2
                history = ""
            else:
                history = state + self.alphabet[token]
            histories.append(history)
        self.histories.extend(histories)
        return np.log([self.probabilities(h) for h in histories]), histories


def texts(n):
    return tuple("".join(row) for length in range(n + 1) for row in itertools.product("ab", repeat=length))


def literal(text, key):
    return "".join(key["ab".index(c)] for c in text)


def source_score(tuples, rho, probabilities=distribution):
    value = len(tuples) * math.log(rho) + sum(map(len, tuples)) * math.log1p(-rho)
    for text in tuples:
        for i, c in enumerate(text):
            value += math.log(probabilities(text[:i])["ab".index(c)])
    return value


def all_tuples(keys, records, weights, rho, probabilities=distribution):
    rows = []
    for candidate in itertools.product(*(texts(len(r)) for r in records)):
        compatible = [
            i for i, key in enumerate(keys) if tuple(literal(t, key) for t in candidate) == tuple(records)
        ]
        if compatible:
            high = max(weights[i] for i in compatible)
            key_score = high + math.log(math.fsum(math.exp(weights[i] - high) for i in compatible))
            rows.append(
                (source_score(candidate, rho, probabilities) + key_score, candidate, tuple(compatible))
            )
    return sorted(rows, key=lambda row: (-row[0], row[1]))


def test_every_tiny_bank_and_two_record_observation_matches_exhaustive_global_key_text_enumeration():
    keys = tuple(itertools.product(("x", "y", "xx", "xy"), repeat=2))
    weights = tuple(map(math.log, (0.7, 0.3)))
    count = 0
    for bank in itertools.combinations(keys, 2):
        for records in itertools.product(("", "x", "xx", "xy"), repeat=2):
            expected = all_tuples(bank, records, weights, 0.2)
            provider = Provider()
            actual = decode_shared_prefix(provider, bank, records, 0.2, log_weights=weights, beam_width=256)
            assert actual["discarded_prefixes"] == 0
            assert actual["support_empty"] == (not expected)
            if expected:
                assert actual["plaintexts"] == expected[0][1]
                assert actual["joint_log_probability"] == pytest.approx(expected[0][0], abs=2e-12)
                assert actual["compatible_key_indices"] == expected[0][2]
                assert actual["floating_map_bound_separated"]
            else:
                assert actual["plaintexts"] is actual["joint_log_probability"] is None
                assert not provider.histories
            assert (
                len(provider.histories) == len(set(provider.histories)) == actual["distinct_source_prefixes"]
            )
            count += 1
    assert count == 1920


@pytest.mark.parametrize("width", [1, 2, 256])
def test_every_reported_retained_and_discarded_bound_covers_all_its_complete_descendants(width):
    bank = (("x", "xx"), ("xx", "x"), ("x", "x"))
    records = ("xxxx", "xx")
    weights = tuple(map(math.log, (0.2, 0.3, 0.5)))
    expected = all_tuples(bank, records, weights, 0.2)
    trace = []
    actual = decode_shared_prefix(
        Provider(), bank, records, 0.2, log_weights=weights, beam_width=width, observe=trace.append
    )
    for row in trace:
        assert row["upper_bound"] <= row["basic_upper_bound"] + 1e-12
        descendants = [
            score
            for score, candidate, _ in expected
            if candidate[: len(row["completed"])] == row["completed"]
            and candidate[len(row["completed"])].startswith(row["text"])
        ]
        assert descendants and max(descendants) <= row["upper_bound"] + 2e-12
    if actual["floating_map_bound_separated"]:
        assert actual["joint_log_probability"] == pytest.approx(expected[0][0], abs=2e-12)
    assert not actual["interval_certificate"] and not actual["exact_evidence_computed"]


def test_finish_and_extend_branches_both_survive_and_record_source_resets_are_scored_once():
    bank = (("x", "xx"), ("xx", "x"))
    weights = (-math.log(2),) * 2
    records = ("xx", "", "xx")
    expected = all_tuples(bank, records, weights, 0.3)
    provider = Provider()
    actual = decode_shared_prefix(provider, bank, records, 0.3, log_weights=weights, beam_width=256)
    assert actual["plaintexts"] == expected[0][1]
    assert actual["joint_log_probability"] == pytest.approx(expected[0][0], abs=2e-12)
    assert provider.histories.count("") == 1
    assert actual["source_cache_hits"] > 0


def test_cannot_change_keys_between_records_and_tiny_positive_weights_never_become_impossible():
    bank = (("x", "x"), ("y", "y"))
    weights = (0.0, -10000.0)
    assert decode_shared_prefix(Provider(), bank, ("x", "y"), 0.2, log_weights=weights)["support_empty"]
    actual = decode_shared_prefix(Provider(), bank, ("yy", "y"), 0.2, log_weights=weights, beam_width=256)
    expected = all_tuples(bank, ("yy", "y"), weights, 0.2)
    assert actual["compatible_key_indices"] == (1,)
    assert math.isfinite(actual["joint_log_probability"]) and actual["joint_log_probability"] < -10000
    assert actual["joint_log_probability"] == pytest.approx(expected[0][0], abs=2e-12)


def test_neural_greedy_trap_is_detected_by_discarded_bound_and_wide_search_reaches_exhaustive_best():
    def probabilities(history):
        return (0.6, 0.4) if not history else (0.001, 0.999) if history.startswith("b") else (0.5, 0.5)

    bank, records = (("x", "x"),), ("xxxx",)
    thin = decode_shared_prefix(Provider(probabilities), bank, records, 0.2, beam_width=1)
    wide = decode_shared_prefix(Provider(probabilities), bank, records, 0.2, beam_width=256)
    expected = all_tuples(bank, records, (0.0,), 0.2, probabilities)
    assert expected[0][1] == ("bbbb",)
    assert thin["plaintexts"] != expected[0][1] and not thin["floating_map_bound_separated"]
    assert thin["discarded_completion_upper_bound"] >= expected[0][0]
    assert wide["plaintexts"] == expected[0][1] and wide["floating_map_bound_separated"]


@pytest.mark.parametrize("cap", ["max_expanded", "max_source_prefixes", "max_channel_cells"])
def test_resource_exhaustion_is_failure_and_does_not_return_a_false_empty_support_or_certificate(cap):
    with pytest.raises(RuntimeError, match="cap"):
        decode_shared_prefix(Provider(), (("x", "xx"),), ("xxxx",), 0.2, **{cap: 1})


@pytest.mark.parametrize("bad", ["normalization", "positive_log", "nan", "shape", "states"])
def test_malformed_neural_distribution_is_a_failure(bad):
    class Broken(Provider):
        def advance(self, tokens, states):
            values, following = super().advance(tokens, states)
            if bad == "normalization":
                values -= 1
            elif bad == "positive_log":
                values[0, 0] = 0.1
            elif bad == "nan":
                values[0, 0] = math.nan
            elif bad == "shape":
                values = values[:, :1]
            else:
                following = []
            return values, following

    with pytest.raises(ValueError, match="Provider"):
        decode_shared_prefix(Broken(), (("x", "xx"),), ("x",), 0.2)


@pytest.mark.parametrize("bad", ["duplicate_keys", "empty_units", "weights", "rho", "beam", "records"])
def test_invalid_key_prior_channel_and_limits_rejected(bad):
    bank, records, rho, kwargs = (("x", "xx"),), ("x",), 0.2, {}
    if bad == "duplicate_keys":
        bank = bank * 2
    elif bad == "empty_units":
        bank = (("", "x"),)
    elif bad == "weights":
        kwargs["log_weights"] = (1.0,)
    elif bad == "rho":
        rho = True
    elif bad == "beam":
        kwargs["beam_width"] = 0
    else:
        records = ()
    with pytest.raises(ValueError):
        decode_shared_prefix(Provider(), bank, records, rho, **kwargs)


@pytest.mark.parametrize("seed", [71021, 71029])
def test_real_tiny_lstm_cached_steps_equal_independent_whole_sequence_scores(seed):
    import torch
    from torch.nn import functional as F
    from voynich.recurrent_latin_source import RecurrentSource
    from voynich.recurrent_unit_beam import RecurrentProvider

    torch.manual_seed(seed)
    model = RecurrentSource(alphabet="ab", embedding=5, width=7, layers=2).eval()
    keys = (("x", "xx"), ("xx", "x"), ("x", "x"))
    weights = tuple(map(math.log, (0.2, 0.3, 0.5)))
    records, rho = ("xx", "xxx"), 0.2
    actual = decode_shared_prefix(
        RecurrentProvider(model), keys, records, rho, log_weights=weights, beam_width=256
    )
    rows = []
    with torch.inference_mode():
        for candidate in itertools.product(*(texts(len(r)) for r in records)):
            compatible = [
                i for i, key in enumerate(keys) if tuple(literal(t, key) for t in candidate) == records
            ]
            if not compatible:
                continue
            score = len(records) * math.log(rho) + sum(map(len, candidate)) * math.log1p(-rho)
            for text in candidate:
                inputs = torch.tensor([[2] + ["ab".index(c) for c in text[:-1]]])
                logits, _ = model(inputs)
                logp = F.log_softmax(logits[0].double(), dim=-1)
                score += sum(float(logp[i, "ab".index(c)]) for i, c in enumerate(text))
            high = max(weights[i] for i in compatible)
            score += high + math.log(math.fsum(math.exp(weights[i] - high) for i in compatible))
            rows.append((score, candidate, tuple(compatible)))
    expected = sorted(rows, key=lambda row: (-row[0], row[1]))[0]
    assert actual["plaintexts"] == expected[1]
    assert actual["compatible_key_indices"] == expected[2]
    assert actual["joint_log_probability"] == pytest.approx(expected[0], abs=2e-6)
    assert actual["floating_map_bound_separated"] and not actual["interval_certificate"]


def test_equal_next_letter_predictions_do_not_license_merging_distinct_histories():
    def probabilities(history):
        if not history:
            return (0.55, 0.45)
        if len(history) == 1:
            return (0.5, 0.5)
        return (0.999, 0.001) if history.startswith("b") else (0.5, 0.5)

    provider = Provider(probabilities)
    first, states = provider.advance([0, 1], ["", ""])
    assert np.array_equal(first[0], first[1])
    following, _ = provider.advance([0, 0], states)
    assert not np.array_equal(following[0], following[1])
    bank, records = (("x", "x"),), ("xxxx",)
    expected = all_tuples(bank, records, (0.0,), 0.2, probabilities)
    actual = decode_shared_prefix(Provider(probabilities), bank, records, 0.2, beam_width=32)
    assert actual["plaintexts"] == expected[0][1] == ("baaa",)
    assert actual["joint_log_probability"] == pytest.approx(expected[0][0], abs=1e-12)
