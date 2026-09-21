from dataclasses import FrozenInstanceError
import json
import math

import pytest

from voynich.communication.evidence import (
    BeliefState, Evidence, Hypothesis, Query, expected_information_gain, rank_queries,
)


def state(priors=(0.5, 0.5)):
    return BeliefState(tuple(Hypothesis(name, math.log(prior) if prior else -math.inf, {"name": name})
                             for name, prior in zip(("a", "b"), priors, strict=True)))


def test_bayesian_update_is_analytic_and_does_not_mutate_prior():
    prior = state()
    updated = prior.update(Evidence("e1", "independent-scan", {"a": math.log(0.8), "b": math.log(0.2)}))
    assert prior.probabilities == {"a": 0.5, "b": 0.5}
    assert updated.probabilities == pytest.approx({"a": 0.8, "b": 0.2})
    assert updated.entropy == pytest.approx(-0.8 * math.log(0.8) - 0.2 * math.log(0.2))
    assert updated.effective_sample_size == pytest.approx(1 / (0.8**2 + 0.2**2))
    assert prior.ledger == () and len(updated.ledger) == 1
    assert sum(math.exp(value) for value in updated.log_weights.values()) == pytest.approx(1)
    twice = updated.update(Evidence("e2", "independent-annotation", {"a": math.log(0.2), "b": math.log(0.8)}))
    assert twice.probabilities == pytest.approx({"a": 0.5, "b": 0.5})


def test_duplicate_ids_dependent_sources_and_missing_hypotheses_are_refused():
    evidence = Evidence("e1", "one-scan", {"a": 0., "b": -1.})
    updated = state().update(evidence)
    with pytest.raises(ValueError, match="duplicate evidence id"):
        updated.update(Evidence("e1", "different-group", {"a": 0., "b": -1.}))
    with pytest.raises(ValueError, match="dependent source_group"):
        updated.update(Evidence("different-id", "one-scan", {"a": 0., "b": -1.}))
    with pytest.raises(ValueError, match="exactly every hypothesis"):
        state().update(Evidence("e2", "another-scan", {"a": 0.}))
    with pytest.raises(ValueError, match="exactly every hypothesis"):
        state().update(Evidence("e2", "another-scan", {"a": 0., "b": 0., "c": 0.}))
    with pytest.raises(ValueError, match="duplicate hypothesis"):
        BeliefState((Hypothesis("a", 0), Hypothesis("a", -1)))


def test_extreme_log_probabilities_remain_stable_and_underflowed_tails_can_recover():
    prior = BeliefState((Hypothesis("a", -10_000), Hypothesis("b", 0)))
    assert prior.probabilities["a"] == 0
    assert prior.log_weights["a"] == -10_000
    recovered = prior.update(Evidence("opposite", "independent-source", {"a": 0., "b": -10_000.}))
    assert recovered.probabilities == pytest.approx({"a": 0.5, "b": 0.5})
    almost_certain = state().update(Evidence("tiny", "tiny-source", {"a": -100_000., "b": -100_001.}))
    assert almost_certain.probabilities["a"] == pytest.approx(1 / (1 + math.exp(-1)))
    stable = state()
    for index in range(3):
        stable = stable.update(Evidence(str(index), f"source-{index}", {"a": -1e308, "b": -1e308}))
    assert stable.probabilities == {"a": 0.5, "b": 0.5}
    impossible_prior = state((1, 0)).update(Evidence("extreme", "source", {"a": -1e308, "b": 1e308}))
    assert impossible_prior.probabilities == {"a": 1.0, "b": 0.0}


def test_impossible_evidence_is_explicit_and_not_uniform_fallback():
    with pytest.raises(ValueError, match="impossible evidence"):
        state().update(Evidence("none", "source", {"a": -math.inf, "b": -math.inf}))
    with pytest.raises(ValueError, match="impossible evidence"):
        state((1, 0)).update(Evidence("only-dead", "source", {"a": -math.inf, "b": 0.}))
    with pytest.raises(ValueError, match="zero weight"):
        BeliefState((Hypothesis("a", -math.inf),))


def test_belief_and_evidence_payloads_are_deeply_immutable():
    payload = {"list": [{"value": 1}]}
    weights = {"a": 0., "b": -1.}
    hypothesis = Hypothesis("a", 0., payload)
    evidence = Evidence("e", "group", weights)
    payload["list"][0]["value"] = 99
    weights["a"] = -99.
    assert hypothesis.payload["list"][0]["value"] == 1
    assert evidence.log_likelihoods["a"] == 0
    with pytest.raises(TypeError):
        hypothesis.payload["list"][0]["value"] = 9
    with pytest.raises(TypeError):
        evidence.log_likelihoods["a"] = -2
    with pytest.raises(FrozenInstanceError):
        hypothesis.id = "changed"
    saved = state()
    probabilities = saved.probabilities
    probabilities["a"] = 1.
    assert saved.probabilities["a"] == 0.5


def test_json_round_trip_reconstructs_posterior_from_ledger_and_handles_log_zero():
    original = state().update(Evidence("anchor", "independent-anchor", {"a": 0., "b": -math.inf}, "known mark"))
    serialized = json.dumps(original.to_dict(), allow_nan=False)
    assert "Infinity" not in serialized and "NaN" not in serialized
    restored = BeliefState.from_dict(json.loads(serialized))
    assert restored.probabilities == original.probabilities
    assert restored.log_weights == original.log_weights
    assert restored.to_dict() == original.to_dict()
    altered = original.to_dict()
    altered["ledger"].append(altered["ledger"][0])
    with pytest.raises(ValueError, match="duplicate evidence id"):
        BeliefState.from_dict(altered)
    evidence = original.ledger[0]
    assert Evidence.from_dict(json.loads(json.dumps(evidence.to_dict(), allow_nan=False))) == evidence


def test_information_gain_matches_perfect_uninformative_and_noisy_binary_channels():
    prior = state()
    perfect = Query("perfect", 1, {"a": {"left": 1., "right": 0.}, "b": {"left": 0., "right": 1.}})
    empty = Query("uninformative", 1, {"a": {"left": 0.4, "right": 0.6}, "b": {"left": 0.4, "right": 0.6}})
    noisy = Query("noisy", 1, {"a": {"left": 0.8, "right": 0.2}, "b": {"left": 0.2, "right": 0.8}})
    entropy_binary_02 = -0.8 * math.log(0.8) - 0.2 * math.log(0.2)
    assert expected_information_gain(prior, perfect) == pytest.approx(math.log(2))
    assert expected_information_gain(prior, empty) == pytest.approx(0, abs=1e-14)
    assert expected_information_gain(prior, noisy) == pytest.approx(math.log(2) - entropy_binary_02)
    assert expected_information_gain(state((1, 0)), perfect) == 0
    assert prior.ledger == ()  # Ranking is hypothetical and never fabricates a measurement.


def test_information_gain_handles_zero_marginal_outcomes_and_unequal_priors():
    prior = state((0.9, 0.1))
    query = Query("perfect", 2, {
        "a": {"a-result": 1., "b-result": 0., "never": 0.},
        "b": {"a-result": 0., "b-result": 1., "never": 0.},
    })
    assert expected_information_gain(prior, query) == pytest.approx(prior.entropy)
    missing = Query("missing", 1, {"a": {"yes": 1.}})
    with pytest.raises(ValueError, match="exactly every hypothesis"):
        expected_information_gain(prior, missing)


def test_query_ranking_uses_information_per_cost_and_stable_id_ties():
    probabilities = {"a": {"yes": 1., "no": 0.}, "b": {"yes": 0., "no": 1.}}
    queries = [Query("expensive", 10, probabilities), Query("z-cheap", 1, probabilities),
               Query("a-cheap", 1, probabilities)]
    ranked = rank_queries(state(), queries)
    assert [item["query_id"] for item in ranked] == ["a-cheap", "z-cheap", "expensive"]
    assert ranked[-1]["information_per_cost"] == pytest.approx(math.log(2) / 10)
    with pytest.raises(ValueError, match="duplicate query id"):
        rank_queries(state(), [queries[0], queries[0]])


@pytest.mark.parametrize("constructor", [
    lambda: Hypothesis("", 0), lambda: Hypothesis("a", float("nan")), lambda: Hypothesis("a", math.inf),
    lambda: Hypothesis("a", True), lambda: Hypothesis("a", 0, {"x": math.inf}),
    lambda: Hypothesis("a", 0, {1: "bad-key"}), lambda: Evidence("", "g", {"a": 0}),
    lambda: Evidence("e", " ", {"a": 0}), lambda: Evidence("e", "g", {"a": float("nan")}),
    lambda: Evidence("e", "g", {"a": math.inf}), lambda: Evidence("e", "g", {}),
    lambda: BeliefState(()), lambda: BeliefState(("not a hypothesis",)),
    lambda: Query("q", 0, {"a": {"yes": 1}}), lambda: Query("q", -1, {"a": {"yes": 1}}),
    lambda: Query("q", True, {"a": {"yes": 1}}), lambda: Query("q", math.inf, {"a": {"yes": 1}}),
    lambda: Query("q", 1, {"a": {"yes": .7}}), lambda: Query("q", 1, {"a": {"yes": 2., "no": -1.}}),
    lambda: Query("q", 1, {"a": {"yes": float("nan")}}),
    lambda: Query("q", 1, {"a": {"yes": True}}),
    lambda: Query("q", 1, {"a": {"yes": 1.}, "b": {"different": 1.}}),
])
def test_invalid_evidence_inputs_are_rejected(constructor):
    with pytest.raises(ValueError):
        constructor()


@pytest.mark.parametrize("change", [
    lambda value: value.update(extra=True), lambda value: value.update(schema_version=True),
    lambda value: value.update(schema_version=2), lambda value: value.update(ledger="invalid"),
    lambda value: value["hypotheses"][0].update(log_prior=math.inf),
    lambda value: value["hypotheses"][0].update(untrusted_extra=1),
])
def test_invalid_serialized_states_are_rejected(change):
    value = state().to_dict()
    change(value)
    with pytest.raises(ValueError):
        BeliefState.from_dict(value)
