"""Compiler search invariants: a shared key, complete coverage and bounded work."""

import math

import pytest

from voynich.communication.pipeline import verify_candidate
from voynich.communication.schema import Anchor, FAMILIES, GRAMMARS, MORPHOLOGIES, Observation
from voynich.communication.search import compiler_search, search_grammar
from voynich.communication.worlds import WorldConfig, decode, generate_world, parse_document


def _observation(symbols):
    return Observation("synthetic-text", tuple(symbols), 24, source_group="text-only")


def test_fixed_grammar_proposals_obey_global_channel_and_real_word_boundaries():
    observation = _observation((2, 3, 4, 5, 6))
    result = search_grammar(observation, grammar="SOV", morphology="suffix", family="copy",
                            max_null_fraction=0, beam_size=64, max_candidates=64)
    assert len(result["candidates"]) == 48  # 4 subject stems * 4 object stems * 3 verbs
    for candidate in result["candidates"]:
        hypothesis = candidate["hypothesis"]
        assert hypothesis["boundaries"] == (True, False, True, False, True)
        decoded = decode(observation.symbols, hypothesis["inverse_key"], hypothesis["keep"],
                         hypothesis["boundaries"])
        clauses = parse_document(decoded.plaintext, hypothesis["boundaries"], entities=4,
                                 grammar="SOV", morphology="suffix", family="copy")
        assert len(clauses) == 1
        assert verify_candidate(observation, hypothesis)["valid"]


def test_repeated_glyph_cannot_switch_between_subject_and_object_markers():
    observation = _observation((2, 3, 4, 3, 6))
    result = search_grammar(observation, grammar="SOV", morphology="suffix", max_null_fraction=0)
    assert result["candidates"] == []
    # It cannot delete just one of the conflicting occurrences either.
    result = search_grammar(observation, grammar="SOV", morphology="suffix", max_null_fraction=0.7)
    assert result["candidates"] == []


def test_partial_clause_or_exhausted_prefix_is_never_returned_as_a_document():
    incomplete = search_grammar(_observation((2, 3, 4, 5)), grammar="SOV", morphology="suffix",
                                max_null_fraction=0)
    assert incomplete["candidates"] == []
    exhausted = search_grammar(_observation((2, 3, 4, 5, 6)), grammar="SOV", morphology="suffix",
                               max_expansions=1)
    assert exhausted["expansions"] == 1
    assert exhausted["budget_exhausted"]
    assert not exhausted["all_observations_processed"]
    assert not exhausted["candidates"]


def test_explicit_anchors_and_null_budget_are_respected_without_gold_input():
    world = generate_world(WorldConfig(8, events=4, homophones=2, null_rate=0.25))
    anchors = tuple(Anchor(symbol, world.inverse_key[symbol - 2], "explicit-synthetic-correspondence")
                    for symbol in sorted(set(world.observation)) if world.inverse_key[symbol - 2])
    observation = Observation("anchored", world.observation, 24, anchors, "text")
    result = search_grammar(observation, grammar="SOV", morphology="suffix", max_candidates=64,
                            max_expansions=20000, beam_size=64, max_null_fraction=0.3)
    assert result["candidates"]
    assert any(decode(observation.symbols, row["hypothesis"]["inverse_key"]).plaintext == world.plaintext
               for row in result["candidates"])
    for row in result["candidates"]:
        hypothesis = row["hypothesis"]
        assert sum(not value for value in hypothesis["keep"]) <= math.floor(len(world.observation) * 0.3)
        for anchor in anchors:
            assert hypothesis["inverse_key"][anchor.observed - 2] == anchor.canonical
        assert all(not boundary or keep for boundary, keep in zip(hypothesis["boundaries"], hypothesis["keep"]))


def test_log_key_prior_is_charged_once_per_global_assignment_not_per_occurrence():
    observation = _observation((2, 3, 4, 5, 6) * 2)
    flat = search_grammar(observation, grammar="SOV", morphology="suffix", max_null_fraction=0)
    penalized = search_grammar(observation, grammar="SOV", morphology="suffix", max_null_fraction=0,
                               key_log_probs=[[-1.0] * 26 for _ in range(24)])
    assert flat["candidates"][0]["hypothesis"] == penalized["candidates"][0]["hypothesis"]
    assert flat["candidates"][0]["compiler_score"] - penalized["candidates"][0]["compiler_score"] == pytest.approx(5)


def test_shared_compiler_budget_and_explicit_prior_guidance():
    world = generate_world(WorldConfig(41, grammar="VOS", morphology="prefix", events=3))
    observation = _observation(world.observation)
    # Positive-control likelihood supplied explicitly, never read secretly by search.
    scores = [[-math.inf] * 26 for _ in range(24)]
    for index, canonical in enumerate(world.inverse_key):
        scores[index][canonical] = 0
    global_scores = {name: [0.0 if value == desired else -math.inf for value in options]
                     for name, options, desired in (("grammar", GRAMMARS, "VOS"),
                                                    ("morphology", MORPHOLOGIES, "prefix"),
                                                    ("family", FAMILIES, "procedure"))}
    result = compiler_search(observation, key_log_probs=scores, global_log_probs=global_scores,
                             max_expansions=500, max_candidates=4)
    assert result["expansions"] <= 500
    assert result["combinations_attempted"] == 1
    assert result["candidates"]
    selected = result["candidates"][0]["hypothesis"]
    assert decode(observation.symbols, selected["inverse_key"]).plaintext == world.plaintext
    assert verify_candidate(observation, selected)["valid"]
    plain = compiler_search(_observation((2, 3, 4, 5, 6)), max_expansions=50)
    assert plain["expansions"] <= 50
    assert sum(row["expansions"] for row in plain["combination_reports"]) == plain["expansions"]


def test_unobserved_anchor_is_preserved_while_other_unknown_slots_stay_unresolved():
    observation = Observation("x", (2, 3, 4, 5, 6), 24, (Anchor(24, 8, "external"),), "text")
    result = search_grammar(observation, grammar="SOV", morphology="suffix", max_null_fraction=0)
    hypothesis = result["candidates"][0]["hypothesis"]
    assert hypothesis["inverse_key"][22] == 8
    assert 24 not in hypothesis["unresolved_key_symbols"]
    assert 25 in hypothesis["unresolved_key_symbols"]


@pytest.mark.parametrize("kwargs", [{"max_expansions": 0}, {"max_expansions": 1_000_001},
                                    {"beam_size": True}, {"max_candidates": 0},
                                    {"max_null_fraction": 1}, {"max_null_fraction": float("nan")}])
def test_bad_search_budget_rejected(kwargs):
    with pytest.raises(ValueError):
        compiler_search(_observation((2, 3, 4)), **kwargs)


def test_gold_objects_and_malformed_priors_rejected():
    with pytest.raises(ValueError, match="Observation"):
        compiler_search(generate_world(WorldConfig(3)))
    with pytest.raises(ValueError, match="one row"):
        compiler_search(_observation((2, 3, 4)), key_log_probs=[[0] * 26])
    with pytest.raises(ValueError, match="nonpositive"):
        compiler_search(_observation((2, 3, 4)), key_log_probs=[[1] * 26 for _ in range(24)])
    with pytest.raises(ValueError, match="vectors"):
        compiler_search(_observation((2, 3, 4)), global_log_probs={})
