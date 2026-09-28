"""Fabricated trace checks; these tests open no experimental artifacts."""
import copy
import math

import pytest

from scripts.summarize_blind_channel_dev001_search import summarize, validate_case


def fixture():
    first = {"states": ["s0"], "rows": [{"emissions": [{"next_state": "s0", "glyphs": "x"}]}]}
    second = {"states": ["s0"], "rows": [{"emissions": [{"next_state": "s0", "glyphs": "xx"}]}]}

    def score(model, data):
        return {"model_bits": model, "data_bits": data, "total_bits": model + data,
                "log_likelihood": -data * math.log(2)}

    initial_score, raw_score, refined_score = score(2, 8.), score(3, 9.), score(3, 6.)

    def stage(phase, channel, value):
        return {"phase": phase, "channel": channel, "score": value, "status": "scored", "records_scored": 2}

    config = {"seed": 0, "state_counts": [1, 2], "restarts_per_state": 2,
              "proposals_per_restart": 10, "initialization_attempts": 3,
              "max_seconds": 1., "improvement_tolerance_bits": 1e-9}
    full = {"channel": second, "score": refined_score, "config": config, "seconds": 1.1,
            "stop_reason": "time_limit", "proposals": 1, "completed_candidates": 3,
            "unit_pool": ["x", "xx"], "trace": [
                {"event": "initialization", "state_count": 1, "restart": 0, "attempt": 0,
                 "method": "frequency", "accepted": True, "stages": [stage("raw", first, initial_score)],
                 "selected_channel": first, "selected_score": initial_score, "best_total_bits": 10.},
                {"event": "proposal", "state_count": 1, "restart": 0, "step": 0,
                 "move": {"kind": "replace_unit"}, "parent_score": initial_score, "accepted": True,
                 "stages": [stage("raw", second, raw_score), stage("refined", second, refined_score)],
                 "selected_channel": second, "selected_score": refined_score, "best_total_bits": 9.}]}
    frozen = {key: copy.deepcopy(full[key]) for key in
              ("channel", "score", "config", "seconds", "stop_reason", "proposals", "completed_candidates")}
    frozen.update(source_freeze="a" * 40, cpu_seconds_before_final_write=1.2, peak_rss_bytes=1000)
    return frozen, full


def test_refined_stage_minimum_counts_and_budget_coverage_are_reported():
    frozen, full = fixture()
    result = validate_case(frozen, full)
    assert result["returned_minimum_verified"]
    assert result["minimum_recorded_total_bits"] == 9
    assert result["completed_candidates"] == 3 and result["proposals"] == 1
    assert result["attempted_state_counts"] == [1]
    assert result["requested_state_counts"] == [1, 2]
    assert result["attempted_restart_state_pairs"] == 1 and result["maximum_restart_state_pairs"] == 4
    assert result["completed_stage_phases"] == {"raw": 2, "refined": 1}
    assert result["accepted_move_counts"] == {"replace_unit": 1}
    assert result["selected_model"]["emission_length_counts"] == {"2": 1}


def test_rejected_subtolerance_proposal_may_still_be_the_returned_best():
    frozen, full = fixture()
    frozen["config"]["improvement_tolerance_bits"] = full["config"]["improvement_tolerance_bits"] = 2.
    full["trace"][-1].update(accepted=False, rejection_reason="no_sufficient_score_improvement")
    result = validate_case(frozen, full)
    assert result["minimum_recorded_total_bits"] == 9
    assert result["accepted_proposals"] == 0


@pytest.mark.parametrize("field,value", [("proposals", 2), ("completed_candidates", 2), ("seconds", 1.2)])
def test_frozen_full_metadata_mismatch_is_rejected(field, value):
    frozen, full = fixture()
    frozen[field] = value
    with pytest.raises(ValueError, match="Frozen/full mismatch"):
        validate_case(frozen, full)


def test_consistent_but_false_candidate_count_and_nonminimum_return_are_rejected():
    frozen, full = fixture()
    frozen["completed_candidates"] = full["completed_candidates"] = 2
    with pytest.raises(ValueError, match="counts differ"):
        validate_case(frozen, full)
    frozen, full = fixture()
    frozen["channel"] = full["channel"] = full["trace"][0]["selected_channel"]
    frozen["score"] = full["score"] = full["trace"][0]["selected_score"]
    with pytest.raises(ValueError, match="minimum-scored"):
        validate_case(frozen, full)


def test_missing_cases_remain_pending_without_any_other_file_reads(tmp_path):
    result = summarize(tmp_path, names=("toy-a", "toy-b"))
    assert result["status"] == "pending" and result["cases_verified"] == 0
    assert result["pending"] == ["toy-a", "toy-b"]
    assert not result["ciphertext_corpus_transfer_answers_or_evaluation_opened"]
