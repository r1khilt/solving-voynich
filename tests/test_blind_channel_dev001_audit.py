"""Auditor tests use only hand fixtures and temporary toy artifact chains."""
from __future__ import annotations

import copy
import gzip
import itertools
import json
import math
from fractions import Fraction as F

import pytest

from scripts import audit_blind_channel_dev001 as auditor


def source(alphabet=("a",), probabilities=None):
    return {"schema_version": 1, "alphabet": list(alphabet), "order": 0,
            "probabilities": {"": probabilities or {"a": 1.0}}}


def channel(rows=None, glyphs=("x",), states=("s",), initial=None, rho=0.5):
    return {"schema_version": 1, "states": list(states), "glyph_alphabet": list(glyphs),
            "initial": initial or {"s": 1.0}, "stop_probability": rho, "max_emission_length": 2,
            "rows": rows or [row("s", "a", [("s", "x", 0.5), ("s", "xx", 0.5)])]}


def row(state, letter, emissions):
    return {"state": state, "letter": letter,
            "emissions": [{"next_state": destination, "glyphs": unit, "probability": probability}
                          for destination, unit, probability in emissions]}


def test_ambiguous_segmentation_sums_all_paths_and_decodes_joint_maximum():
    result = auditor.evaluate_channel(source(), channel(), ["xx"], ["aa"])
    assert result["log_likelihood"] == pytest.approx(math.log(F(5, 32)), abs=1e-13)
    assert result["records"][0]["joint_log_probability"] == pytest.approx(math.log(F(1, 8)), abs=1e-13)
    assert result["records"][0]["plaintext"] == "a"
    assert result["edits"] == 1
    assert result["gold_characters"] == 2
    assert result["decoded_characters"] == 1
    assert result["exact_records"] == 0


def test_duplicate_latent_alternatives_do_not_change_joint_map_to_plaintext_map():
    model = channel([row("s", "a", [("s", "x", .5), ("s", "x", .5)]),
                     row("s", "b", [("s", "x", 1.)])])
    inferred = auditor.infer_record(source(("a", "b"), {"a": .6, "b": .4}), model, "x")
    assert inferred["log_likelihood"] == pytest.approx(math.log(F(1, 4)), abs=1e-13)
    assert inferred["joint_log_probability"] == pytest.approx(math.log(F(1, 10)), abs=1e-13)
    assert inferred["plaintext"] == "b"  # Summed plaintext mass instead favors a.


def test_empty_record_sums_initial_state_mass_and_full_rows_are_not_conditioned_on_match():
    model = channel([row("s", "a", [("s", "x", .25), ("s", "y", .75)]),
                     row("t", "a", [("t", "x", .25), ("t", "y", .75)])],
                    glyphs=("x", "y"), states=("s", "t"), initial={"s": .25, "t": .75})
    empty = auditor.infer_record(source(), model, "")
    assert empty["log_likelihood"] == pytest.approx(math.log(F(1, 2)))
    assert empty["joint_log_probability"] == pytest.approx(math.log(F(3, 8)))
    assert empty["plaintext"] == ""
    assert auditor.infer_record(source(), model, "x")["log_likelihood"] == pytest.approx(math.log(F(1, 16)))


def test_impossible_missing_and_null_cases_use_strict_json_and_literal_deletion_penalty():
    model = channel([row("s", "a", [("s", "xx", 1.)])])
    result = auditor.evaluate_channel(source(), model, ["x", "xx", "z"], ["aaa", "a", "aa"])
    assert result["log_likelihood"] is None
    assert result["edits"] == 5
    assert result["gold_characters"] == 6
    assert result["decoded_characters"] == result["exact_records"] == 1
    assert result["records"][0]["plaintext"] is None
    no_model = auditor.evaluate_channel(source(), None, ["x"], ["aa"])
    assert no_model["edits"] == 2 and no_model["decoded_characters"] == 0
    null = auditor.evaluate_channel(source(), model, ["xx"])
    assert null["edits"] is null["gold_characters"] is null["exact_records"] is None
    json.dumps([result, no_model, null], allow_nan=False)


def test_independent_stateful_log_dp_matches_fraction_path_enumeration():
    src = {"schema_version": 1, "alphabet": ["a", "b"], "order": 1,
           "probabilities": {"": {"a": .75, "b": .25}, "a": {"a": .25, "b": .75},
                             "b": {"a": .5, "b": .5}}}
    model = channel([row("s", "a", [("t", "x", .5), ("s", "xy", .5)]),
                     row("s", "b", [("s", "y", 1.)]),
                     row("t", "a", [("s", "xx", 1.)]),
                     row("t", "b", [("s", "x", .75), ("t", "y", .25)])],
                    glyphs=("x", "y"), states=("s", "t"), initial={"s": .25, "t": .75})
    alternatives = {(item["state"], item["letter"]): item["emissions"] for item in model["rows"]}
    paths = []

    def visit(text, plaintext, state, weight):
        paths.append((text, plaintext, weight * F(1, 2)))
        if len(text) >= 3:
            return
        context = plaintext[-1:]
        for letter in src["alphabet"]:
            for event in alternatives[state, letter]:
                if len(text + event["glyphs"]) <= 3:
                    visit(text + event["glyphs"], plaintext + letter, event["next_state"],
                          weight * F(1, 2) * F(src["probabilities"][context][letter]) * F(event["probability"]))

    for state, weight in model["initial"].items():
        visit("", "", state, F(weight))
    for length in range(4):
        for letters in itertools.product("xy", repeat=length):
            text = "".join(letters)
            matched = [path for path in paths if path[0] == text]
            mass = sum((path[2] for path in matched), F(0))
            result = auditor.infer_record(src, model, text)
            if not mass:
                assert result["plaintext"] is None
                continue
            best = max(path[2] for path in matched)
            assert result["log_likelihood"] == pytest.approx(math.log(mass), abs=1e-13)
            assert result["joint_log_probability"] == pytest.approx(math.log(best), abs=1e-13)
            assert any(path[1] == result["plaintext"] and path[2] == best for path in matched)


@pytest.mark.parametrize("left,right,expected", [("", "abc", 3), ("abc", "abc", 0), ("ab", "ba", 2),
                                               ("kitten", "sitting", 3), ("aaaa", "a", 3), ("aλ", "λa", 2)])
def test_exact_edit_distance_known_cases(left, right, expected):
    assert auditor.edit_distance(left, right) == auditor.edit_distance(right, left) == expected


def test_model_code_and_iid_baseline_have_independent_hand_computed_bits():
    context = {"source_alphabet": ["a"], "glyph_alphabet": ["x"], "denominator": 2,
               "max_states": 1, "max_emission_length": 2, "max_alternatives": 2,
               "source_count": 1, "stop_probability": .5}
    # One row-count bit plus two length bits, with no other non-singleton fields.
    assert auditor.model_bits(channel(), context) == 3
    baseline = auditor.baseline_from_fit(["xx", "xx"], ["x"])
    assert baseline["counts"] == [256]
    assert baseline["stop_count"] == 1365
    assert baseline["model_bits"] == 13
    assert baseline["model_code"] == "1" + format(1364, "012b")
    rho = F(1365, 4096)
    assert auditor.baseline_score(baseline, ["xx", "xx"]) == pytest.approx(math.log(rho**2 * (1 - rho)**4))
    balanced = auditor.baseline_from_fit(["xy"], ["x", "y"])
    assert balanced["counts"] == [128, 128]
    assert balanced["model_bits"] == 21
    assert balanced["model_code"].endswith("01111111")  # rank 127 among 255 positive compositions


def test_after_evaluation_guard_runs_before_any_artifact_read(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Read attempted before answer-access guard")

    monkeypatch.setattr(auditor, "_read", forbidden)
    with pytest.raises(PermissionError, match="after-evaluation"):
        auditor.audit(tmp_path, after_evaluation=False)
    with pytest.raises(SystemExit):
        auditor.main([])


def test_known_field_comparison_rejects_shape_value_type_and_tolerance_errors():
    assert auditor.compare_known({"x": [1., True, None]}, {"x": [1. + 1e-8, True, None], "baseline": 99}) < 1e-7
    for actual in ({"x": [1.1, True, None]}, {"x": [1., 1, None]}, {"x": [1., True]}, {}):
        with pytest.raises(AssertionError):
            auditor.compare_known({"x": [1., True, None]}, actual)


def toy_artifact_chain(tmp_path):
    """Write fabricated toy files only; never copy or read project artifacts."""
    def save(path, value, compressed=False):
        raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        if compressed:
            raw = gzip.compress(raw, mtime=0)
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        return {"path": path, "sha256": auditor.digest(raw), "bytes": len(raw)}

    src, model, name = source(), channel([row("s", "a", [("s", "x", 1.)])]), "toy"
    context = {"source_alphabet": ["a"], "glyph_alphabet": ["x"], "denominator": 2,
               "max_states": 1, "max_emission_length": 2, "max_alternatives": 2,
               "source_count": 1, "stop_probability": .5}
    records, truths = {"fit": ["x"], "transfer": ["xx"]}, {"fit": ["a"], "transfer": ["aa"]}
    source_spec = save("source.json", {"source_model": src})
    artifacts = {split: save(f"{split}.json", {"case_id": name, "split": split, "records": strings, "context": context})
                 for split, strings in records.items()}
    artifacts["answer"] = save("answer.json", {"case_id": name, "positive": True,
                                               "gold_channel": model, "plaintext": truths})
    baseline = auditor.baseline_from_fit(records["fit"], ["x"])
    score = {"model_bits": 2, "data_bits": 2., "total_bits": 4., "log_likelihood": math.log(.25)}
    full = {"channel": model, "score": score, "config": {"seed": 0}, "source_index": 0}
    full_spec = save("full.json.gz", full, compressed=True)
    freeze = {"case_id": name, "channel": model, "score": score, "config": full["config"],
              "source_freeze": "a" * 40, "input_sha256": artifacts["fit"]["sha256"],
              "source_sha256": source_spec["sha256"], "full_output": full_spec, "baseline": baseline}
    freeze_spec = save(f"results/{auditor.EXPERIMENT}/toy_freeze.json", freeze)
    manifest = {"source": source_spec, "cases": {name: {"family": "A", "positive": True, "artifacts": artifacts}}}
    manifest_spec = save("manifest.json", manifest)
    case = {"positive": True, "family": "A", "freeze_sha256": freeze_spec["sha256"],
            "model_bits_with_selector": 3, "baseline_model_bits": baseline["model_bits"], "splits": {}}
    for split in records:
        metrics = auditor.evaluate_channel(src, model, records[split], truths[split])
        case["splits"][split] = {"learned": metrics, "oracle": copy.deepcopy(metrics),
                                  "baseline_log_likelihood": auditor.baseline_score(baseline, records[split]),
                                  "glyph_characters": sum(map(len, records[split]))}
    case.update(fit_saving_vs_iid_bits=10., transfer_gain_vs_iid_bits_per_glyph=0., diagnostic_flag=False)
    full_evaluation = {"manifest_sha256": manifest_spec["sha256"], "source_sha256": source_spec["sha256"],
                       "cases": {name: case}}
    prediction_spec = save("predictions.json", full_evaluation)
    compact = copy.deepcopy(full_evaluation)
    for split in compact["cases"][name]["splits"].values():
        split["learned"].pop("records")
        split["oracle"].pop("records")
    compact["full_predictions"] = prediction_spec
    compact["diagnostic_summary"] = {"positives_cer_at_most_10pct": 1, "positives_flagged": 0, "nulls_flagged": 0}
    save("evaluation.json", compact)
    return artifacts


def test_complete_toy_artifact_chain_audits_without_real_data(tmp_path):
    toy_artifact_chain(tmp_path)
    result = auditor.audit(tmp_path, after_evaluation=True, manifest_path="manifest.json", evaluation_path="evaluation.json")
    assert result["status"] == "pass"
    assert result["cases_checked"] == 1 and result["record_evaluations_checked"] == 4
    assert result["maximum_numeric_delta"] < 1e-12


@pytest.mark.parametrize("path", ["fit.json", "answer.json", "full.json.gz", "predictions.json",
                                  f"results/{auditor.EXPERIMENT}/toy_freeze.json"])
def test_every_data_prediction_and_frozen_model_hash_is_enforced(tmp_path, path):
    toy_artifact_chain(tmp_path)
    target = tmp_path / path
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        auditor.audit(tmp_path, after_evaluation=True, manifest_path="manifest.json", evaluation_path="evaluation.json")


def test_wrong_prediction_is_rejected_even_with_consistent_recomputed_artifact_hash(tmp_path):
    toy_artifact_chain(tmp_path)
    prediction_path = tmp_path / "predictions.json"
    predictions = json.loads(prediction_path.read_text())
    predictions["cases"]["toy"]["splits"]["transfer"]["learned"]["records"][0]["plaintext"] = "wrong"
    raw = json.dumps(predictions, sort_keys=True).encode()
    prediction_path.write_bytes(raw)
    compact_path = tmp_path / "evaluation.json"
    compact = json.loads(compact_path.read_text())
    compact["full_predictions"].update(sha256=auditor.digest(raw), bytes=len(raw))
    compact_path.write_text(json.dumps(compact))
    with pytest.raises(AssertionError, match="plaintext"):
        auditor.audit(tmp_path, after_evaluation=True, manifest_path="manifest.json", evaluation_path="evaluation.json")


def test_production_and_independent_helpers_share_schema_on_tiny_fixtures():
    # Integration comparison only; rational fixtures above supply the actual
    # independent evidence. Production functions are never imported by auditor.
    from scripts.evaluate_blind_channel_dev001 import evaluate_channel as production
    from voynich.finite_state_channel import Channel, SourceModel

    src, model = source(), channel()
    observations, gold = ["", "x", "xx", "xxx", "z"], ["", "a", "aa", "aaa", "a"]
    expected = auditor.evaluate_channel(src, model, observations, gold)
    actual = production(SourceModel.from_dict(src), Channel.from_dict(model), observations, gold)
    assert auditor.compare_known(expected, actual) < 1e-12
