"""Artificial exact oracles and adversarial archive mutations for DEV004 audit."""
import ast
import copy
import gzip
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_blind_channel_dev004 as audit


def rational_source(order):
    alphabet = ("a", "b")
    rows = {}
    for depth in range(order + 1):
        for chars in itertools.product(alphabet, repeat=depth):
            context = "".join(chars)
            pa = Fraction(1 + context.count("a"), 3 + len(context))
            rows[context] = {"a": pa, "b": 1 - pa}
    raw = {"schema_version": 1, "alphabet": list(alphabet), "order": order,
           "probabilities": {h: {a: float(p) for a, p in row.items()} for h, row in rows.items()}}
    return raw, rows


def enumerate_readings(rows, order, units, record, rho=Fraction(1, 4)):
    result = []

    def visit(remaining, text, mass):
        if not remaining:
            result.append((text, mass * rho))
        else:
            context = text[-order:] if order else ""
            for char, unit in zip(("a", "b"), units, strict=True):
                if remaining.startswith(unit) and rows[context][char]:
                    visit(remaining[len(unit):], text + char, mass * (1 - rho) * rows[context][char])
    visit(record, "", Fraction(1))
    return result


@pytest.mark.parametrize("order", range(4))
def test_independent_suffix_inference_matches_fraction_enumeration_every_small_case(order):
    source, fractions = rational_source(order)
    for units in itertools.product(("x", "y", "xx", "xy"), repeat=2):
        for size in range(5):
            for glyphs in itertools.product("xy", repeat=size):
                record = "".join(glyphs)
                readings = enumerate_readings(fractions, order, units, record)
                actual = audit.infer_record(source, units, record, .25)
                if not readings:
                    assert actual["plaintext"] is actual["log_likelihood"] is actual["joint_log_probability"] is None
                    continue
                probability = sum(value for _, value in readings)
                best = max(value for _, value in readings)
                assert actual["log_likelihood"] == pytest.approx(math.log(float(probability)), abs=1e-12)
                assert actual["joint_log_probability"] == pytest.approx(math.log(float(best)), abs=1e-12)
                assert (actual["plaintext"], best) in readings
                assert audit.reading_log_probability(source, units, actual["plaintext"], record, .25) == pytest.approx(
                    math.log(float(best)), abs=1e-12)


def test_manual_estimation_uses_exact_suffix_interpolation_and_never_crosses_boundaries():
    source = audit.manual_source(["aab", "b"], ("a", "b"), 3, 2.)
    rows = source["probabilities"]
    assert rows[""] == {"a": .5, "b": .5}
    assert rows["a"] == {"a": .5, "b": .5}
    assert rows["aa"] == pytest.approx({"a": float(Fraction(1, 3)), "b": float(Fraction(2, 3))})
    assert rows["b"] == rows["ab"] == rows["aab"] == rows[""]
    assert rows["baa"] == rows["aa"]  # unseen context backs off completely
    assert len(rows) == 15
    audit.validate_source(source)
    expected = -math.log2(.5 * .5 * (2 / 3) * .5)
    assert audit.plaintext_bits(source, ["aab", "b"]) == pytest.approx(expected)
    assert audit.plaintext_bits(source, ["", "aab", "", "b"]) == pytest.approx(expected)


def test_contexts_are_full_histories_even_when_probability_rows_happen_to_equal():
    source, _ = rational_source(2)
    source["probabilities"]["a"] = source["probabilities"][""]
    source["probabilities"]["aa"] = {"a": 1., "b": 0.}
    result = audit.infer_record(source, ("x", "y"), "xxx", .5)
    assert result["plaintext"] == "aaa"
    assert result["log_likelihood"] == pytest.approx(math.log(.5 * (.5 / 3) ** 2 * .5))
    with pytest.raises(ValueError, match="zero posterior"):
        audit.reading_log_probability(source, ("x", "y"), "aab", "xxy", .5)
    with pytest.raises(RuntimeError, match="cap"):
        audit.infer_record(source, ("x", "x"), "xx", .5, max_nodes=1)


def test_vectorized_integer_edit_dp_equals_separate_scalar_recurrence_exhaustively():
    strings = ["".join(row) for size in range(4) for row in itertools.product(("a", "é", "🙂"), repeat=size)]
    for left in strings:
        expected = [audit.edit_distance(left, right) for right in strings]
        assert audit.edit_distances(left, strings) == expected
        assert expected == [audit.edit_distance(right, left) for right in strings]
    assert audit.edit_distances("abc", []) == []
    assert audit.edit_distances("", ["", "abc"]) == [0, 3]
    assert audit.edit_distances("a" * 300, ["a" * 300, "b" * 300, ""]) == [0, 300, 300]


def test_edit_floor_matches_all_exhaustively_compatible_plaintexts():
    source, fractions = rational_source(0)
    for units in itertools.product(("x", "y", "xx"), repeat=2):
        for size in range(5):
            for glyphs in itertools.product("xy", repeat=size):
                record = "".join(glyphs)
                readings = enumerate_readings(fractions, 0, units, record)
                for truth in ("", "a", "b", "aa", "aba"):
                    expected = min((audit.edit_distance(text, truth) for text, _ in readings), default=None)
                    assert audit.dictionary_edit_floor(source["alphabet"], units, record, truth) == expected
    with pytest.raises(ValueError, match="cell cap"):
        audit.dictionary_edit_floor(["a"], ("x",), "xxx", "aaa", max_cells=3)


def decision_fixture(source, units, record, seed, candidates=None, references=None):
    inferred = audit.infer_record(source, units, record, .25)
    map_text = inferred["plaintext"]
    candidates = [map_text] * 32 if candidates is None else candidates
    references = [map_text] * 256 if references is None else references
    texts = list(dict.fromkeys([map_text, *candidates]))
    risks = [{"plaintext": text, "total_edit_distance": sum(audit.edit_distance(text, ref) for ref in references)}
             for text in texts]
    for row in risks:
        row["mean_edit_distance"] = row["total_edit_distance"] / 256
    selected = min(risks, key=lambda row: row["total_edit_distance"])
    return {"status": "ok", "plaintext": selected["plaintext"], "map_plaintext": map_text,
            "log_likelihood": inferred["log_likelihood"], "map_log_probability": inferred["joint_log_probability"],
            "estimated_risk": selected["mean_edit_distance"], "map_estimated_risk": risks[0]["mean_edit_distance"],
            "candidate_draws": 32, "risk_draws": 256, "seed": seed,
            "candidate_seed": audit.digest(f"unit-channel-decision/v1:candidates:{seed}".encode()),
            "risk_seed": audit.digest(f"unit-channel-decision/v1:risk:{seed}".encode()),
            "candidate_samples": candidates, "risk_samples": references,
            "candidate_bank_sha256": audit.bank_hash(candidates), "risk_bank_sha256": audit.bank_hash(references),
            "candidate_risks": risks, "tie_break": "map_first_then_candidate_first_occurrence"}


def test_decision_risks_preserve_multiplicity_and_map_first_ties():
    source, _ = rational_source(0)
    source["probabilities"][""] = {"a": .8, "b": .2}
    units, record = ("x", "x"), "x"
    inference = audit.infer_record(source, units, record, .25)
    row = decision_fixture(source, units, record, 7, ["b", "a"] * 16, ["a"] * 64 + ["b"] * 192)
    assert row["plaintext"] == "b" and row["map_plaintext"] == "a"
    result = audit.audit_decision(row, source, units, record, .25, 7, inference)
    assert result["candidate_reference_pairs"] == 512 and result["distinct_distance_pairs_computed"] == 4
    assert not result["numerically_resampled"]
    tied = decision_fixture(source, units, record, 7, ["b", "a"] * 16, ["a", "b"] * 128)
    assert tied["plaintext"] == "a"
    audit.audit_decision(tied, source, units, record, .25, 7, inference)
    tied["plaintext"] = "b"
    with pytest.raises(ValueError, match="winner"):
        audit.audit_decision(tied, source, units, record, .25, 7, inference)


@pytest.mark.parametrize("mutation", ["hash", "seed", "count", "unsupported", "risk", "marginal", "candidate_order"])
def test_decision_archive_mutations_are_detected(mutation):
    source, _ = rational_source(0)
    units, record = ("x", "x"), "x"
    row = decision_fixture(source, units, record, 7, ["a", "b"] * 16, ["a", "b"] * 128)
    if mutation == "hash":
        row["risk_bank_sha256"] = "0" * 64
    elif mutation == "seed":
        row["candidate_seed"] = row["risk_seed"]
    elif mutation == "count":
        row["risk_samples"].pop()
    elif mutation == "unsupported":
        row["candidate_samples"][0] = "aa"
        row["candidate_bank_sha256"] = audit.bank_hash(row["candidate_samples"])
    elif mutation == "risk":
        row["candidate_risks"][0]["total_edit_distance"] += 1
    elif mutation == "marginal":
        row["log_likelihood"] += 1
    else:
        row["candidate_risks"].reverse()
    with pytest.raises(ValueError):
        audit.audit_decision(row, source, units, record, .25, 7, audit.infer_record(source, units, record, .25))


def complete_fixture():
    alphabet, bodies = ("a", "b"), {"caesar": ["ab", "ba"], "virgil": ["baab"]}
    payloads, corpus = {}, {"alphabet": list(alphabet), "sources": {}}
    for name, role in (("caesar", "P1"), ("virgil", "P2")):
        text = "".join(bodies[name])
        path = f"{name}.json"
        payloads[path] = {"text": text, "body_boundaries": [2, 4] if name == "caesar" else [4]}
        corpus["sources"][name] = {"role": role, "derived_path": path, "selected_text_sha256": audit.digest(text.encode())}
    candidates = [{"order": order, "tau": tau,
                   "validation_bits_per_character": audit.plaintext_bits(
                       audit.manual_source(bodies["caesar"], alphabet, order, tau), bodies["virgil"]) / 4}
                  for order, tau in audit.GRID]
    per_order = {str(order): min((row for row in candidates if row["order"] == order),
                                key=lambda row: row["validation_bits_per_character"]) for order in range(4)}
    selected = min(candidates, key=lambda row: row["validation_bits_per_character"])
    models = {f"order{order}": audit.manual_source(bodies["caesar"] + bodies["virgil"], alphabet, order,
                                                  per_order[str(order)]["tau"]) for order in range(4)}
    old = audit.manual_source(bodies["caesar"] + bodies["virgil"], alphabet, 1, 64.)
    payloads["models.gz"], payloads["old.json"] = {"models": models}, {"source_model": old}
    old_artifact = {"path": "old.json", "sha256": "oldhash"}
    selection = {"source_authors_accessed": ["caesar", "virgil"], "source_characters_per_author": 4,
                 "candidates": candidates, "selected": selected, "selected_per_order": per_order,
                 "primary_model": f"order{selected['order']}", "models": {"path": "models.gz"},
                 "old_source": old_artifact, "old_source_regression_max_delta": 0.}
    context = {"source_alphabet": list(alphabet), "glyph_alphabet": ["x", "y"], "denominator": 3,
               "max_states": 2, "max_emission_length": 2, "max_alternatives": 3, "source_count": 1,
               "stop_probability": .25}
    channel = {"schema_version": 1, "states": ["s"], "initial": {"s": 1.}, "glyph_alphabet": ["x", "y"],
               "stop_probability": .25, "max_emission_length": 2,
               "rows": [{"state": "s", "letter": char,
                         "emissions": [{"next_state": "s", "glyphs": unit, "probability": 1.}]}
                        for char, unit in zip(alphabet, ("x", "y"), strict=True)]}
    manifest, predictions, evaluation, freezes = {"source": old_artifact, "cases": {}}, {"cases": {}}, {"cases": {}}, {}
    records = {"fit": ["xy", "yx", "", "xx"], "transfer": ["yy", "xyx"]}
    texts = {key: [row.translate(str.maketrans("xy", "ab")) for row in value] for key, value in records.items()}
    for case_index, name in enumerate(audit.CASE_NAMES):
        positive = "shuffle" not in name
        case = {"positive": positive, "artifacts": {}}
        for split in ("fit", "transfer", "answer"):
            path = f"{name}-{split}.json"
            case["artifacts"][split] = {"path": path, "sha256": f"{name}-{split}-hash"}
            payloads[path] = ({"plaintext": texts, "gold_channel": channel} if split == "answer" else
                              {"records": records[split], "context": context})
        manifest["cases"][name] = case
        freezes[name] = {"source_sha256": "oldhash", "input_sha256": case["artifacts"]["fit"]["sha256"],
                         "channel": channel}
        predicted, measured = {"positive": positive, "arms": {}}, {"positive": positive, "arms": {}}
        for arm in (("learned", "oracle") if positive else ("learned",)):
            bits = audit.literal_model_bits(("x", "y"), context) + 1
            report = {"model_bits_with_selector": bits, "sources": {}, "decisions": {}}
            scored = {"model_bits_with_selector": bits, "sources": {}, "decisions": {}}
            if positive:
                scored["dictionary_edit_floor"] = {}
            for source_name, source in {"old": old, **models}.items():
                report["sources"][source_name], scored["sources"][source_name] = {}, {}
                for split, observations in records.items():
                    rows = [audit.infer_record(source, ("x", "y"), record, .25) for record in observations]
                    report["sources"][source_name][split] = rows
                    metrics = audit._metrics(rows, texts[split] if positive else None)
                    likelihood = math.fsum(row["log_likelihood"] for row in rows)
                    metrics |= {"log_likelihood": likelihood, "data_bits": -likelihood / math.log(2),
                                "total_bits": bits - likelihood / math.log(2),
                                "map_surprisal_bits": [(row["log_likelihood"] - row["joint_log_probability"]) / math.log(2)
                                                       for row in rows]}
                    scored["sources"][source_name][split] = metrics
            for split_index, (split, observations) in enumerate(records.items()):
                decisions = [decision_fixture(old, ("x", "y"), record,
                             61101 + 1000 * case_index + 100 * split_index + index + (10000 if arm == "oracle" else 0))
                             for index, record in enumerate(observations)]
                report["decisions"][split] = decisions
                scored["decisions"][split] = audit._metrics(decisions, texts[split] if positive else None)
                if positive:
                    scored["dictionary_edit_floor"][split] = {"record_edits": [0] * len(observations), "edits": 0}
            predicted["arms"][arm], measured["arms"][arm] = report, scored
        predictions["cases"][name], evaluation["cases"][name] = predicted, measured
    return manifest, corpus, selection, predictions, evaluation, payloads, freezes


def test_full_artificial_panel_replays_exact_expected_inventory_and_costs():
    result = audit.audit_payloads(*complete_fixture(), expected_author_characters=4)
    assert result["status"] == "pass"
    assert result["source_selection"]["candidates_checked"] == 19
    assert result["source_record_predictions_checked"] == 180
    assert result["literal_MAP_agreements"] == 180
    assert result["decision_records_checked"] == 36
    assert result["dictionary_edit_floor_records_checked"] == 24
    assert result["bank_draws_checked"] == 36 * 288
    assert not result["posterior_banks_numerically_resampled"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("mutation", ["source_grid", "source_score", "selection", "refit", "old",
                                   "source_boundary", "case", "channel", "row", "cost", "MAP", "likelihood",
                                   "edits", "surprisal", "floor", "null_floor", "null_accuracy", "decision"])
def test_full_panel_rejects_adversarial_archive_changes(mutation):
    arguments = copy.deepcopy(complete_fixture())
    manifest, corpus, selection, predictions, evaluation, payloads, freezes = arguments
    report = predictions["cases"]["B-key1"]["arms"]["learned"]
    measured = evaluation["cases"]["B-key1"]["arms"]["learned"]
    if mutation == "source_grid":
        selection["candidates"].pop()
    elif mutation == "source_score":
        selection["candidates"][0]["validation_bits_per_character"] += .1
    elif mutation == "selection":
        selection["primary_model"] = "wrong"
    elif mutation == "refit":
        payloads["models.gz"]["models"]["order3"]["probabilities"]["aaa"]["a"] += .01
    elif mutation == "old":
        payloads["old.json"]["source_model"]["order"] = 0
    elif mutation == "source_boundary":
        payloads["caesar.json"]["body_boundaries"] = [3, 4]
    elif mutation == "case":
        predictions["cases"].pop("B-key2-shuffle")
    elif mutation == "channel":
        freezes["B-key1"]["channel"]["rows"][0]["emissions"][0]["glyphs"] = "xx"
    elif mutation == "row":
        report["sources"]["order3"]["fit"].pop()
    elif mutation == "cost":
        report["model_bits_with_selector"] += 1
    elif mutation == "MAP":
        report["sources"]["order3"]["fit"][0]["plaintext"] = "ba"
    elif mutation == "likelihood":
        report["sources"]["order3"]["fit"][0]["log_likelihood"] += 1
    elif mutation == "edits":
        measured["sources"]["old"]["transfer"]["edits"] += 1
    elif mutation == "surprisal":
        measured["sources"]["old"]["transfer"]["map_surprisal_bits"][0] += 1
    elif mutation == "floor":
        measured["dictionary_edit_floor"]["fit"]["record_edits"][0] = 1
    elif mutation == "null_floor":
        evaluation["cases"]["B-key1-shuffle"]["arms"]["learned"]["dictionary_edit_floor"] = {}
    elif mutation == "null_accuracy":
        evaluation["cases"]["B-key1-shuffle"]["arms"]["learned"]["sources"]["old"]["fit"]["edits"] = 0
    else:
        report["decisions"]["fit"][0]["candidate_bank_sha256"] = "bad"
    with pytest.raises(ValueError):
        audit.audit_payloads(*arguments, expected_author_characters=4)


def test_hash_and_size_verified_before_json_or_gzip_decode(tmp_path):
    path = tmp_path / "fixture.json.gz"
    path.write_bytes(b"not gzip")
    with pytest.raises(ValueError, match="hash/size"):
        audit.verified_json(tmp_path, {"path": path.name, "sha256": "wrong"})
    raw = gzip.compress(b'{"ok":true}', mtime=0)
    path.write_bytes(raw)
    spec = {"path": path.name, "sha256": audit.digest(raw), "bytes": len(raw)}
    assert audit.verified_json(tmp_path, spec) == {"ok": True}
    with pytest.raises(ValueError, match="hash/size"):
        audit.verified_json(tmp_path, spec | {"bytes": len(raw) + 1})
    with pytest.raises(ValueError, match="escapes"):
        audit.verified_json(tmp_path, {"path": "../outside.json", "sha256": "wrong"})


def test_cli_gating_and_no_overwrite_happen_before_artifact_access(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(audit, "run_audit", lambda *args: called.append(args))
    with pytest.raises(SystemExit):
        audit.main(["--freeze", "unread"])
    assert called == []
    monkeypatch.undo()
    destination = tmp_path / "old.json"
    destination.write_text("preserved")
    monkeypatch.setattr(audit.subprocess, "run", lambda *args, **kwargs: pytest.fail("read frozen data"))
    with pytest.raises(FileExistsError):
        audit.run_audit(tmp_path, "unread", "old.json")
    assert destination.read_text() == "preserved"


def test_auditor_does_not_import_production_or_other_auditor_algorithms():
    tree = ast.parse(Path(audit.__file__).read_text())
    imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(module and module.startswith(("voynich", "scripts")) for module in imported)
