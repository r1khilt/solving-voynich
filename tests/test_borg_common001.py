"""Artificial source/key fixtures: no historical answer or real Borg fit."""
from collections import Counter
import itertools
import json
import math

import numpy as np
import pytest

from scripts.borg_control_parser import parse_bytes
from scripts import run_borg_common001 as producer
from voynich.borg_common import (COMMON, LATIN, Objective, decode, digest,
                                encode_records, fit, select_body, train_source)


def test_selection_resets_and_quarantines_whole_exception_chunks():
    raw = (b"#page 20r\n0124 5689\nMMMM[NOTE]mmmm\n0000.1111 2222\n"
           b"4444?6666\n8888\n#page 20v\n9999\n")
    rows = select_body(raw, parse_bytes(raw), leaves=range(20, 21))
    assert [row["symbols"] for row in rows] == ["01245689", "MMMM", "mmmm", "2222", "8888", "9999"]
    assert len(rows[0]["pieces"]) == 2
    for row in rows:
        assert row["symbols_sha256"] == digest(row["symbols"].encode())
        for piece in row["pieces"]:
            actual = raw[piece["start_byte"]:piece["end_byte"]]
            assert actual.decode() == piece["symbols"]
            assert digest(actual) == piece["raw_sha256"]
    excluded = select_body(raw, parse_bytes(raw), leaves=range(20, 21), excluded_lines={2})
    assert not any(row["line"] == 2 for row in excluded)


def test_selection_fixed_leaf_and_cleartext_exclusion():
    raw = (b"#page 10r\n0000\n#page 20r\n1111\n#page 49v\n2222\n"
           b"#page 99r\n4444\n#pahe 150r.01\n6666\n#page 21v.01\n8888\n"
           b"#page 22r\n<CLEARTEXT-LA>\n9999\n#page 120r\nmmmm\n")
    parsed = parse_bytes(raw)
    rows = select_body(raw, parsed, leaves=range(1, 161))
    assert [(row["leaf"], row["symbols"]) for row in rows] == [(20, "1111"), (120, "mmmm")]
    with pytest.raises(ValueError):
        select_body(raw + b"x", parsed, leaves=range(20, 21))


def literal_score(records, key, tables, source_size):
    total = 0.
    for record in records:
        decoded = [int(key[int(value)]) for value in record]
        for index in range(len(decoded)):
            order = min(index + 1, 4)
            code = 0
            for letter in decoded[index-order+1:index+1]:
                code = code * source_size + letter
            total += tables[order-1][code]
    return total


def independent_tables(text, alphabet):
    answer = []
    for order in range(1, 5):
        counts = Counter(text[i:i+order] for i in range(len(text)-order+1))
        row = []
        for letters in itertools.product(alphabet, repeat=order):
            gram = "".join(letters)
            context = gram[:-1]
            total = sum(counts[context+letter] for letter in alphabet)
            if order == 1:
                probability = (counts[gram] + .1) / (total + .1 * len(alphabet))
            else:
                lower_code = 0
                for letter in gram[1:]:
                    lower_code = lower_code * len(alphabet) + alphabet.index(letter)
                probability = (counts[gram] + 16 * math.exp(answer[-1][lower_code])) / (total + 16)
            row.append(math.log(probability))
        answer.append(np.asarray(row))
    return answer


def test_normalized_source_and_all_completed_keys_match_literal_arithmetic():
    alphabet, source = "abc", "aabacabbaacbcac"
    tables = train_source(source, alphabet)
    manual = independent_tables(source, alphabet)
    for compiled, direct in zip(tables, manual, strict=True):
        np.testing.assert_allclose(compiled, direct, atol=1e-13, rtol=0)
        np.testing.assert_allclose(np.exp(compiled).reshape(-1, 3).sum(axis=1), 1, atol=1e-13, rtol=0)
    records = encode_records(["a", "bb", "aba", "abaab", "bababb"], "ab")
    objective = Objective(records, tables, source_size=3, observed_size=2)
    keys = np.asarray(list(itertools.permutations(range(3))))
    np.testing.assert_allclose(objective.scores(keys),
        [literal_score(records, key, manual, 3) for key in keys], atol=1e-12, rtol=0)
    assert objective.characters == 17
    # A reset at every record is observable, not an implementation-only check.
    joined = encode_records(["".join(["a", "bb", "aba", "abaab", "bababb"])], "ab")
    assert abs(literal_score(joined, keys[0], manual, 3) - literal_score(records, keys[0], manual, 3)) > .01


@pytest.mark.parametrize("key", [[0., 1., 2.], [False, True, 2], [0, 0, 2], [0, 1], [0, 1, 3]])
def test_invalid_keys_cannot_be_coerced(key):
    objective = Objective(encode_records(["abba"], "ab"), train_source("aababbcc", "abc"), 3, 2)
    with pytest.raises(ValueError):
        objective.scores([key])
    with pytest.raises(ValueError):
        decode(["abba"], key, observed="ab", source="abc")


def test_finite_optimizer_is_deterministic_and_keeps_one_shared_key():
    source = "aaaabbcaaaabccaaaabbcaaaabbc"
    records = encode_records(["bbbbaac", "bbbaac", "bbbbaac"], "abc")
    objective = Objective(records, train_source(source, "abc"), 3, 3)
    calls = []
    first = fit(objective, seed=99271, restarts=4, sweeps=3, guard=lambda: calls.append(True))
    second = fit(objective, seed=99271, restarts=4, sweeps=3)
    assert first == second and calls
    assert len(first["endpoints"]) == 4
    assert all(1 <= row["sweeps"] <= 3 for row in first["endpoints"])
    assert first["score"] == max(row["score"] for row in first["endpoints"])
    assert first["score"] == pytest.approx(literal_score(records, first["key"], objective.tables, 3))
    for row in first["endpoints"]:
        assert sorted(row["key"]) == [0, 1, 2]
    assert first["full_key_scores"] <= 4 * (2 + 3 * 3 * 3)
    with pytest.raises(ValueError):
        fit(objective, seed=1, restarts=0)


def test_controls_censor_without_joining_and_split_without_letter_loss(monkeypatch):
    monkeypatch.setattr(producer, "CONTROL_FIT", 4)
    monkeypatch.setattr(producer, "CONTROL_TRANSFER", 4)
    records = producer.control_records([{"text": "akabycczdda"}], "abcd")
    assert records == ["acab", "cc", "dd"]
    left, right = producer.split_records(records, 5)
    assert left == ["acab", "c"] and right == ["c", "dd"]
    assert sum(map(len, left)) == 5 and sum(map(len, right)) == 3


def test_bound_outputs_are_exclusive_and_detect_byte_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(producer, "ROOT", tmp_path)
    path = tmp_path / "data.json.gz"
    entry = producer.save_new(path, {"value": [1, 2]}, compressed=True)
    assert producer.load_bound(entry, compressed=True) == {"value": [1, 2]}
    with pytest.raises(FileExistsError):
        producer.save_new(path, {})
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="changed"):
        producer.load_bound(entry, compressed=True)
    with pytest.raises(ValueError, match="escapes"):
        producer.load_bound({"path": "../escape", "sha256": "", "bytes": 0})


def controller_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(producer, "ROOT", tmp_path)
    monkeypatch.setattr(producer, "OUT", tmp_path / "results")
    monkeypatch.setattr(producer, "BULK", tmp_path / "outputs")
    monkeypatch.setattr(producer, "PREP", tmp_path / "results" / "prepare.json")
    monkeypatch.setattr(producer, "PATHS", ("dependency",))
    (tmp_path / "dependency").write_text("artificial fixture\n")
    controls = {name: {"fit": ["01245689", "Mcdhimno"], "transfer": ["01245689"],
        "kind": "positive" if name.startswith("control") else "paired_row_shuffle",
        "search_seed": index} for index, name in enumerate(("control-0", "control-1", "shuffle-0", "shuffle-1"))}
    inputs = {"controls": controls, "historical": {"fit": [{"symbols": "01245689"}],
                                                   "transfer": [{"symbols": "Mcdhimno"}]}}
    gold = {name: {"plain_transfer": decode(["01245689"], list(range(23))), "key": list(range(23))}
            for name in ("control-0", "control-1")}
    input_entry = producer.save_new(tmp_path / "outputs" / "inputs.json.gz", inputs, compressed=True)
    gold_entry = producer.save_new(tmp_path / "outputs" / "gold.json.gz", gold, compressed=True)
    sources = {name: {"text": LATIN * 4, "entry": {"fixture": name}} for name in ("latin", "english")}
    producer.save_new(producer.PREP, {"inputs": input_entry, "control_gold": gold_entry,
        "source_files": {name: row["entry"] for name, row in sources.items()},
        "preparation_source_hashes": {"dependency": digest((tmp_path / "dependency").read_bytes())}})
    monkeypatch.setattr(producer, "source_inputs", lambda: sources)
    monkeypatch.setattr(producer.resource, "setrlimit", lambda *_: None)
    monkeypatch.setattr(producer.signal, "alarm", lambda *_: None)
    monkeypatch.setattr(producer.subprocess, "check_output", lambda args, **_:
                        "fixturefreeze\n" if args[1] == "rev-parse" else
                        (tmp_path / args[-1].split(":", 1)[1]).read_bytes())
    calls = []

    def artificial_fit(objective, *, seed, restarts, sweeps, guard):
        guard()
        key = list(range(23))
        calls.append(seed)
        return {"key": key, "score": float(objective.scores(np.asarray(key)[None])[0]),
                "seed": seed, "restarts": restarts, "max_sweeps": sweeps, "endpoints": [], "full_key_scores": 1}

    monkeypatch.setattr(producer, "fit", artificial_fit)
    return inputs, calls


def test_producer_seals_all_seven_keys_before_any_control_answer_read(tmp_path, monkeypatch):
    _, calls = controller_fixture(tmp_path, monkeypatch)
    original_load = producer.load_bound
    observed_gold = []

    def check_seal(entry, compressed=False):
        if entry["path"].endswith("gold.json.gz"):
            sealed = json.loads((producer.OUT / "sealed_predictions.json").read_text())
            assert len(sealed["fits"]) == 7 and sealed["historical_answer_opened"] is False
            observed_gold.append(True)
        return original_load(entry, compressed)

    monkeypatch.setattr(producer, "load_bound", check_seal)
    producer.run("fixturefreeze")
    assert len(calls) == 7 and observed_gold == [True]
    result = json.loads((producer.OUT / "result.json").read_text())
    assert result["control_competence"] == "PASS"
    assert result["historical_recovery"] == "UNASSESSED_NO_EXTERNAL_ANSWER"
    assert set(result["control_metrics"]) == {"control-0", "control-1"}
    with pytest.raises(FileExistsError):
        producer.run("fixturefreeze")


def test_empty_control_allocation_fails_before_any_fit(tmp_path, monkeypatch):
    inputs, calls = controller_fixture(tmp_path, monkeypatch)
    inputs["controls"] = {}
    original_load = producer.load_bound
    monkeypatch.setattr(producer, "load_bound", lambda entry, compressed=False:
                        inputs if entry["path"].endswith("inputs.json.gz") else original_load(entry, compressed))
    with pytest.raises(ValueError, match="control allocation"):
        producer.run("fixturefreeze")
    assert not calls and not (producer.OUT / "sealed_predictions.json").exists()
    assert json.loads((producer.OUT / "failure.json").read_text())["no_retry_or_extension"] is True


def test_common_names_are_not_silently_normalized():
    assert len(COMMON) == 21 and len(LATIN) == 23 and COMMON.count("M") == COMMON.count("m") == 1
    assert encode_records(["Mm"], COMMON)[0].tolist() == [COMMON.index("M"), COMMON.index("m")]
