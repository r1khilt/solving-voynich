from collections import Counter
import copy
import json

import pytest

from scripts import build_blind_channel_confirm001 as builder
from scripts import run_blind_channel_confirm001 as runner


def test_eight_new_keys_preserve_every_source_letter_and_null_histogram():
    alphabet = tuple("abcdefghiklmnopqrstuxyz")

    def windows(_payload, role, key):
        text = "".join(alphabet[(i + key + (5 if role == "transfer" else 0)) % 23] for i in range(224))
        return [{"text": text, "start": 4096 * key + n * 256,
                 "end": 4096 * key + n * 256 + 224, "body_index": 0}
                for n in range(4 if role == "fit" else 2)]

    panel = builder.construct({"sallust": {}, "tacitus": {}}, alphabet, windows)
    assert set(panel) == set(runner.CASE_NAMES)
    signatures = []
    for k in range(1, 9):
        positive, null = panel[f"B-key{k}"], panel[f"B-key{k}-shuffle"]
        rows = positive["answer"]["gold_channel"]["rows"]
        units = {row["letter"]: row["emissions"][0]["glyphs"] for row in rows}
        signatures.append(tuple(units[a] for a in alphabet))
        assert len(set(units.values())) == 23
        assert sum(len(unit) == 1 for unit in units.values()) == 6
        for role in ("fit", "transfer"):
            for text, cipher, shuffled in zip(positive["answer"]["plaintext"][role],
                                               positive["records"][role], null["records"][role], strict=True):
                assert cipher == "".join(units[a] for a in text)
                assert Counter(cipher) == Counter(shuffled)
        assert null["answer"]["plaintext"] is None and null["answer"]["gold_channel"] is None
        assert positive["search_seed"] != null["search_seed"]
    assert len(set(signatures)) == 8
    assert panel == builder.construct({"sallust": {}, "tacitus": {}}, alphabet, windows)


def test_builder_does_not_replace_invalid_windows():
    with pytest.raises(ValueError, match="schedule"):
        builder.construct({"sallust": {}, "tacitus": {}}, tuple("abcdefghiklmnopqrstuxyz"),
                          lambda *_: [{"text": "abc", "start": 0, "end": 3, "body_index": 0}])


def passing_cases():
    return {name: {**{arm: {"transfer": {"edits": edits, "gold_characters": 448}}
                     for arm, edits in (("learned", 8), ("oracle", 0), ("frequency", 400))},
                   "iid_diagnostic": {"flag": not name.endswith("-shuffle")}}
            for name in runner.CASE_NAMES}


def test_preregistered_gates_are_key_weighted_and_failure_is_not_dropped():
    cases = passing_cases()
    decision = runner.decision_gates(cases, True)
    assert decision["recovery_pass"] and decision["iid_screen_pass"]
    assert not decision["semantic_rejection_qualified"]
    assert decision["macro_transfer_cer"] == pytest.approx(8 / 448)
    assert not runner.decision_gates(cases, False)["recovery_pass"]
    cases["B-key8"]["learned"]["transfer"]["edits"] = 9
    assert not runner.decision_gates(cases, True)["recovery_gates"]["each_oracle_excess_at_most_002"]
    cases["B-key8"]["learned"]["transfer"]["edits"] = 448
    assert not runner.decision_gates(cases, False)["recovery_pass"]


def test_frequency_zero_is_not_a_division_error_or_automatic_success():
    cases = passing_cases()
    for case in cases.values():
        for arm in ("learned", "oracle", "frequency"):
            case[arm]["transfer"]["edits"] = 0
    result = runner.decision_gates(cases, True)
    assert result["relative_frequency_reduction"] is None
    assert not result["recovery_gates"]["each_beats_frequency"]


def test_null_flags_are_separate_from_plaintext_recovery():
    cases = copy.deepcopy(passing_cases())
    cases["B-key1-shuffle"]["iid_diagnostic"]["flag"] = True
    result = runner.decision_gates(cases, True)
    assert result["recovery_pass"] and not result["iid_screen_pass"]


def test_exclusive_outputs_validate_path_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = tmp_path / "result.json"
    artifact = runner.write_new(path, {"fixed": True})
    assert artifact["sha256"] == runner.digest(path.read_bytes())
    assert json.loads(path.read_text()) == {"fixed": True}
    with pytest.raises(FileExistsError):
        runner.write_new(path, {"fixed": False})
    outside = tmp_path.parent / "must-not-be-written-confirm001.json"
    with pytest.raises(ValueError):
        runner.write_new(outside, {})
    assert not outside.exists()


def test_attempted_campaign_is_not_relaunched(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "inputs", lambda _: ({"pipeline_freeze": "fixture"},))
    path = tmp_path / f"results/{runner.EXPERIMENT}/started.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    monkeypatch.setattr(runner.subprocess, "run", lambda *_a, **_k: pytest.fail("Cannot relaunch"))
    with pytest.raises(FileExistsError):
        runner.campaign("fixture")
