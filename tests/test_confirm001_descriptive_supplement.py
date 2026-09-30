import gzip
import hashlib
import json
import math

import pytest

from scripts import summarize_blind_channel_confirm001 as report


def test_unfinished_evaluation_cannot_open_answers(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "artifact", lambda *_: pytest.fail("Answers opened early"))
    with pytest.raises(FileNotFoundError):
        report.summarize(tmp_path)


def test_missing_key_does_not_become_23_correct_rows():
    result = report.key_diagnostic(None, tuple(report.ALPHABET),
                                   {"fit": ["aaa"], "transfer": ["bbb"]})
    assert result["literal_rows_correct"] is None
    assert result["incorrect_row_occurrences"] == {"fit": None, "transfer": None}
    assert all(row["correct"] is None for row in result["rows"])


def test_unobserved_wrong_row_and_exposure_are_kept_separate():
    gold = tuple(report.ALPHABET)
    learned = ("WRONG", *gold[1:-1], "ALSO WRONG")
    result = report.key_diagnostic(learned, gold, {"fit": ["aaab"], "transfer": ["abbb"]})
    assert result["literal_rows_correct"] == 21
    assert result["incorrect_row_occurrences"] == {"fit": 3, "transfer": 1}
    assert result["rows"][-1] == {"letter": "z", "correct": False,
                                  "fit_occurrences": 0, "transfer_occurrences": 0}


def test_map_uncertainty_is_one_bit_for_half_posterior_and_missing_stays_missing():
    result = report.posterior_surprisal([
        {"log_likelihood": math.log(.5), "joint_log_probability": math.log(.25)},
        {"log_likelihood": None, "joint_log_probability": None}])
    assert result == [1., None]
    with pytest.raises(ValueError, match="marginal"):
        report.posterior_surprisal([{"log_likelihood": -4., "joint_log_probability": -3.}])


@pytest.mark.parametrize("compressed", [False, True])
def test_archive_integrity_checks_raw_compressed_bytes(tmp_path, compressed):
    raw = json.dumps({"x": [1, 2]}).encode()
    if compressed:
        raw = gzip.compress(raw)
    name = "value.json.gz" if compressed else "value.json"
    (tmp_path / name).write_bytes(raw)
    spec = {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    assert report.artifact(tmp_path, spec) == {"x": [1, 2]}
    with pytest.raises(ValueError, match="size"):
        report.artifact(tmp_path, {**spec, "bytes": 0})
    (tmp_path / name).write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="hash"):
        report.artifact(tmp_path, spec)
