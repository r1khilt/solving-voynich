"""Artifact isolation and scoring invariants for the fixed-channel diagnostic."""
import gzip
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_blind_channel_dev004 as runner
from voynich.finite_state_channel import Channel, Emission


def test_archive_roundtrip_hash_and_exclusive_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = tmp_path / "predictions.json.gz"
    artifact = runner.save_new(path, {"predictions": ["ab", "a"]}, compressed=True)
    assert runner.load_archive(artifact) == {"predictions": ["ab", "a"]}
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        runner.save_new(path, {"new": "replacement"}, compressed=True)
    assert path.read_bytes() == original
    path.write_bytes(gzip.compress(b'{"predictions": ["wrong"]}', mtime=0))
    with pytest.raises(ValueError, match="identity"):
        runner.load_archive(artifact)


def test_metrics_keep_insertions_nulls_and_unavailable_distinct():
    rows = [{"plaintext": "abcde"}, {"plaintext": None}, {"plaintext": ""}]
    result = runner.metrics(rows, ["a", "bb", ""])
    assert result["edits"] == 6
    assert result["record_edits"] == [4, 2, 0]
    assert result["gold_characters"] == 3
    assert result["exact_records"] == 1
    assert result["supported_records"] == 2
    assert runner.metrics(rows, None)["edits"] is None
    with pytest.raises(ValueError, match="mismatch"):
        runner.metrics(rows, ["a"])


def test_deterministic_extraction_preserves_duplicate_units_and_letter_order():
    channel = Channel(("s",), ("X", "Y"), {"s": 1.}, {
        ("s", "a"): (Emission("s", "XY", 1.),),
        ("s", "b"): (Emission("s", "XY", 1.),),
    }, .2, 2)
    assert runner.get_units(channel, ("b", "a")) == ("XY", "XY")
    stochastic = Channel(("s",), ("X", "Y"), {"s": 1.}, {
        ("s", "a"): (Emission("s", "X", .5), Emission("s", "Y", .5)),
    }, .2, 2)
    with pytest.raises(ValueError, match="deterministic"):
        runner.get_units(stochastic, ("a",))


def test_serialization_rejects_nonfinite_without_creating_file(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        runner.save_new(path, {"score": float("nan")})
    assert not path.exists()


def test_fixed_inputs_requires_manifest_content_not_only_git_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = tmp_path / runner.MANIFEST
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"source": {"path": "source.json"}}))
    checked = []
    monkeypatch.setattr(runner, "require_frozen", lambda commit, paths: checked.extend(paths))
    with pytest.raises(ValueError, match="manifest changed"):
        runner.fixed_inputs("testcommit")
    assert runner.CORPUS in checked
    assert all(f"results/BLIND-CHANNEL-DEV-003/{name}_freeze.json" in checked
               for name in runner.CASE_NAMES)


def test_source_selection_never_opens_cipher_author_or_answers_and_keeps_stable_ties(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    corpus_path = tmp_path / runner.CORPUS
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text(json.dumps({"alphabet": ["a"], "sources": {
        name: {"derived_path": name, "derived_sha256": "test"}
        for name in ("caesar", "virgil", "cicero")}}))
    manifest = {"source": {"path": "old", "sha256": "test"}}
    monkeypatch.setattr(runner, "fixed_inputs", lambda _: manifest)
    opened = []

    def checked(artifact):
        opened.append(artifact["path"])
        if artifact["path"] in ("caesar", "virgil"):
            return {"text": "a" * 50_000, "body_boundaries": [25_000, 50_000]}
        if artifact["path"] == "old":
            return {"source_model": {"schema_version": 1, "alphabet": ["a"], "order": 1,
                                     "probabilities": {"": {"a": 1.}, "a": {"a": 1.}}}}
        raise AssertionError("Cipher author or answer access during source selection")

    monkeypatch.setattr(runner, "checked_artifact", checked)
    runner.select("toy-freeze")
    assert opened == ["caesar", "virgil", "old"]
    result = json.loads((tmp_path / runner.SELECTION).read_text())
    assert len(result["candidates"]) == 19
    assert result["primary_model"] == "order0"
    assert all(result["selected_per_order"][str(order)]["tau"] == .25 for order in (1, 2, 3))
    assert result["old_source_regression_max_delta"] == 0.
    with pytest.raises(FileExistsError):
        runner.select("toy-freeze")
