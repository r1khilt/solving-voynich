import hashlib
import json
from pathlib import Path

import pytest

from voynich.communication.__main__ import main
from voynich.communication.adapters import export_manuscript
from voynich.communication.schema import Observation


def corpus(root):
    root.mkdir()
    for split in ("train", "validation", "test"):
        data = {
            "page_id": split + "-folio",
            "leaf_id": split + "-leaf",
            "split": split,
            "text": "abcdef abc\n",
        }
        (root / f"{split}.jsonl").write_text(json.dumps(data) + "\n")
    (root / "tokenizer.json").write_text("{}")
    manifest = {
        "derived_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    }
    (root / "preparation.json").write_text(json.dumps(manifest))
    return root


def test_adapter_preserves_every_unit_and_leaf_provenance(tmp_path, monkeypatch):
    data = corpus(tmp_path / "data")
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        if path.name == "test.jsonl":
            raise AssertionError("Final text must not be parsed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    report = export_manuscript(data, tmp_path / "out", window=4)
    records = [
        Observation.from_dict(json.loads(s))
        for s in (tmp_path / "out/observations.jsonl").read_text().splitlines()
    ]
    text = "".join(report["symbol_inventory"][symbol - 2] for r in records for symbol in r.symbols)
    assert text == "abcdef abc\n"
    assert all(r.source_group == "manuscript-leaf:validation-leaf" for r in records)
    assert all(r.split == "validation" for r in records)
    with pytest.raises(ValueError, match="sealed"):
        export_manuscript(data, tmp_path / "test-out", split="test")


def test_adapter_detects_drift_and_refuses_overwrite(tmp_path):
    data = corpus(tmp_path / "data")
    export_manuscript(data, tmp_path / "out")
    with pytest.raises(ValueError):
        export_manuscript(data, tmp_path / "out")
    (data / "validation.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="differs"):
        export_manuscript(data, tmp_path / "new")


def test_cli_generation_separates_inputs_from_hidden_audit(tmp_path, capsys):
    destination = tmp_path / "generated"
    result = main(["generate", "--output", str(destination), "--count", "3", "--anchors", "2"])
    assert result["count"] == 3
    observations = [json.loads(s) for s in (destination / "observations.jsonl").read_text().splitlines()]
    gold = [json.loads(s) for s in (destination / "gold.jsonl").read_text().splitlines()]
    assert len(gold) == len(observations) == 3
    for visible, hidden in zip(observations, gold):
        Observation.from_dict(visible)
        assert "world" not in visible and "targets" not in visible
        assert str(hidden["world"]["config"]["seed"]) not in json.dumps(visible)
    with pytest.raises(ValueError):
        main(["generate", "--output", str(destination), "--count", "3"])
    capsys.readouterr()
