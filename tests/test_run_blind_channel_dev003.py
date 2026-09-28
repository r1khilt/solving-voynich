"""Guard the experiment boundary: fitting must not open held-out answers."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import run_blind_channel_dev003 as runner
from voynich.finite_state_channel import SourceModel


def panel():
    source = SourceModel(("a", "b"), 0, {"": {"a": .6, "b": .4}}).to_dict()
    context = {"source_alphabet": ["a", "b"], "glyph_alphabet": ["X", "Y"],
               "denominator": 4, "max_states": 2, "max_emission_length": 2,
               "max_alternatives": 3, "source_count": 1, "stop_probability": .2}
    names = ["B-key2-shuffle", "B-key1", "B-key2", "B-key1-shuffle"]
    manifest = {"source": {"path": "source.json", "sha256": "source-hash"}, "cases": {}}
    for name in names:
        manifest["cases"][name] = {"positive": not name.endswith("shuffle"), "artifacts": {
            split: {"path": name + "-" + split + ".json", "sha256": split + "-hash"}
            for split in ("fit", "transfer", "answer")}}
    return manifest, source, {"context": context, "records": ["XXY", "XY"]}


def test_panel_order_and_no_substituted_cases():
    manifest, _, _ = panel()
    assert runner.case_names(manifest) == ["B-key1", "B-key1-shuffle", "B-key2", "B-key2-shuffle"]
    del manifest["cases"]["B-key2"]
    with pytest.raises(ValueError, match="panel changed"):
        runner.case_names(manifest)


def test_fit_reads_source_and_fit_only_after_freeze_check(tmp_path, monkeypatch):
    manifest, source, fit = panel()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.resource, "setrlimit", lambda *_: None)
    reads = []

    def require(commit, paths):
        assert commit == "declared-source-freeze"
        assert "tests/test_unit_channel_search_independent.py" in paths
        reads.append("freeze")

    def checked(artifact):
        assert reads[0] == "freeze"
        reads.append(artifact["path"])
        if artifact == manifest["source"]:
            return {"source_model": source}
        assert artifact == manifest["cases"]["B-key2"]["artifacts"]["fit"]
        return fit

    def search(received_source, records, context, *, config):
        assert received_source.to_dict() == source
        assert records == fit["records"]
        assert config.seed == 52303
        assert config.max_seconds == 300.
        assert config.restarts == 16
        assert context.stop_probability == .2
        payload = {"trace": [], "channel": None, "score": None, "seconds": 0.,
                   "stop_reason": "artificial_empty", "evaluated_neighbors": 0}
        return SimpleNamespace(**payload, to_dict=lambda: payload)

    for path in runner.paths_for(manifest):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("artificial source bytes\n")
    monkeypatch.setattr(runner, "require_frozen", require)
    monkeypatch.setattr(runner, "checked_artifact", checked)
    monkeypatch.setattr(runner, "search_unit_channel", search)
    runner.fit_case("B-key2", "declared-source-freeze", manifest)
    assert reads == ["freeze", "source.json", "B-key2-fit.json"]
    saved = json.loads(runner.freeze_path("B-key2").read_text())
    assert saved["input_sha256"] == "fit-hash"
    assert (tmp_path / saved["full_output"]["path"]).stat().st_size == saved["full_output"]["bytes"]


def test_existing_model_or_evaluation_cannot_be_overwritten(tmp_path, monkeypatch):
    manifest, _, _ = panel()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.resource, "setrlimit", lambda *_: None)
    path = runner.freeze_path("B-key1")
    path.parent.mkdir(parents=True)
    path.write_text("preserved model")
    with pytest.raises(FileExistsError):
        runner.fit_case("B-key1", "unused", manifest)
    with pytest.raises(FileExistsError):
        runner.campaign("unused", manifest)
    result = path.parent / "evaluation.json"
    result.write_text("preserved evaluation")
    with pytest.raises(FileExistsError):
        runner.evaluate("unused", manifest)
    assert path.read_text() == "preserved model"
    assert result.read_text() == "preserved evaluation"


def test_campaign_failure_does_not_replace_or_extend_case(tmp_path, monkeypatch):
    manifest, _, _ = panel()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "require_frozen", lambda *_: None)
    launched = []

    def run(command, **kwargs):
        name = command[command.index("--case") + 1]
        launched.append(name)
        assert kwargs["timeout"] == 420
        assert kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(runner.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="worker failed"):
        runner.campaign("freeze", manifest)
    assert sorted(launched) == ["B-key1", "B-key1-shuffle"]
    status = json.loads((tmp_path / f"results/{runner.EXPERIMENT}/campaign.json").read_text())
    assert status["status"] == "failed"
    assert len(status["cases"]) == 2


def test_evaluation_requires_all_models_frozen_before_answer_reads(tmp_path, monkeypatch):
    manifest, _, _ = panel()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    expected = {f"results/{runner.EXPERIMENT}/{name}_freeze.json" for name in runner.case_names(manifest)}

    def reject(commit, paths):
        assert expected <= set(paths)
        raise ValueError("deliberately unfrozen")

    monkeypatch.setattr(runner, "require_frozen", reject)
    monkeypatch.setattr(runner, "checked_artifact", lambda _: pytest.fail("An answer/input was read before freezing"))
    with pytest.raises(ValueError, match="deliberately unfrozen"):
        runner.evaluate("unfrozen", manifest)
