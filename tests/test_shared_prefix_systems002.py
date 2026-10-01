from types import SimpleNamespace

import pytest

from scripts import benchmark_shared_prefix_systems002 as correction
from scripts import benchmark_shared_prefix_systems001 as engine
from scripts import audit_shared_prefix_systems001 as auditor
from tests.test_shared_prefix_systems001 import configure


def test_scoped_namespace_child_entrypoint_and_original_failure_binding(tmp_path, monkeypatch):
    save, artifact = configure(tmp_path, monkeypatch)
    newout, newbulk = tmp_path/"new-results", tmp_path/"new-outputs"
    monkeypatch.setattr(correction, "ROOT", tmp_path)
    monkeypatch.setattr(correction, "OUT", newout)
    monkeypatch.setattr(correction, "BULK", newbulk)
    monkeypatch.setattr(correction, "save_new", save)
    monkeypatch.setattr(correction, "artifact", artifact)
    monkeypatch.setattr(correction, "require_frozen", lambda *_: None)
    monkeypatch.setattr(correction, "previous_failure", lambda: {"original": "retained_FAILED"})
    path = tmp_path/"scripts/benchmark_shared_prefix_systems002.py"
    path.parent.mkdir(parents=True)
    path.write_text("fake corrected child entrypoint\n")
    before = (engine.OUT, engine.BULK, engine.PATHS, engine.__file__, auditor.OUT, auditor.PATHS)
    calls = []
    def run(command, **kwargs):
        assert command[2] == str(path) and kwargs["timeout"] == 615
        calls.append(command[command.index("--arm")+1])
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(engine.subprocess, "run", run)
    correction.run("campaign", "frozen")
    assert calls == list(engine.ARMS)
    assert (newout/"campaign.json").exists()
    assert (newout/"execution-binding.json").exists()
    assert not (before[0]/"campaign.json").exists()
    assert (engine.OUT, engine.BULK, engine.PATHS, engine.__file__, auditor.OUT, auditor.PATHS) == before
    with pytest.raises(FileExistsError):
        correction.run("campaign", "frozen")
    assert len(calls) == 2


def test_scoped_aliases_restore_after_arm_failure(monkeypatch):
    monkeypatch.setattr(correction, "require_frozen", lambda *_: None)
    monkeypatch.setattr(correction, "previous_failure", lambda: {})
    before = (engine.OUT, engine.__file__, auditor.OUT, auditor.PATHS)
    def fail(arm, freeze):
        assert engine.OUT == correction.OUT and auditor.OUT == correction.OUT
        assert engine.__file__.endswith("benchmark_shared_prefix_systems002.py")
        assert arm == "fixed8-host" and freeze == "frozen"
        raise RuntimeError("preserved failure")
    monkeypatch.setattr(engine, "run_arm", fail)
    with pytest.raises(RuntimeError, match="preserved"):
        correction.run("arm", "frozen", "fixed8-host")
    assert (engine.OUT, engine.__file__, auditor.OUT, auditor.PATHS) == before


def test_original_serialization_failure_is_verified_from_published_artifacts():
    bindings = correction.previous_failure()
    assert set(bindings) == {"campaign", "audit", *engine.ARMS}
    assert all(row["bytes"] > 0 and len(row["sha256"]) == 64 for row in bindings.values())
