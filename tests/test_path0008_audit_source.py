"""The PATH-0008 audit checks frozen Git blobs and current task generation."""

import hashlib
import subprocess

import pytest

from scripts.path0008_analyze import (LAUNCH_SNAPSHOTS, committed_digest, frozen_tasks,
                                      verified_launch_source)
from voynich.workspace.campaign import canonical_digest
from voynich.workspace.path8_tasks import path8_tasks


def test_committed_source_digest_ignores_later_worktree_edit(tmp_path, monkeypatch):
    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True)
    source = tmp_path/'source.py'
    original = b'original source\n'
    source.write_bytes(original)
    subprocess.run(['git', 'add', 'source.py'], cwd=tmp_path, check=True)
    subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'frozen'], cwd=tmp_path, check=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tmp_path,
                                       text=True).strip()
    source.write_bytes(b'later clarification\n')
    monkeypatch.chdir(tmp_path)
    assert committed_digest(revision, 'source.py') == hashlib.sha256(original).hexdigest()
    assert committed_digest(revision, 'source.py') != hashlib.sha256(source.read_bytes()).hexdigest()


def test_current_task_generator_must_match_frozen_hash():
    current = {split: path8_tasks(split) for split in ('discovery', 'confirmation')}
    expected = canonical_digest(current)
    assert frozen_tasks(expected) == current
    with pytest.raises(AssertionError, match='frozen task manifest'):
        frozen_tasks('0'*64)


@pytest.mark.parametrize('path', sorted(LAUNCH_SNAPSHOTS))
def test_only_exact_hash_matched_prelaunch_amendments_are_allowed(path, tmp_path, monkeypatch):
    revision = subprocess.check_output(['git', 'rev-parse', '627be79^{commit}'],
                                       text=True).strip()
    snapshot = LAUNCH_SNAPSHOTS[path]
    expected = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert verified_launch_source(revision, path, expected) == 'hash_matched_launch_amendment'
    changed = tmp_path/snapshot.name
    changed.write_bytes(snapshot.read_bytes()+b'\nextra behavioral change\n')
    monkeypatch.setitem(LAUNCH_SNAPSHOTS, path, changed)
    with pytest.raises(AssertionError, match='unregistered changes'):
        verified_launch_source(revision, path, hashlib.sha256(changed.read_bytes()).hexdigest())


def test_unamended_source_uses_base_commit():
    revision = subprocess.check_output(['git', 'rev-parse', '627be79^{commit}'],
                                       text=True).strip()
    path = 'src/voynich/workspace/path8_tasks.py'
    assert verified_launch_source(revision, path, committed_digest(revision, path)) == 'base_commit'
