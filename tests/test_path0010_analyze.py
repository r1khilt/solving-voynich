"""CPU-only checks of the independent PATH-0010 compact auditor."""

import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import path0010_analyze as audit  # noqa: E402


def test_independent_interaction_algebra_handles_antagonism_and_four_way_term():
    effects = {mask: (2 * bool(mask & 1) + bool(mask & 2)
                      - 4 * (mask & 3 == 3) + 3 * (mask == 15))
               for mask in audit.MASKS}
    margins = {mask: 10.0 - effect for mask, effect in effects.items()}
    recovered, terms = audit.interaction_terms(margins)
    assert recovered == effects
    assert terms[1] == 2 and terms[2] == 1 and terms[3] == -4
    assert np.isclose(sum(terms.values()), effects[15])
    assert audit.layers(0) == [] and audit.layers(15) == list(audit.WINDOW)


def test_source_hash_uses_frozen_git_blob_not_mutable_worktree(tmp_path):
    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True)
    source = tmp_path / 'method.py'
    source.write_text('frozen = True\n')
    subprocess.run(['git', 'add', 'method.py'], cwd=tmp_path, check=True)
    subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'freeze'], cwd=tmp_path, check=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tmp_path,
                                       text=True).strip()
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_text('frozen = False\n')
    blob = subprocess.check_output(['git', 'show', f'{revision}:method.py'], cwd=tmp_path)
    assert hashlib.sha256(blob).hexdigest() == expected
    assert hashlib.sha256(source.read_bytes()).hexdigest() != expected
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_output(['git', 'show', f'{revision}:missing.py'], cwd=tmp_path,
                                stderr=subprocess.DEVNULL)
