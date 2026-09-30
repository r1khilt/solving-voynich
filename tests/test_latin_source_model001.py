"""No reserved-author reads, fixed inputs, exclusive output and saved metadata."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_latin_source_model001 as runner
from voynich.recurrent_latin_source import RecurrentSource


@pytest.mark.parametrize('author,role', [('phi0588', 'reserved_reader_author'), ('phi1212', 'training_pool'),
                                       ('phi1318', 'training_pool'), ('phi0448', 'source_validation')])
def test_wrong_role_rejected_before_any_file_access(monkeypatch, author, role):
    manifest = {'authors': {author: {'role': 'reserved_reader_author' if author in ('phi0588', 'phi1212')
                                    else 'source_validation' if author == 'phi1318' else 'training_pool'}}}
    def fail(*_):
        raise AssertionError('Forbidden archive load')
    monkeypatch.setattr(runner, 'load_archive', fail)
    with pytest.raises(ValueError):
        runner.load_author(manifest, author, role)


def test_allowed_roles_retain_alphabet_and_author_binding(monkeypatch):
    manifest = {'authors': {'phi0448': {'role': 'training_pool', 'artifact': {}}}}
    monkeypatch.setattr(runner, 'load_archive', lambda _: {'alphabet': runner.ALPHABET, 'author': 'phi1318', 'records': []})
    with pytest.raises(ValueError, match='mismatch'):
        runner.load_author(manifest, 'phi0448', 'training_pool')


def test_no_unregistered_seed_or_dataset():
    with pytest.raises(ValueError):
        runner.train('small', 123, 'not-a-commit')
    with pytest.raises(ValueError):
        runner.load_inputs('reserved')


def test_checkpoint_identity_and_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    model = RecurrentSource('abc', 4, 7, 2)
    path = tmp_path / 'checkpoint.pt'
    identity = runner.save_checkpoint(model, path, 100, 31103, 'small', 'frozen')
    assert identity['bytes'] > 0 and len(identity['sha256']) == 64
    saved = torch.load(path, weights_only=True)
    assert saved['step'] == 100 and saved['dataset'] == 'small' and saved['freeze'] == 'frozen'
    with pytest.raises(FileExistsError):
        runner.save_checkpoint(model, path, 500, 31103, 'small', 'frozen')
