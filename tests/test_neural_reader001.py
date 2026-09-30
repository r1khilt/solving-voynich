import hashlib
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_neural_reader001 as runner
from scripts.evaluate_naibbe001 import edit_distance


def test_incomplete_source_validation_blocks_reserved_archive_access(monkeypatch):
    def stop():
        raise ValueError('Incomplete source fits')
    def forbidden(*_):
        raise AssertionError('Reserved archive accessed')
    monkeypatch.setattr(runner, 'sources', stop)
    monkeypatch.setattr(runner, 'load_archive', forbidden)
    with pytest.raises(ValueError, match='Incomplete'):
        runner.prepare('revision')


def test_complete_panel_fixed_keys_two_authors_and_matched_nulls(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'BULK', tmp_path / 'bulk')
    monkeypatch.setattr(runner, 'CORPUS', 'corpus.json')
    manifest = {'authors': {a: {'role': 'reserved_reader_author', 'artifact': {'author': a}} for a in runner.AUTHORS}}
    raw = json.dumps(manifest).encode()
    (tmp_path / 'corpus.json').write_bytes(raw)
    monkeypatch.setattr(runner, 'CORPUS_SHA', hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(runner, 'sources', lambda: {'frozen': 'models'})
    def archive(spec):
        text = runner.ALPHABET * 700
        return {'author': spec['author'], 'alphabet': runner.ALPHABET,
                'records': [{'id': spec['author'], 'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest()}]}
    monkeypatch.setattr(runner, 'load_archive', archive)
    saved = {}
    def save(path, payload):
        saved[path.name] = payload
        return {'path': str(path), 'sha256': 'test', 'bytes': 1}
    monkeypatch.setattr(runner, 'save_new', save)
    result = runner.prepare('revision')
    panel, answers = saved['panel.json']['cases'], saved['answers.json']['cases']
    assert len(panel) == len(answers) == 32
    assert result['positive_records'] == result['null_records'] == 32
    signatures = set()
    for i in range(1, 17):
        name = f'B-key{i}'
        a, b = panel[name], panel[name + '-shuffle']
        signatures.add(tuple(a['units']))
        assert a['units'] == b['units']
        assert [len(r) for r in a['records']] == [len(r) for r in b['records']]
        assert all(len(p) == 224 for p in answers[name])
    assert len(signatures) == 16
    assert all('text' not in row for rows in result['source_windows'].values() for row in rows)
    runner.validate_panel(panel)
    del panel['B-key16-shuffle']
    with pytest.raises(ValueError, match='Complete'):
        runner.validate_panel(panel)


def test_independent_full_grid_edit_distance_matches_existing_metric():
    rng = random.Random(991)
    for _ in range(100):
        a = ''.join(rng.choice('abc') for _ in range(rng.randrange(12)))
        b = ''.join(rng.choice('abc') for _ in range(rng.randrange(12)))
        assert runner.edit_reference(a, b) == edit_distance(a, b)


def test_prediction_source_has_no_answer_read():
    import ast
    source = ast.parse(Path(runner.__file__).read_text())
    predict = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'predict')
    strings = [node.value for node in ast.walk(predict) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert 'answers' not in strings and 'answers.json' not in strings
