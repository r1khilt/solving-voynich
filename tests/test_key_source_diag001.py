import gzip
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_key_source_diag001 as runner
from voynich.compact_suffix_source import fit_compact
from voynich.recurrent_latin_source import RecurrentSource, score_records
from voynich.recurrent_unit_beam import RecurrentProvider


def neural_row(text, score, bound=None):
    return {'plaintext': text, 'joint_log_probability': score, 'discarded_completion_upper_bound': bound}


def test_boolean_support_matches_exhaustive_nonempty_unit_parses():
    units = ('x', 'yx', 'yy')
    for length in range(7):
        for chars in itertools.product('xy', repeat=length):
            observed = ''.join(chars)
            expected = any(''.join(parts) == observed for n in range(length + 1)
                           for parts in itertools.product(units, repeat=n))
            assert runner.support(units, observed) == expected


def test_unsupported_record_stays_in_edit_denominator_and_has_no_score():
    row = neural_row(None, None)
    r = runner.summarize_rows([row], ['abc'], 17, True)
    assert r['edits'] == r['gold_characters'] == 3
    assert not r['all_supported'] and r['joint_map_lower_nats'] is None
    row['log_likelihood'] = None
    r = runner.summarize_rows([row], ['abc'], 17, False)
    assert r['total_bits'] is None and r['edits'] == 3


def test_sum_joint_bounds_preserves_discarded_mass_uncertainty_and_code_cost():
    rows = [neural_row('ab', -2., -1.), neural_row('a', -3.)]
    r = runner.summarize_rows(rows, ['ab', 'b'], 7, True)
    assert r['joint_map_lower_nats'] == pytest.approx(-5 - 7 * math.log(2))
    assert r['joint_map_upper_nats'] == pytest.approx(-4 - 7 * math.log(2))
    assert r['record_edits'] == [0, 1]
    assert not r['marginal_likelihood_available']


def test_better_neural_returned_score_is_not_a_certified_key_preference():
    a = runner.summarize_rows([neural_row('ab', -10., -2.)], ['ab'], 0, True)
    b = runner.summarize_rows([neural_row('ab', -9., -3.)], ['ab'], 0, True)
    comparison = runner.compare_keys(a, b, True)
    assert comparison['gold_minus_learned_returned_joint_nats'] == 1
    assert not comparison['floating_bounds_favor_gold']
    assert not comparison['floating_bounds_favor_learned']
    c = runner.summarize_rows([neural_row('ab', -1.)], ['ab'], 0, True)
    assert runner.compare_keys(a, c, True)['floating_bounds_favor_gold']


def test_statistical_key_difference_uses_marginal_and_actual_literal_code():
    a = runner.summarize_rows([{'plaintext': 'a', 'log_likelihood': -math.log(2)}], ['a'], 2, False)
    b = runner.summarize_rows([{'plaintext': 'a', 'log_likelihood': -2 * math.log(2)}], ['a'], 0, False)
    r = runner.compare_keys(a, b, False)
    assert r['learned_minus_gold_total_bits'] == pytest.approx(1.)
    assert r['true_key_is_known_better_candidate']


def test_per_record_error_check_catches_cancelling_errors(monkeypatch):
    original = runner.metrics
    def corrupted(*args):
        result = original(*args)
        result['record_edits'] = [1, 0]
        return result
    monkeypatch.setattr(runner, 'metrics', corrupted)
    with pytest.raises(ValueError, match='per-record'):
        runner.summarize_rows([neural_row('a', -1.), neural_row('a', -1.)], ['a', 'b'], 0, True)


def test_complete_runner_uses_real_decoders_on_artificial_records(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'OUT', tmp_path / 'result')
    monkeypatch.setattr(runner, 'BULK', tmp_path / 'bulk')
    monkeypatch.setattr(runner, 'ALPHABET', 'ab')
    for path in (tmp_path / runner.MANIFEST, tmp_path / runner.ORIGINAL / 'evaluation.json'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    def identity(path):
        raw = path.read_bytes()
        return {'path': str(path.relative_to(tmp_path)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    def save(path, value, compressed=False):
        raw = json.dumps(value, allow_nan=False).encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(gzip.compress(raw, mtime=0) if compressed else raw)
        return identity(path)
    monkeypatch.setattr(runner, 'save_new', save)
    monkeypatch.setattr(runner, 'artifact', identity)
    model = fit_compact(['abbaabbaab', 'baabbaabba'], 'ab', 12, 1)
    model.save(tmp_path / 'counts.npz')
    selected = {a: {'counts': {'path': 'counts.npz'}, 'tau': 4.} for a in runner.ARMS}
    monkeypatch.setattr(runner, 'sources', lambda: selected)
    context = {'source_count': 1, 'max_states': 2, 'max_alternatives': 3,
               'max_emission_length': 2, 'glyph_alphabet': ['x', 'y']}
    cases = {}
    for name in runner.NAMES:
        positive = not name.endswith('-shuffle')
        cases[name] = {'data': {split: {'context': context, 'records': ['xyy'] * n}
                                for split, n in (('fit', 4), ('transfer', 2))},
                       'units': {'learned': ('x', 'y'), **({'gold': ('x', 'yy')} if positive else {})},
                       'truth': {'fit': ['ab'] * 4, 'transfer': ['ab'] * 2} if positive else None}
    monkeypatch.setattr(runner, 'load_cases', lambda: cases)
    torch.manual_seed(971)
    cpu = RecurrentSource('ab', embedding=3, width=4, layers=1).eval()
    monkeypatch.setattr(runner, 'load_neural', lambda _: cpu)
    monkeypatch.setattr(runner, 'RecurrentProvider', lambda model, device: RecurrentProvider(model, 'cpu'))
    def path_score(model, text):
        r = score_records(model, [text], 'cpu', length=len(text))
        return -r['bits'] * math.log(2) + len(text) * math.log1p(-1 / 225) + math.log(1 / 225)
    monkeypatch.setattr(runner, 'path_score', path_score)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    monkeypatch.setattr(torch.mps, 'driver_allocated_memory', lambda: 0)
    monkeypatch.setattr(torch.mps, 'empty_cache', lambda: None)
    threads = torch.get_num_threads()
    try:
        result = runner.run('artificial')
    finally:
        torch.set_num_threads(threads)
    assert result['reading_count'] == 576 and result['new_key_fits'] == 0
    assert len(list((tmp_path / 'bulk').glob('*/*.json.gz'))) == 64
    for arm in runner.ARMS:
        assert len(result['case_metrics'][arm]) == 16
        assert result['case_metrics'][arm]['B-key1']['gold']['transfer']['edits'] == 0
        assert result['case_metrics'][arm]['B-key1-shuffle']['learned']['transfer']['edits'] is None
