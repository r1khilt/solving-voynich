import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_key_bank_expand001 import score_fit
from scripts.run_key_bank_fit001 import build_bank
from voynich.expanded_key_bank import expand_key_bank
from voynich.native_suffix_marginal import marginal_python
from voynich.sparse_suffix_source import SuffixSource


def inputs():
    source = SuffixSource('ab', 0, 1., {'': {'a': 3, 'b': 1}})
    observed = {'records': ['xy', 'xxy'], 'context': {
        'source_alphabet': list('ab'), 'glyph_alphabet': list('xy'),
        'max_states': 2, 'max_alternatives': 3, 'max_emission_length': 2,
        'stop_probability': .25}}
    scorer = SimpleNamespace(alphabet=source.alphabet,
                             score=lambda *args: marginal_python(source, *args))
    return source, observed, scorer


def test_expanded_one_round_reproduces_original_exact_bank_and_weights():
    source, observed, scorer = inputs()
    original = build_bank(source, observed, ('x', 'y'))
    expanded = expand_key_bank(lambda key: score_fit(scorer, observed, key), ('x', 'y'), 'xy', max_rounds=1)
    assert original['best_index'] == expanded['best_index']
    for first, second in zip(original['bank'], expanded['bank'], strict=True):
        assert tuple(first['units']) == tuple(second['units'])
        for field in ['record_log_likelihoods', 'record_nodes', 'fit_log_likelihood',
                      'model_bits', 'log_weight_unnormalized']:
            assert first[field] == second[field]
        assert first['log_weight'] == pytest.approx(second['log_weight'], abs=1e-12) if first['log_weight'] is not None else second['log_weight'] is None


def test_invalid_numerical_score_is_failure_even_if_another_record_unsupported():
    _, observed, scorer = inputs()
    values = iter([-math.inf, float('nan')])
    scorer.score = lambda *args: SimpleNamespace(log_likelihood=next(values))
    with pytest.raises(ValueError, match='Invalid fitting'):
        score_fit(scorer, observed, ('x', 'y'))


def test_alphabet_mismatch_and_native_failure_propagate():
    _, observed, scorer = inputs()
    scorer.alphabet = ('b', 'a')
    with pytest.raises(ValueError, match='alphabet'):
        score_fit(scorer, observed, ('x', 'y'))
    scorer.alphabet = ('a', 'b')
    def fail(*args):
        raise RuntimeError('native cap')
    scorer.score = fail
    with pytest.raises(RuntimeError, match='native cap'):
        score_fit(scorer, observed, ('x', 'y'))
