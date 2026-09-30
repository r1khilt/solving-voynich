import itertools
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_key_bank_fit001 import build_bank
from voynich.sparse_suffix_source import SuffixSource


def inputs():
    source = SuffixSource('ab', 0, 1., {'': {'a': 3, 'b': 1}})
    observed = {'records': ['xy', 'xxy'], 'context': {
        'source_alphabet': list('ab'), 'glyph_alphabet': list('xy'),
        'max_states': 2, 'max_alternatives': 3, 'max_emission_length': 2,
        'stop_probability': .25}}
    return source, observed


def exhaustive_probability(source, key, observed, rho):
    total = 0.
    for n in range(len(observed) + 1):
        for letters in itertools.product(range(2), repeat=n):
            if ''.join(key[i] for i in letters) == observed:
                total += rho * (1 - rho)**n * math.prod(source.probabilities[0, i] for i in letters)
    return total


def test_bank_fit_matches_enumeration_and_normalizes_only_after_all_candidates():
    source, observed = inputs()
    progress = []
    result = build_bank(source, observed, ('x', 'y'), progress=lambda row, n: progress.append((dict(row), n)))
    assert len(progress) == result['bank_size']
    assert all('log_weight' not in r for r, _ in progress)
    masses = []
    for row in result['bank']:
        key = row['units']
        probability = math.prod(exhaustive_probability(source, key, s, .25) for s in observed['records'])
        # Explicit family code: 1 state bit, 2*(2 alternative bits+1 length bit), one glyph bit each.
        cost = 7 + sum(map(len, key))
        assert cost == row['model_bits']
        masses.append(probability * 2**(-cost))
        if probability:
            assert row['fit_log_likelihood'] == pytest.approx(math.log(probability), abs=1e-12)
        else:
            assert row['log_weight'] is None and row['fit_log_likelihood'] is None
    for row, mass in zip(result['bank'], masses):
        assert (0 if row['log_weight'] is None else math.exp(row['log_weight'])) == pytest.approx(mass / sum(masses))
    assert result['best_index'] == max(range(len(masses)), key=masses.__getitem__)
    assert len(json.dumps(result, allow_nan=False)) > 0


def test_progress_failure_cannot_return_an_incomplete_bank_as_complete():
    source, observed = inputs()
    saved = []
    def stop(row, count):
        saved.append(row)
        raise TimeoutError('fixed cap')
    with pytest.raises(TimeoutError):
        build_bank(source, observed, ('x', 'y'), progress=stop)
    assert len(saved) == 1 and 'log_weight' not in saved[0]


def test_exact_node_cap_is_failure_not_zero_mass_or_pruned_key():
    source, observed = inputs()
    with pytest.raises(RuntimeError, match='cap'):
        build_bank(source, observed, ('x', 'y'), max_nodes=1)


def test_alphabet_mismatch_stops_before_reading():
    source, observed = inputs()
    observed['context']['source_alphabet'].reverse()
    with pytest.raises(ValueError, match='alphabet'):
        build_bank(source, observed, ('x', 'y'))
