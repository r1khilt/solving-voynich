import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_key_bank_expand001 import inventory
from scripts.run_key_bank_expand001 import score_fit
from voynich.expanded_key_bank import expand_key_bank
from voynich.native_suffix_marginal import marginal_python
from voynich.sparse_suffix_source import SuffixSource


def fixture():
    source = SuffixSource('ab', 0, 1., {'': {'a': 3, 'b': 1}})
    context = {'source_alphabet': ['a', 'b'], 'glyph_alphabet': ['x', 'y'],
               'max_states': 2, 'max_alternatives': 3, 'max_emission_length': 2, 'stop_probability': .25}
    observed = {'records': ['xy', 'xxy'], 'context': context}
    scorer = SimpleNamespace(alphabet=source.alphabet, score=lambda *a: marginal_python(source, *a))
    result = expand_key_bank(lambda key: score_fit(scorer, observed, key), ('x', 'y'), 'xy')
    return result, context


def test_independent_inventory_accepts_complete_bank_and_reconstructs_progress():
    bank, context = fixture()
    events = inventory(bank, ('x', 'y'), context)
    assert sum(e['kind'] == 'candidate' for e in events) == bank['bank_size']
    assert sum(e['kind'] == 'round' for e in events) == bank['complete_rounds']
    assert all('log_weight' not in e['value'] for e in events if e['kind'] == 'candidate')


@pytest.mark.parametrize('field', ['weight', 'selection', 'round_member', 'code', 'terminal', 'record_sum'])
def test_independent_inventory_rejects_corrupt_scoring_or_search_history(field):
    original, context = fixture()
    bank = copy.deepcopy(original)
    if field == 'weight':
        bank['bank'][0]['log_weight'] += .1
    elif field == 'selection':
        bank['rounds'][0]['selected_index'] = -1
    elif field == 'round_member':
        bank['rounds'][0]['member_indices'].pop()
    elif field == 'code':
        bank['bank'][0]['model_bits'] += 1
    elif field == 'terminal':
        bank['search_center_index'] = -1
    else:
        bank['bank'][0]['fit_log_likelihood'] += 1.
    with pytest.raises(ValueError):
        inventory(bank, ('x', 'y'), context)
