import copy
import math

import pytest

from scripts.benchmark_shared_key002 import serializable
from scripts.run_key_bank_read001 import read_case
from voynich.candidate_rerank import prepare_candidates, rank_candidates
from voynich.sparse_suffix_source import SuffixSource


def fixture():
    source = SuffixSource('ab', 0, 1., {'': {'a': 7, 'b': 3}})
    bank = {'bank': [{'units': units, 'log_weight': -math.log(3)}
                    for units in [('X', 'XX'), ('XX', 'X'), ('X', 'X')]], 'best_index': 0}
    records = ['XXXX', 'XX']
    reading = serializable(read_case(source, records, ['X', 'XX'], bank, .2))
    return bank, records, reading


def test_union_preserves_all_texts_and_reproduces_original_mixture():
    bank, records, reading = fixture()
    value = prepare_candidates('ab', records, bank, reading)
    assert len(value['candidates']) == reading['mixture']['candidate_tuples']
    for row in value['candidates']:
        texts = [value['texts'][i] for i in row['text_indices']]
        count = 0
        for item in bank['bank']:
            mapping = dict(zip('ab', item['units'], strict=True))
            count += all(''.join(mapping[c] for c in t) == r for t, r in zip(texts, records, strict=True))
        assert row['key_log_mass'] == pytest.approx(math.log(count/3), abs=1e-12)


@pytest.mark.parametrize('fault', ['indices', 'count', 'source', 'weight', 'duplicate', 'winner'])
def test_corrupt_input_inventory_rejected(fault):
    bank, records, reading = fixture()
    if fault == 'indices':
        reading['active_bank_indices'].pop()
    elif fault == 'count':
        reading['mixture']['candidate_tuples'] += 1
    elif fault == 'source':
        reading['mixture']['per_key'][0]['readings'][0][1] = math.nan
    elif fault == 'weight':
        bank['bank'][0]['log_weight'] += 1
    elif fault == 'duplicate':
        bank['bank'][0]['units'] = bank['bank'][1]['units']
    else:
        reading['mixture']['joint_log_probability'] += .1
    with pytest.raises(ValueError):
        prepare_candidates('ab', records, bank, reading)


def test_individually_possible_records_do_not_allow_switching_keys():
    bank = {'bank': [{'units': ['X', 'Y'], 'log_weight': -math.log(2)},
                     {'units': ['Y', 'X'], 'log_weight': -math.log(2)}]}
    reading = {'active_bank_indices': [0, 1], 'mixture': {'per_key': [
        {'readings': [[['a', 'a'], -1.]]}, {'readings': []}], 'candidate_tuples': 1}}
    with pytest.raises(ValueError, match='globally consistent'):
        prepare_candidates('ab', ['X', 'Y'], bank, reading)


def test_rank_uses_whole_tuple_and_key_mass_no_bound_reuse():
    prepared = {'texts': ['a', 'b'], 'candidates': [
        {'text_indices': [0, 0], 'key_log_mass': -1000.},
        {'text_indices': [1, 1], 'key_log_mass': 0.}]}
    result = rank_candidates(prepared, [-1., -2.])
    assert result['plaintexts'] == ['b', 'b'] and result['score'] == -4.
    assert result['margin_nats'] == 998.
    assert not result['global_map_claimed'] and not result['neural_evidence_computed']
    tie = copy.deepcopy(prepared)
    tie['candidates'][0]['key_log_mass'] = 0.
    assert rank_candidates(tie, [-1., -1.])['candidate_index'] == 0
    with pytest.raises(ValueError):
        rank_candidates(prepared, [-1.])
