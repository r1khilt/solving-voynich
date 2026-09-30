import copy
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_key_bank_read001 import audit_reading
from scripts.benchmark_shared_key002 import serializable
from scripts.run_key_bank_read001 import read_case
from voynich.sparse_suffix_source import SuffixSource


def fixture():
    source = SuffixSource('ab', 0, 1., {'': {'a': 7, 'b': 3}})
    bank = {'bank': [{'units': k, 'log_weight': w} for k, w in
                    [(('x', 'xx'), -math.log(3)), (('xx', 'x'), -math.log(3)), (('x', 'x'), -math.log(3)), (('z', 'z'), None)]],
            'best_index': 0}
    records = ['xxxx', 'xx']
    reading = serializable(read_case(source, records, bank['bank'][0]['units'], bank, .2))
    return source, records, bank, reading


def test_auditor_checks_complete_inventory_and_direct_global_support():
    source, records, bank, reading = fixture()
    report = audit_reading(source.alphabet, records, bank, reading)
    assert report['active_keys'] == 3
    assert report['sampled_candidate_tuples'] == report['unique_candidates']
    assert report['literal_candidate_key_checks'] == 3 * report['unique_candidates']


@pytest.mark.parametrize('fault', ['inventory', 'support', 'bound', 'joint', 'counts', 'order'])
def test_auditor_rejects_corrupted_inference(fault):
    source, records, bank, original = fixture()
    reading = copy.deepcopy(original)
    if fault == 'inventory':
        reading['active_bank_indices'].pop()
    elif fault == 'support':
        reading['mixture']['compatible_key_indices'] = []
    elif fault == 'bound':
        reading['mixture']['unseen_log_probability_upper_bound'] = 0.
    elif fault == 'joint':
        reading['joint']['log_probability'] += 1.
    elif fault == 'counts':
        reading['mixture']['candidate_tuples'] += 1
    else:
        reading['mixture']['per_key'][0]['readings'][0][1] -= 100.
    with pytest.raises(ValueError):
        audit_reading(source.alphabet, records, bank, reading)
