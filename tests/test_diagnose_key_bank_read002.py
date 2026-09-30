import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_key_bank_read002 import posterior_summary


def test_posterior_key_revision_matches_analytic_bayes():
    bank = {'bank': [{'log_weight': math.log(.9)}, {'log_weight': math.log(.1)}]}
    evidence = {'active_bank_indices': [0, 1], 'per_key': [
        {'bank_index': 0, 'log_likelihood': math.log(.1)},
        {'bank_index': 1, 'log_likelihood': math.log(.9)}], 'log_likelihood': math.log(.18)}
    value = posterior_summary(bank, evidence)
    assert value['maximum_posterior_key_mass'] == pytest.approx(.5)
    assert value['transfer_key_entropy_bits'] == pytest.approx(1.)
    assert value['transfer_to_fitting_key_kl_bits'] == pytest.approx(.5*math.log2(.5/.9)+.5*math.log2(.5/.1))
    # A particular observation can increase key entropy while KL remains positive.
    assert value['transfer_key_entropy_bits'] > value['fitting_key_entropy_bits']


def test_unsupported_mass_tiny_supported_weight_and_text_half_mass():
    bank = {'bank': [{'log_weight': 0.}, {'log_weight': -10000.}]}
    evidence = {'active_bank_indices': [0, 1], 'per_key': [
        {'bank_index': 0, 'log_likelihood': None},
        {'bank_index': 1, 'log_likelihood': -1.}], 'log_likelihood': -10001.}
    reading = {'mixture': {'plaintexts': ['a'], 'joint_log_probability': -10001.1, 'floating_bound_separated': True}}
    value = posterior_summary(bank, evidence, reading)
    assert value['maximum_posterior_key_index'] == 1 and value['maximum_posterior_key_mass'] == pytest.approx(1.)
    assert value['selected_text_above_half_mass']
    assert value['selected_text_posterior'] == pytest.approx(math.exp(-.1))
    assert value['transfer_to_fitting_key_kl_bits'] == pytest.approx(10000/math.log(2))


def test_all_unsupported_has_no_posterior():
    bank = {'bank': [{'log_weight': 0.}]}
    evidence = {'active_bank_indices': [0], 'per_key': [{'bank_index': 0, 'log_likelihood': None}], 'log_likelihood': None}
    value = posterior_summary(bank, evidence)
    assert value['status'] == 'entire_bank_unsupported' and 'selected_text_posterior' not in value


def test_invalid_evidence_or_text_mass_rejected():
    bank = {'bank': [{'log_weight': 0.}]}
    evidence = {'active_bank_indices': [0], 'per_key': [{'bank_index': 0, 'log_likelihood': -1.}], 'log_likelihood': -2.}
    with pytest.raises(ValueError):
        posterior_summary(bank, evidence)
    evidence['log_likelihood'] = -1.
    with pytest.raises(ValueError):
        posterior_summary(bank, evidence, {'mixture': {'plaintexts': ['a'], 'joint_log_probability': 0.}})
