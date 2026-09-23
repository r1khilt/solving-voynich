"""Frozen PATH-0010 task, subset, and factorial algebra checks."""

import json

import numpy as np
import pytest

from voynich.workspace.path10_campaign import (
    MASKS, SINGLES, _check_logits, mask_layers, mobius_terms, run,
)
from voynich.workspace.path10_tasks import BUNDLES, path10_tasks
from voynich.workspace.path6_tasks import BUNDLES as PATH6_BUNDLES
from voynich.workspace.path8_tasks import CONFIRMATION_BUNDLES as PATH8_BUNDLES


def test_fresh_panel_inverse_pairs_and_disjoint_words():
    rows = path10_tasks()
    assert len(rows) == 48
    assert len({r['prompt'] for r in rows}) == 48
    assert sum(r['family'] == 'binding' for r in rows) == 32
    assert sum(r['family'] == 'copy' for r in rows) == 16
    previous = {word.casefold() for group in PATH6_BUNDLES.values()
                for bundle in group for word in bundle}
    previous.update(word.casefold() for bundle in PATH8_BUNDLES for word in bundle)
    assert previous.isdisjoint(word.casefold() for bundle in BUNDLES for word in bundle)
    lookup = {r['prompt']: r for r in rows}
    for row in rows:
        opposite = lookup[row['donor_prompt']]
        assert opposite['donor_prompt'] == row['prompt']
        assert opposite['answer'] == row['donor_answer']
        if row['family'] == 'binding':
            assert row['answer'] != row['donor_answer']


def test_all_sixteen_masks_and_nonmonotone_mobius_reconstruction():
    assert len(MASKS) == 16
    assert mask_layers(0) == ()
    assert mask_layers(15) == (28, 29, 30, 31)
    assert [mask_layers(mask) for mask in SINGLES] == [(28,), (29,), (30,), (31,)]
    assert len({mask_layers(mask) for mask in MASKS}) == 16
    with pytest.raises(ValueError):
        mask_layers(16)
    # Two positive single effects, an antagonistic pair, and a four-way term.
    effects = {mask: 2.0*bool(mask & 1)+1.0*bool(mask & 2)
               -4.0*bool(mask & 3 == 3)+3.0*bool(mask == 15) for mask in MASKS}
    margins = {mask: 10.0-effects[mask] for mask in MASKS}
    recovered, terms = mobius_terms(margins)
    np.testing.assert_allclose([recovered[k] for k in MASKS], [effects[k] for k in MASKS])
    assert terms[1] == 2.0 and terms[2] == 1.0
    assert terms[3] == -4.0
    np.testing.assert_allclose(sum(terms.values()), effects[15], atol=1e-8)


def test_launch_rejects_dirty_registered_source_before_loading_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('voynich.workspace.path10_campaign.subprocess.check_output',
                        lambda *args, **kwargs: '?? docs/experiments/PATH-0010.md')
    with pytest.raises(RuntimeError, match='must be committed'):
        run()
    assert not (tmp_path/'results/PATH-0010/inputs.json').exists()


def test_failed_numerical_control_records_uninformative_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    controls = {}
    with pytest.raises(ValueError, match='numerical control'):
        _check_logits(np.array([1.0, 0.0]), np.array([0.0, 1.0]), 'test_control', controls)
    failure = json.loads((tmp_path/'results/PATH-0010/failure.json').read_text())
    assert failure['status'] == 'uninformative_numerical'
    assert failure['reason'] == 'test_control'
    assert failure['max_abs_logit_error'] == 1.0
