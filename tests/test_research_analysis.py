"""Controls that distinguish latent recovery from leakage and intervention artifacts."""

import itertools

import numpy as np
import pytest
import torch

from voynich.calibration import fit_readout, predict_readout, subspace_patch
from voynich.context_analysis import paired_summary, remote_shuffle, sample_contexts
from voynich.runtime import PageWindows, corpus_identity
from voynich.synthetic import ALPHABET, filter_sequence, generate_page, prepare


def test_shuffle_changes_only_distant_order():
    x = np.arange(256).reshape(2, 128)
    y = remote_shuffle(x)
    np.testing.assert_array_equal(y[:, -16:], x[:, -16:])
    np.testing.assert_array_equal(np.sort(y[:, :-16]), x[:, :-16])
    assert not np.array_equal(y, x)
    np.testing.assert_array_equal(remote_shuffle(x), y)
    np.testing.assert_array_equal(x, np.arange(256).reshape(2, 128))


@pytest.mark.parametrize('keep', [0, 128, 130, -1])
def test_bad_shuffle_rejected(keep):
    with pytest.raises(ValueError):
        remote_shuffle(np.zeros((2, 128)), keep)


def test_context_targets_are_excluded_and_page_contained(tmp_path):
    prepare(tmp_path, 'cycle_null', counts=(2, 2, 2))
    data = PageWindows(tmp_path, 'validation', 256)
    samples = sample_contexts(data)
    assert len(samples) == 16
    assert sample_contexts(data) == samples
    for example in samples:
        seq = data.sequences[example['page_id']]
        pos = example['position']
        assert example['prefix'] == seq[pos-128:pos]
        assert example['target'] == seq[pos]
        assert pos < len(seq)-1
    assert corpus_identity(tmp_path)


def test_cluster_summary_does_not_weight_long_leaves_more():
    result = paired_summary([0, 0, 9], [{'leaf_id': 'a'}, {'leaf_id': 'a'}, {'leaf_id': 'b'}])
    assert result['sample_mean'] == 3
    assert result['equal_leaf_mean'] == 4.5
    assert result['leaf_count'] == 2


def test_filter_against_exhaustive_hidden_paths():
    text = 'acb'
    p = .65
    mass = np.zeros(4)
    signal_mass = 0.
    # Enumerate all possible initial states and signal/null paths.
    for initial, roles in itertools.product(range(4), itertools.product([0, 1], repeat=len(text))):
        state, weight = initial, .25
        for char, role in zip(text, roles, strict=True):
            if role:
                state = (state+1) % 4
                weight *= p/2 if ALPHABET.index(char)//2 == state else 0
            else:
                weight *= (1-p)/8
        mass[state] += weight
        signal_mass += weight*roles[-1]
    state, role, prediction = filter_sequence(text)
    np.testing.assert_allclose(state[-1], mass/mass.sum(), atol=1e-12)
    assert role[-1] == pytest.approx(signal_mass/mass.sum())
    np.testing.assert_allclose(prediction.sum(-1), 1)
    # Each observed symbol has equal-probability homophones in prediction.
    np.testing.assert_allclose(prediction[:, ::2], prediction[:, 1::2])


def test_iid_filter_cannot_recover_independent_labels():
    state, role, prediction = filter_sequence('abccfehg', 'iid')
    np.testing.assert_allclose(state, .25)
    np.testing.assert_allclose(role, .65)
    np.testing.assert_allclose(prediction, .125)


def test_generator_filler_preserves_state_and_signal_cycles():
    text, states, roles = generate_page(np.random.default_rng(15), 1000)
    for i in range(1, len(text)):
        if roles[i]:
            assert states[i] == (states[i-1]+1) % 4
            assert ALPHABET.index(text[i])//2 == states[i]
        else:
            assert states[i] == states[i-1]
    assert 250 < roles.count(0) < 450


def test_synthetic_splits_independent_reproducible_and_no_overwrite(tmp_path):
    a, b = tmp_path/'a', tmp_path/'b'
    first = prepare(a, 'iid', counts=(3, 2, 2))
    second = prepare(b, 'iid', counts=(3, 2, 2))
    assert first == second
    with pytest.raises(ValueError):
        prepare(a, 'iid')
    train = PageWindows(a, 'train', 256)
    test = PageWindows(a, 'test', 256, allow_test=True)
    assert not {p['text'] for p in train.pages} & {p['text'] for p in test.pages}
    x, targets, _ = train.batch([0], 'cpu')
    assert x.shape == targets[1].shape == (1, 256)
    assert x[0].tolist() == train.tokenizer.encode(train.pages[0]['text'], add_eos=False)


def test_readout_uses_train_normalization_and_generalizes_linear_signal():
    rng = np.random.default_rng(8)
    train, test = rng.normal(size=(1000, 6)), rng.normal(size=(200, 6))
    labels = (train[:, 0] > 0).astype(int)
    probe = fit_readout(train, labels, 2)
    assert np.mean(predict_readout(probe, test) == (test[:, 0] > 0)) > .95
    np.testing.assert_array_equal(probe['mean'], train.mean(0))


def test_subspace_patch_preserves_orthogonal_components_and_positions():
    recipient = torch.randn(3, 5, 8)
    donor = torch.randn(3, 5, 8)
    basis = torch.eye(8)[:, :3]
    patched = subspace_patch(donor, basis)(recipient)
    torch.testing.assert_close(patched[:, :-1], recipient[:, :-1])
    torch.testing.assert_close(patched[:, -1, :3], donor[:, -1, :3])
    torch.testing.assert_close(patched[:, -1, 3:], recipient[:, -1, 3:])
    random_basis = torch.eye(8)[:, 3:6]
    desired = (patched[:, -1]-recipient[:, -1]).norm(dim=-1, keepdim=True)
    controlled = subspace_patch(donor, random_basis, desired)(recipient)
    torch.testing.assert_close((controlled[:, -1]-recipient[:, -1]).norm(dim=-1, keepdim=True), desired)
