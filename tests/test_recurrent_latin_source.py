"""Causality, manual gates, masking, window support and checkpoint reproducibility."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_latin_source_model001 import manual_logits
from voynich.recurrent_latin_source import (
    RecurrentSource, WindowSampler, chunks, encode, learning_rate, padded_batch, score_records,
)


def tiny():
    torch.manual_seed(31)
    return RecurrentSource('abc', embedding=4, width=7, layers=2).eval()


def test_future_tokens_never_affect_earlier_logits():
    model = tiny()
    with torch.no_grad():
        before, _ = model(torch.tensor([[3, 0, 1, 2, 0]]))
        after, _ = model(torch.tensor([[3, 0, 1, 0, 2]]))
    torch.testing.assert_close(before[:, :3], after[:, :3], rtol=0, atol=0)
    assert not torch.equal(before[:, 3:], after[:, 3:])


def test_incremental_full_and_independent_gate_equations_match():
    model = tiny()
    tokens = [3, 0, 1, 2, 2, 0]
    with torch.no_grad():
        full, final = model(torch.tensor([tokens]))
        state, rows = None, []
        for token in tokens:
            logits, state = model(torch.tensor([[token]]), state)
            rows.append(logits)
    torch.testing.assert_close(full, torch.cat(rows, dim=1), atol=1e-7, rtol=0)
    for a, b in zip(final, state):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=0)
    torch.testing.assert_close(full[0].double(), manual_logits(model, tokens), atol=1e-7, rtol=0)


def test_only_letters_in_normalized_output_and_bos_input():
    model = tiny()
    probabilities, _ = model.next_log_probabilities(torch.tensor([[3, 0, 1]]))
    assert probabilities.shape == (1, 3, 3)
    assert model.embedding.num_embeddings == 4 and model.bos == 3
    assert torch.all(probabilities.exp() > 0)
    torch.testing.assert_close(probabilities.exp().sum(-1), torch.ones((1, 3)))


@pytest.mark.parametrize('batch', [1, 2, 4])
def test_scoring_equals_explicit_incremental_product_and_masks_padding(batch):
    model = tiny()
    records = ['abcab', 'c', 'baac']
    result = score_records(model, records, batch=batch, length=3)
    expected = [0.] * 3
    with torch.no_grad():
        for index, part in chunks(records, 3):
            state, token = None, 3
            for char in part:
                logits, state = model(torch.tensor([[token]]), state)
                # Python log-sum-exp independent from vector log_softmax.
                values = logits[0, 0].double().tolist()
                letter = 'abc'.index(char)
                expected[index] += (math.log(math.fsum(math.exp(v) for v in values)) - values[letter]) / math.log(2)
                token = letter
    assert result['characters'] == 10
    assert result['record_bits'] == pytest.approx(expected, abs=2e-6)
    assert result['bits'] == pytest.approx(sum(expected), abs=2e-6)


def test_padded_shift_and_context_resets_are_exact():
    x, y, mask = padded_batch(['abc', 'b'], 'abc')
    assert x.tolist() == [[3, 0, 1], [3, 3, 3]]
    assert y.tolist() == [[0, 1, 2], [1, 0, 0]]
    assert mask.tolist() == [[True, True, True], [True, False, False]]
    assert chunks(['abca', 'bc'], 3) == [(0, 'abc'), (0, 'a'), (1, 'bc')]


def test_every_legal_window_once_no_boundary_crossing():
    sampler = WindowSampler(['a', 'abcab', 'ccc'], 3, 'abc')
    x, y = sampler.at(np.arange(sampler.ends[-1]))
    assert [''.join('abc'[i] for i in row) for row in y] == ['abc', 'bca', 'cab', 'ccc']
    assert np.all(x[:, 0] == 3)
    assert np.array_equal(x[:, 1:], y[:, :-1])
    a = sampler.sample(np.random.default_rng(123), 100)
    b = sampler.sample(np.random.default_rng(123), 100)
    assert all(np.array_equal(i, j) for i, j in zip(a, b))
    for indices in ([-1], [4], []):
        with pytest.raises(ValueError):
            sampler.at(indices)


def test_checkpoint_round_trip_and_seed_identity(tmp_path):
    model = tiny()
    twin = tiny()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, twin.state_dict()[key], rtol=0, atol=0)
    path = tmp_path / 'model.pt'
    torch.save({'config': model.config, 'state_dict': model.state_dict()}, path)
    payload = torch.load(path, weights_only=True)
    replay = RecurrentSource(**payload['config'])
    replay.load_state_dict(payload['state_dict'])
    x = torch.tensor([[3, 0, 1]])
    torch.testing.assert_close(model(x)[0], replay(x)[0], rtol=0, atol=0)


def test_learning_rate_endpoints_and_invalid_data():
    assert learning_rate(1) == .00001
    assert learning_rate(100) == .001
    assert learning_rate(6000) == .0001
    assert all(learning_rate(i) >= learning_rate(i + 1) for i in range(100, 6000))
    for step in (0, 6001, True):
        with pytest.raises(ValueError):
            learning_rate(step)
    for value in ('', 'd'):
        with pytest.raises(ValueError):
            encode(value, 'abc')
    with pytest.raises(ValueError):
        WindowSampler(['a'], 3, 'abc')


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable in current sandbox')
def test_mps_forward_backward_clipped_step_and_reload_match_cpu():
    model = tiny()
    x = torch.tensor([[3, 0, 1, 2], [3, 2, 1, 0]])
    y = torch.tensor([[0, 1, 2, 0], [2, 1, 0, 2]])
    with torch.no_grad():
        expected = model(x)[0]
    model.to('mps')
    with torch.no_grad():
        actual = model(x.to('mps'))[0].cpu()
    torch.testing.assert_close(expected, actual, atol=1e-6, rtol=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    logits, _ = model(x.to('mps'))
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 3), y.to('mps').reshape(-1))
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    assert torch.isfinite(gradient)
    optimizer.step()
    assert all(torch.all(torch.isfinite(v)) for v in model.state_dict().values())
    with torch.no_grad():
        device_logits = model(x.to('mps'))[0].cpu()
        model.cpu()
        host_logits = model(x)[0]
    torch.testing.assert_close(host_logits, device_logits, atol=1e-6, rtol=0)
