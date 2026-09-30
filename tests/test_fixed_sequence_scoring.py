import math

import pytest
import torch

from voynich.fixed_sequence_scoring import fixed_record_logps
from voynich.recurrent_latin_source import RecurrentSource, score_records


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_fixed_padding_preserves_full_history_and_partial_batch(device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('Device-access test executed separately')
    torch.manual_seed(7181)
    model = RecurrentSource('ab', embedding=4, width=8, layers=2).to(device).eval()
    texts = ['a', 'ba', 'abba', 'b'*39, 'ab'*258]
    shapes, calls = [], []
    hook = model.register_forward_pre_hook(lambda _, args: shapes.append(list(args[0].shape)))
    values = fixed_record_logps(model, texts, device, batch=3, length=520, after_batch=calls.append)
    hook.remove()
    assert shapes == [[3, 520]]*2
    assert [v['records'] for v in calls] == [3, 5]
    old = score_records(model, texts, device, batch=2, length=520)['record_bits']
    assert values == pytest.approx([-v*math.log(2) for v in old], abs=1e-4, rel=0.)
    model = model.to('cpu').double()
    independent = []
    with torch.inference_mode():
        for text in texts:
            state, previous, pieces = None, model.bos, []
            for char in text:
                logits, state = model(torch.tensor([[previous]]), state)
                target = model.config['alphabet'].index(char)
                pieces.append(float(torch.log_softmax(logits[0, 0], -1)[target]))
                previous = target
            independent.append(math.fsum(pieces))
    assert values == pytest.approx(independent, abs=1e-3, rel=0.)
    double = fixed_record_logps(model, texts, batch=2, length=520)
    assert double == pytest.approx(independent, abs=1e-10, rel=0.)


@pytest.mark.parametrize('texts,batch,length', [([], 8, 8), ([''], 8, 8),
    (['a'*9], 8, 8), (['a', 'z'], 8, 8), (['a'], 0, 8), (['a'], 8, True)])
def test_invalid_inputs_fail_before_forward(texts, batch, length):
    model = RecurrentSource('ab', embedding=4, width=8, layers=1)
    def forbidden(*_):
        raise AssertionError('Bad input reached model')
    model.register_forward_pre_hook(forbidden)
    with pytest.raises(ValueError):
        fixed_record_logps(model, texts, batch=batch, length=length)


def test_nonfinite_and_callback_failures_propagate():
    model = RecurrentSource('ab', embedding=4, width=8, layers=1)
    with torch.no_grad():
        model.output.bias[0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        fixed_record_logps(model, ['ab'], batch=1, length=2)
    with torch.no_grad():
        model.output.bias.zero_()
    def stopped(_):
        raise MemoryError('guard')
    with pytest.raises(MemoryError, match='guard'):
        fixed_record_logps(model, ['ab'], batch=1, length=2, after_batch=stopped)
