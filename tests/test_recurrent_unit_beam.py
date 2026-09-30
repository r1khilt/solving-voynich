"""Literal enumeration, deliberate search failures, and probability-bound checks."""
import itertools
import math

import numpy as np
import pytest
import torch

from voynich.recurrent_latin_source import RecurrentSource
from voynich.recurrent_unit_beam import RecurrentProvider, decode_beam


class ToySource:
    alphabet = 'ab'

    def probabilities(self, history):
        # Genuinely full-history source, not just observed offset or last letter.
        value = sum((i + 1) * (1 if char == 'a' else 3) for i, char in enumerate(history))
        p = (value % 7 + 1) / 9
        return p, 1 - p

    def advance(self, tokens, states):
        histories = ['' if state is None else state + self.alphabet[token] for token, state in zip(tokens, states)]
        return np.log([self.probabilities(h) for h in histories]), histories


def enumerate_paths(source, units, observed, rho):
    paths = {}
    for size in range(len(observed) + 1):
        for letters in itertools.product('ab', repeat=size):
            if ''.join(units['ab'.index(c)] for c in letters) != observed:
                continue
            text = ''.join(letters)
            score = math.log(rho)
            for i, c in enumerate(text):
                score += math.log1p(-rho) + math.log(source.probabilities(text[:i])['ab'.index(c)])
            paths[text] = score
    return paths


@pytest.mark.parametrize('units', [('x', 'x'), ('x', 'xx'), ('xy', 'y'), ('xx', 'yy')])
@pytest.mark.parametrize('beam', [1, 2, 64])
def test_every_short_record_against_full_history_exhaustive_enumeration(units, beam):
    source, rho = ToySource(), .2
    for size in range(6):
        for glyphs in itertools.product('xy', repeat=size):
            observed = ''.join(glyphs)
            paths = enumerate_paths(source, units, observed, rho)
            result = decode_beam(source, units, observed, rho, beam_width=beam)
            if not paths:
                assert result.support_empty and result.plaintext is None
                continue
            assert not result.support_empty
            assert result.joint_log_probability == pytest.approx(paths[result.plaintext], abs=1e-13)
            assert result.joint_log_probability <= max(paths.values()) + 1e-13
            if beam == 64 or result.score_bound_certifies_map:
                assert result.joint_log_probability == pytest.approx(max(paths.values()), abs=1e-13)
            if result.discarded_completion_upper_bound is not None:
                assert max(paths.values()) <= max(result.joint_log_probability,
                                                  result.discarded_completion_upper_bound) + 1e-13


def test_false_certificate_not_issued_when_greedy_discards_optimum():
    source = ToySource()
    failures = []
    for size in range(2, 10):
        observed = 'x' * size
        result = decode_beam(source, ('x', 'x'), observed, .2, beam_width=1)
        gold = max(enumerate_paths(source, ('x', 'x'), observed, .2).values())
        if result.joint_log_probability < gold - 1e-10:
            failures.append(result)
            assert not result.score_bound_certifies_map
    assert failures


def test_opaque_recurrent_states_match_independent_full_forward_scoring():
    torch.manual_seed(783)
    model = RecurrentSource('ab', embedding=3, width=5, layers=2).double().eval()
    provider = RecurrentProvider(model)
    result = decode_beam(provider, ('x', 'xx'), 'xxxxx', .2, beam_width=128)
    values = {}
    with torch.inference_mode():
        for n in range(3, 6):
            for letters in itertools.product('ab', repeat=n):
                if ''.join('x' if c == 'a' else 'xx' for c in letters) != 'xxxxx':
                    continue
                ids = ['ab'.index(c) for c in letters]
                logits, _ = model(torch.tensor([[2] + ids[:-1]]))
                logp = torch.log_softmax(logits[0], dim=-1)
                score = math.log(.2) + n * math.log(.8) + sum(float(logp[i, c]) for i, c in enumerate(ids))
                values[''.join(letters)] = score
    assert result.score_bound_certifies_map and result.discarded_prefixes == 0
    assert result.joint_log_probability == pytest.approx(max(values.values()), abs=1e-12)
    assert result.joint_log_probability == pytest.approx(values[result.plaintext], abs=1e-12)


def test_malformed_probabilities_cap_and_empty_record():
    source = ToySource()
    empty = decode_beam(source, ('x', 'xx'), '', .2)
    assert empty.plaintext == '' and empty.joint_log_probability == math.log(.2)
    assert empty.expanded_prefixes == 0
    with pytest.raises(RuntimeError, match='expansion cap'):
        decode_beam(source, ('x', 'x'), 'xxxx', .2, max_expanded=2)
    for kwargs in ({'beam_width': 0}, {'beam_width': True}, {'max_expanded': 0}):
        with pytest.raises(ValueError):
            decode_beam(source, ('x', 'x'), 'x', .2, **kwargs)
    class Bad(ToySource):
        def advance(self, tokens, states):
            return np.zeros((len(tokens), 2)), [''] * len(tokens)
    with pytest.raises(ValueError, match='normalized'):
        decode_beam(Bad(), ('x', 'x'), 'x', .2)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable in current sandbox')
def test_mps_state_carry_and_batched_beam_match_cpu_full_forward():
    torch.manual_seed(9917)
    cpu = RecurrentSource('ab', embedding=4, width=7, layers=2).eval()
    gpu = RecurrentSource('ab', embedding=4, width=7, layers=2).eval()
    gpu.load_state_dict(cpu.state_dict())
    gpu.to('mps')
    result = decode_beam(RecurrentProvider(gpu, 'mps'), ('x', 'xx'), 'xxxxxx', .2, beam_width=128)
    values = {}
    with torch.inference_mode():
        for n in range(3, 7):
            for letters in itertools.product('ab', repeat=n):
                text = ''.join(letters)
                if ''.join('x' if c == 'a' else 'xx' for c in text) != 'xxxxxx':
                    continue
                ids = ['ab'.index(c) for c in text]
                logits, _ = cpu(torch.tensor([[2] + ids[:-1]]))
                logp = torch.log_softmax(logits[0].double(), dim=-1)
                values[text] = math.log(.2) + n * math.log(.8) + sum(float(logp[i, c]) for i, c in enumerate(ids))
    assert result.joint_log_probability == pytest.approx(max(values.values()), abs=2e-6)
    assert result.joint_log_probability == pytest.approx(values[result.plaintext], abs=2e-6)
    assert result.discarded_prefixes == 0
