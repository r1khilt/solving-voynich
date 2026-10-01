import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from voynich.fixed_transition_provider import FixedTransitionProvider
from voynich.recurrent_latin_source import RecurrentSource
from voynich.recurrent_shared_prefix import decode_shared_prefix
from voynich.recurrent_unit_beam import RecurrentProvider


@pytest.mark.parametrize("layers", [1, 2])
def test_noncontiguous_branch_parents_padding_and_complete_history_probabilities(layers):
    torch.manual_seed(71801)
    model = RecurrentSource(alphabet="abc", embedding=5, width=7, layers=layers).eval()
    trace = []
    provider = FixedTransitionProvider(model, batch=8, after_batch=trace.append)
    _, bos = provider.advance([3], [None])
    _, first = provider.advance([0, 1, 2], bos*3)
    # Repeated and reordered parents exercise independent branch histories.
    choices = [(i*7 % 3, i//3 % 3) for i in range(19)]
    original = [[v.clone() for v in state] for state in first]
    logp, states = provider.advance([c for _, c in choices], [first[p] for p, _ in choices])
    with torch.inference_mode():
        inputs = torch.tensor([[3, p, c] for p, c in choices])
        logits, expected_state = model(inputs)
        expected = F.log_softmax(logits[:, -1].double(), dim=-1).numpy()
    assert np.max(np.abs(logp-expected)) < 2e-6
    assert [r["shape"] for r in trace] == [[8, 1]]*5
    assert provider.real_rows == 23 and provider.padded_rows == 17
    for i, state in enumerate(states):
        for j, value in enumerate(state):
            assert value.device.type == "cpu"
            assert value.untyped_storage().nbytes() == value.numel()*value.element_size()
            assert torch.allclose(value, expected_state[j][:, i:i+1], atol=2e-6)
    for previous, saved in zip(first, original, strict=True):
        assert all(torch.equal(a, b) for a, b in zip(previous, saved, strict=True))
    saved = states[1][0].clone()
    with torch.inference_mode():
        states[0][0].fill_(123)
    assert torch.equal(states[1][0], saved)


def test_fixed_provider_preserves_shared_key_tuple_search_and_cache_inventory():
    torch.manual_seed(71809)
    model = RecurrentSource(alphabet="ab", embedding=5, width=7, layers=2).eval()
    kwargs = dict(keys=(("x", "xx"), ("xx", "x"), ("x", "x")), records=("xxx", "xx"),
                  rho=.2, log_weights=tuple(map(math.log, (.2, .3, .5))), beam_width=256)
    variable = decode_shared_prefix(RecurrentProvider(model), **kwargs)
    fixed = decode_shared_prefix(FixedTransitionProvider(model, batch=8), **kwargs)
    for field in ("plaintexts", "compatible_key_indices", "expanded_prefixes", "distinct_source_prefixes",
                  "source_cache_hits", "floating_map_bound_separated", "discarded_prefixes"):
        assert fixed[field] == variable[field]
    assert fixed["joint_log_probability"] == pytest.approx(variable["joint_log_probability"], abs=2e-6)


@pytest.mark.parametrize("tokens,states", [([], []), ([0], []), ([True], [None]),
                                           ([0], [None]), ([2, 2], [None, None]),
                                           ([0], [(torch.zeros(1, 1, 2),)*2])])
def test_bad_token_initialization_and_cache_shapes_fail(tokens, states):
    model = RecurrentSource(alphabet="ab", embedding=3, width=4, layers=2)
    with pytest.raises(ValueError):
        FixedTransitionProvider(model).advance(tokens, states)


@pytest.mark.parametrize("batch", [0, -1, True, 2.0])
def test_invalid_fixed_shape_rejected(batch):
    with pytest.raises(ValueError):
        FixedTransitionProvider(RecurrentSource(alphabet="ab", embedding=3, width=4), batch=batch)
