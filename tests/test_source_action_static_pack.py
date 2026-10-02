import copy

import pytest
import torch

from voynich.source_action_proposal import ReadingEnvironment, SourceActionConfig, SourceActionProposal
from voynich.source_action_static_pack import static_pack


@pytest.mark.parametrize('binding', [True, False])
def test_fixed_padding_preserves_legal_active_logits_whole_path_loss_and_every_gradient(binding):
    torch.manual_seed(47)
    config = SourceActionConfig(width=8, heads=2, encoder_layers=1, decoder_layers=1,
                               max_glyphs=32, max_records=2, binding_input=binding)
    original = SourceActionProposal(config).double()
    fixed = copy.deepcopy(original)
    environments, traces = [], []
    for texts in (((0, 1, 0), (1, 0)), ((2, 0, 1, 2, 1), (1,))):
        key = tuple((0, 6, 0)[i % 3] for i in range(23))
        env0 = ReadingEnvironment([(0,), (0,)])
        records = [tuple(g for a in text for g in env0.pool[key[a]]) for text in texts]
        env = ReadingEnvironment(records)
        environments.append(env)
        traces.append(env.teaching_trace(texts, key))
    a = original.pack(environments, traces)
    b = static_pack(fixed, environments, traces, max_steps=16)
    assert b[0].shape == (2, 2, 32) and b[1].shape == (2, 16, 23)
    active = a[-2]
    logits_a, logits_b = original(a), fixed(b)[:, :a[1].shape[1]]
    legal = a[5] & active[..., None]
    assert torch.allclose(logits_a[legal], logits_b[legal], atol=2e-12, rtol=0)
    loss_a, loss_b = original.loss(a), fixed.loss(b)
    assert torch.allclose(loss_a, loss_b, atol=2e-12, rtol=0)
    loss_a.backward()
    loss_b.backward()
    for (name, left), (other, right) in zip(original.named_parameters(), fixed.named_parameters(), strict=True):
        assert name == other and left.grad is not None and right.grad is not None
        assert torch.allclose(left.grad, right.grad, atol=2e-12, rtol=0), name
    with pytest.raises(ValueError, match='exceeds'):
        static_pack(fixed, environments, traces, max_steps=2)
    with pytest.raises(ValueError, match='bounded'):
        static_pack(fixed, environments, traces, max_steps=897)
