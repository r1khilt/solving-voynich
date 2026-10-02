import torch

from scripts import benchmark_source_action001 as bench
from voynich.source_action_proposal import SourceActionConfig, SourceActionProposal


def test_random_shape_compiles_all_targets_and_real_optimizer_transport():
    config = SourceActionConfig(rows=3, glyphs=2, width=16, heads=2, encoder_layers=1,
                                decoder_layers=1, max_records=2, max_glyphs=8)
    model = SourceActionProposal(config).double()
    env, traces = bench.fixture(config, batch=2, letters=4)
    packed = model.pack(env, traces)
    assert packed[0].shape == (2, 2, 8) and packed[1].shape == (2, 8, 3)
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    steps = bench.train_steps(model, packed, 2)
    assert len(steps) == 2 and all(v['whole_path_nll'] > 0 for v in steps)
    assert any(not torch.equal(initial[name], p) for name, p in model.named_parameters())


def test_declared_full_shapes_fit_no_truncation_and_same_parameter_ablation():
    for config in bench.CONFIGS.values():
        env, traces = bench.fixture(config, batch=4, letters=224)
        assert all(tuple(map(len, e.records)) == (448, 448) for e in env)
        assert all(len(t[1]) == 448 for t in traces)
    config = SourceActionConfig(rows=2, glyphs=2, width=16, heads=2, encoder_layers=1,
                                decoder_layers=1, max_records=2, max_glyphs=8, binding_input=False)
    model = SourceActionProposal(config).double()
    env, traces = bench.fixture(config, batch=1, letters=4)
    model.loss(model.pack(env, traces)).backward()
    assert torch.equal(model.binding.weight.grad, torch.zeros_like(model.binding.weight.grad))
