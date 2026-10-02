from scripts import benchmark_source_action_cache001 as run
from voynich.source_action_proposal import SourceActionConfig, SourceActionProposal


def test_actual_tiny_mixed_and_double_prefix_reference_transport():
    config = SourceActionConfig(rows=3, glyphs=2, width=16, heads=2, encoder_layers=1,
                                decoder_layers=2, max_glyphs=16, max_records=2)
    model = SourceActionProposal(config).double()
    for family in ('mixed', 'double', 'single'):
        env, trace = run.fixture(config, 5, family)
        metrics, full, cached = run.compare(model, env, trace)
        assert metrics['source_letters'] == (8 if family == 'mixed' else 10)
        assert metrics['legal_logit_max_delta'] < 1e-12
        assert metrics['whole_path_logq_delta'] < 1e-12
        assert abs(run.path_logq(full, trace[1])-run.path_logq(cached, trace[1])) < 1e-12
