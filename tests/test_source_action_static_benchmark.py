import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import benchmark_source_action_static001 as bench
from voynich.source_action_proposal import SourceActionConfig, SourceActionProposal
from voynich.source_action_static_pack import static_pack


def test_seeded_variable_truth_becomes_fixed_geometry_without_target_length_features():
    config = SourceActionConfig(width=8, heads=2, encoder_layers=1, decoder_layers=1,
                               max_glyphs=32, max_records=2)
    model = SourceActionProposal(config)
    rng = np.random.default_rng(53)
    lengths = []
    for step in (1, 2, 16):
        environments, traces = bench.fixture(rng, step, config, batch=2, low=2, high=8)
        packed = static_pack(model, environments, traces, max_steps=16)
        assert packed[0].shape == (2, 2, 32) and packed[1].shape == (2, 16, 23)
        for env, trace in zip(environments, traces, strict=True):
            assert trace[0][0] == env.initial
            assert env.selected_record(trace[2]) is None
            lengths.append(len(trace[1]))
        if step == 16:
            assert all(len(trace[1]) == 16 for trace in traces)
    assert len(set(lengths)) > 1
    with pytest.raises(ValueError, match='Bounded'):
        bench.fixture(rng, 1, config, low=1, high=17)


def test_actual_stress_module_entrypoint_imports_before_scientific_launch():
    result = subprocess.run([sys.executable, '-m', 'scripts.benchmark_source_action_static001', '--help'],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'fixed-retain' in result.stdout and 'fixed-clear' in result.stdout
