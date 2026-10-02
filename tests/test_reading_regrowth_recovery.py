"""Linear audit agreement and the entire longer-campaign controller on fixtures."""
import copy
import itertools

import pytest

from scripts.audit_reading_regrowth_recovery001 import linear_literal
from scripts.run_reading_regrowth_recovery001 import expected_cells, gate_summary, summaries
from tests.test_reading_regrowth import enumerate_paths, fixture_source
from voynich.reading_regrowth import RegrowthConfig, SourceRegrowth
from voynich.regrowth_trace import path_receipt
from voynich.source_action_proposal import ReadingEnvironment


def test_linear_decoder_matches_original_for_unequal_records_and_dead_paths():
    records = tuple(r for n in (1, 2) for r in itertools.product(range(2), repeat=n))
    for panel in itertools.product(records, repeat=2):
        env = ReadingEnvironment(panel, rows=2, glyphs=2)
        source, config = fixture_source(True), RegrowthConfig(grid_bits=8)
        sampler = SourceRegrowth(source, env, config)
        for state, actions, _, _, _ in enumerate_paths(source, env, config):
            receipt = path_receipt(sampler.path(forced_actions=actions))
            assert linear_literal(env, receipt) == state
    env = ReadingEnvironment(((0, 1, 0, 1),), rows=1, glyphs=2)
    state = env.advance(env.initial, 0)
    bad = {'actions': [0], 'key': list(state.key), 'offsets': list(state.offsets),
           'texts': [list(t) for t in state.texts], 'complete': False}
    assert linear_literal(env, bad) == state
    for name, value in (('actions', [False]), ('offsets', [0]), ('key', [-1]), ('complete', 0)):
        with pytest.raises(ValueError):
            linear_literal(env, {**bad, name: value})


def test_allocation_and_thresholds_are_strict():
    allocation = expected_cells()
    assert len(allocation) == 388 and len(set(allocation)) == 388
    assert len({s+i for s, i, _ in allocation}) == 194
    score = {'used_rows': 10, 'used_matches': 9, 'true_letters': 100, 'edits': 10,
         'record_attempts': 10, 'exact_records': 5, 'episodes': 10, 'complete_used_key': 5}
    metrics = {str(s): {'regrowth': score, 'independence': {**score, 'edits': 11, 'used_matches': 8}}
               for s in {a[0] for a in allocation}}
    initial = {**score, 'edits': 12, 'used_matches': 7}
    assert gate_summary(metrics, initial)['all_recovery_gates_pass']
    changed = copy.deepcopy(metrics)
    changed[next(iter(changed))]['regrowth']['used_matches'] = 8
    assert not gate_summary(changed, initial)['all_recovery_gates_pass']
    with pytest.raises(AssertionError):
        summaries([])


def test_longer_runner_auditor_json_pipeline_on_artificial_source(tmp_path, monkeypatch):
    import gzip
    import json

    from scripts import audit_reading_regrowth_recovery001 as audit
    from scripts import run_reading_regrowth_recovery001 as run
    from tests.test_reading_regrowth_admission import test_runner_and_audit_entire_json_pipeline_on_artificial_source
    from voynich.joint_key_training import EpisodeSampler
    from voynich.source_action_training import action_episode

    # Reuse the already qualified synthetic source/receipt fixture, no empirical data.
    test_runner_and_audit_entire_json_pipeline_on_artificial_source(tmp_path, monkeypatch)
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'OUT', tmp_path/'results'/'recovery')
    monkeypatch.setattr(run, 'BULK', tmp_path/'outputs'/'recovery')
    monkeypatch.setattr(run, 'PATHS', ['fixture.json'])
    monkeypatch.setattr(run, 'CASES', 3)
    monkeypatch.setattr(run, 'POSITIVE', 2)
    monkeypatch.setattr(run, 'MOVES', {'independence': 3, 'regrowth': 4})
    source = run.admitted.load_source()[0]
    selection = {'counts': run.admitted.prior.training.artifact(tmp_path/'fixture.json'), 'inputs': []}
    monkeypatch.setattr(run, 'load_source', lambda: (source, selection))
    monkeypatch.setattr(audit, 'load_source', lambda: (source, selection))
    monkeypatch.setattr(run, 'load_archive', run.admitted.load_archive)
    monkeypatch.setattr(audit, 'load_archive', run.admitted.load_archive)
    episode_sampler = EpisodeSampler(['a'*400])
    # Real known-answer metric interface, including canonical source/key metadata.
    episode = episode_sampler.make([{'segment': 0, 'start': 0, 'length': 64}]*2, [7]*23)
    env, truth = action_episode(episode_sampler, episode)
    prior_prediction = {'status': 'complete_path', 'actions': list(truth[1]), 'key': list(truth[2].key),
                        'texts': [list(t) for t in truth[2].texts]}
    initial = run.admitted.prior.training.save_new(tmp_path/'outputs'/'recovery-initial.json.gz',
        {'records': [list(r) for r in env.records], 'prediction': prior_prediction}, compressed=True)
    allocated = [('positive' if i < 2 else 'null', i, env.records) for i in range(3)]
    initials = [{'case': i, 'kind': kind, 'paired_case': paired, 'archive': initial}
                for i, (kind, paired, _) in enumerate(allocated)]
    monkeypatch.setattr(run.admitted, 'load_inputs', lambda: (
        {}, [episode]*2, episode_sampler, allocated, initials))
    result = run.recover('artificial-recovery-fixture')
    assert result['not_historical_or_posterior_convergence_gate']
    saved = json.loads((run.OUT/'result.json').read_text())
    assert len(saved['cells']) == 12
    assert sum(c['summary']['attempts'] for c in saved['cells']) == 42
    assert saved['initial_metrics']['exact_records'] == 4
    assert audit.audit('artificial-recovery-fixture') == result
    assert json.loads((run.OUT/'audit.json').read_text())['attempts'] == 42
    # Every cell archive is retained, including failed proposals.
    for c in saved['cells']:
        payload = json.loads(gzip.decompress((tmp_path/c['archive']['path']).read_bytes()))
        assert len(payload['proposals']) == run.MOVES[c['arm']]
