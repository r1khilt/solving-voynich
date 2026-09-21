import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from voynich.long_context import LCBlocks, lc_coverage, lc_targets
from voynich.runtime import PageWindows, corpus_identity, digest


root = Path('outputs/EXP-0010/campaign')
manifest = json.loads((root/'manifest.json').read_text())
blocks = LCBlocks('data/processed/zl3b')
validation = PageWindows('data/processed/zl3b', 'validation', 2048)
assert manifest['environment']['git_dirty'] is False
assert manifest['test_evaluated'] is False
assert corpus_identity('data/processed/zl3b') == manifest['corpus_identity']
assert blocks.coordinates == manifest['train_blocks']
assert lc_targets(validation) == manifest['targets']
assert lc_coverage(manifest['targets']) == manifest['target_coverage']
for name, expected in manifest['environment']['source_sha256'].items():
    assert digest(name) == expected, name

expected_exposure = {}
for seed in manifest['config']['seeds']:
    rng = torch.Generator().manual_seed(seed+100)
    stream = hashlib.sha256()
    count = 0
    totals = blocks.targets.ne(-100).sum(-1)
    for _ in range(manifest['config']['steps']):
        indices = torch.randint(len(blocks.inputs), (manifest['config']['global_source_blocks'],), generator=rng)
        stream.update(np.asarray(indices.tolist(), dtype='<i8').tobytes())
        count += int(totals[indices].sum())
    expected_exposure[seed] = {'scored_training_tokens': count, 'sampled_block_indices_sha256': stream.hexdigest()}

summaries = {}
for path in sorted((root/'runs').glob('*/summary.json')):
    result = json.loads(path.read_text())
    condition = result['condition']
    assert result['test_evaluated'] is False
    assert digest(path.parent/'last.pt') == result['last_checkpoint_sha256']
    if result['stop_reason'] == 'step_budget':
        assert result['completed_steps'] == manifest['config']['steps']
        for name, value in expected_exposure[condition['seed']].items():
            assert result[name] == value, (condition, name)
    summaries[path.parent.name] = result

for size in ['main', 'compact']:
    for seed in manifest['config']['seeds']:
        matching = [value for value in summaries.values() if value['condition']['size'] == size
                    and value['condition']['seed'] == seed]
        assert len({value['initial_weights_sha256'] for value in matching}) <= 1

primary = [i for i, record in enumerate(manifest['targets']) if 'primary' in record['groups']]
rows = {}
for name, result in summaries.items():
    final = result['final']['original']
    row = {'steps': result['completed_steps'], 'stop_reason': result['stop_reason'],
           'primary': final['scores']['primary'], 'full2048': final['scores']['full2048'],
           'seconds': result['elapsed_seconds'], 'tokens': result['scored_training_tokens']}
    original = np.asarray(final['losses_bits'])
    for condition in ['cap256', 'remote_shuffle_keep256']:
        if condition in result['final']:
            shifted = np.asarray(result['final'][condition]['losses_bits'])
            row[condition+'_damage_primary'] = float((shifted-original)[primary].mean())
    rows[name] = row

output = {'verified_complete_run_count': len(summaries), 'expected_exposure_by_seed': expected_exposure,
          'source_commit': manifest['environment']['git_commit'], 'coordinates_and_source_verified': True,
          'results': rows}
Path('outputs/EXP-0010/audit-draft/audit.json').write_text(json.dumps(output, indent=2)+'\n')
print(json.dumps({'verified_complete_run_count': len(summaries), 'expected_exposure': expected_exposure}, indent=2))
