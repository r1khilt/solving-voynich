import json
import time

import numpy as np
import pytest
import torch

from voynich.long_context import (LCBlocks, lc_baselines, lc_config, lc_group_scores, lc_plan,
                                  lc_prefix, lc_score, lc_targets, lc_train_one, lc_update, lc_weight_digest)
from voynich.model import VoynichTransformer
from voynich.runtime import PageWindows
from voynich.tokenizer import EVATokenizer


def lc_fixture(tmp_path):
    texts = [('ab cd\nef gh\n' * 200)[:2300], ('ab cd ef\ngh\n' * 14)[:160]]
    tokenizer = EVATokenizer.fit(texts)
    tokenizer.save(tmp_path/'tokenizer.json')
    for split in ['train', 'validation']:
        rows = [{'text': text, 'split': split, 'page_id': f'{split}{i}', 'leaf_id': f'{split}leaf{i}',
                 'metadata': {'page_variables': {'I': 'H'}}} for i, text in enumerate(texts)]
        (tmp_path/f'{split}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return tokenizer


def test_matched_source_blocks_preserve_targets_and_page_coordinates(tmp_path):
    tokenizer = lc_fixture(tmp_path)
    blocks = LCBlocks(tmp_path)
    assert len(blocks.inputs) == 3
    original, targets = blocks.batch([0, 1, 2])
    merged, merged_targets = blocks.batch([0, 1, 2], 'merge_locus_boundary')
    torch.testing.assert_close(targets, merged_targets)
    changed = original.ne(merged)
    torch.testing.assert_close(changed, original.eq(tokenizer.line_id))
    assert torch.all(merged[changed] == tokenizer.space_id)
    for width in [256, 512, 1024, 2048]:
        torch.testing.assert_close(targets.reshape(-1, width).reshape_as(targets), targets)
    for row, coordinate in enumerate(blocks.coordinates):
        page = blocks.data.sequences[coordinate['page_id']]
        start, length = coordinate['start'], coordinate['length']
        assert original[row, :length].tolist() == page[start:start+length]
        assert torch.all(targets[row, length:] == -100)


def test_common_targets_never_include_future_or_cross_page_and_report_coverage(tmp_path):
    lc_fixture(tmp_path)
    validation = PageWindows(tmp_path, 'validation', 2048)
    records = lc_targets(validation)
    assert sum('primary' in record['groups'] for record in records) == 32
    assert sum('full2048' in record['groups'] for record in records) == 32
    for record in records:
        for context in [256, 512, 1024, 2048]:
            prefix = lc_prefix(validation, record, context, 'original')
            position = record['position']
            assert prefix == validation.sequences[record['page_id']][max(0, position-context):position]
            assert len(prefix) == min(context, position)
        clean = lc_prefix(validation, record, 2048, 'original')
        shuffled = lc_prefix(validation, record, 2048, 'original', 'remote_shuffle_keep256')
        assert clean[-256:] == shuffled[-256:]
        assert sorted(clean) == sorted(shuffled)
    report = lc_group_scores(np.ones(len(records)), records)
    assert len(report['primary']['pages']) == 2
    assert len(report['full2048']['pages']) == 1


def test_gradient_accumulation_equals_single_global_loss_with_unequal_padding():
    torch.set_num_threads(2)
    torch.manual_seed(3)
    config = lc_config('smoke', 8, vocab=12)
    config.dropout = 0
    first = VoynichTransformer(config)
    second = VoynichTransformer(config)
    second.load_state_dict(first.state_dict())
    x = torch.randint(1, 12, (2, 16))
    y = torch.randint(1, 12, (2, 16))
    x[1, 8:] = 0
    y[1, 8:] = -100
    y[0, 3:6] = -100
    # Identical optimizer and clipping. Only microbatch partition changes.
    left = lc_update(first, torch.optim.SGD(first.parameters(), lr=.01), x, y, context=8, micro_tokens=8)
    right = lc_update(second, torch.optim.SGD(second.parameters(), lr=.01), x, y, context=8, micro_tokens=64)
    assert left['scored_tokens'] == right['scored_tokens'] == 21
    for p, q in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(p, q, atol=2e-7, rtol=1e-5)
    with pytest.raises(ValueError, match='No scorable'):
        lc_update(first, torch.optim.SGD(first.parameters(), lr=.01), x, y.fill_(-100), context=8)


def test_validation_padding_matches_individual_prefixes_and_no_future_leakage(tmp_path):
    lc_fixture(tmp_path)
    validation = PageWindows(tmp_path, 'validation', 2048)
    records = lc_targets(validation, per_page=3, long_per_page=0)
    torch.manual_seed(3)
    model = VoynichTransformer(lc_config('smoke', 256, vocab=validation.tokenizer.vocab_size)).eval()
    batched = lc_score(model, validation, records, 'cpu', 256, 'original', batch_size=4)
    single = lc_score(model, validation, records, 'cpu', 256, 'original', batch_size=1)
    np.testing.assert_allclose(batched['losses_bits'], single['losses_bits'], atol=1e-6)
    record = records[0]
    before = lc_prefix(validation, record, 256, 'original')
    validation.sequences[record['page_id']][record['position']:] = [0]*500
    assert lc_prefix(validation, record, 256, 'original') == before


def test_context_capacity_and_initialization_controls():
    for size, expected in [('main', 1814208), ('compact', 430720)]:
        digests = []
        for width in [256, 512, 1024, 2048]:
            torch.manual_seed(10111)
            model = VoynichTransformer(lc_config(size, width))
            assert model.parameter_count == expected
            digests.append(lc_weight_digest(model))
        assert len(set(digests)) == 1
    plan = lc_plan({'seeds': [1, 2, 3]})
    assert len(plan) == len({row['name'] for row in plan}) == 21


def test_training_smoke_and_fixed_baselines_do_not_require_manuscript_test(tmp_path):
    lc_fixture(tmp_path)
    blocks = LCBlocks(tmp_path, block_width=32)
    validation = PageWindows(tmp_path, 'validation', 32)
    records = lc_targets(validation, per_page=2, long_per_page=0)
    baselines = lc_baselines(blocks.data, validation, records)
    assert len(baselines['losses_bits']) == 6
    assert all(len(row) == len(records) for row in baselines['losses_bits'].values())
    config = {'steps': 2, 'eval_interval': 1, 'warmup_steps': 1, 'learning_rate': .0006,
              'global_source_blocks': 2, 'microbatch_tokens': 32, 'max_process_rss_gib': 12}
    condition = {'seed': 10111, 'size': 'smoke', 'context': 16, 'representation': 'original', 'name': 'smoke'}
    result = lc_train_one(condition, config, blocks, validation, records, {}, tmp_path/'run', 'cpu',
                          time.monotonic()+30, {'requested': False})
    assert result['completed_steps'] == 2
    assert result['stop_reason'] == 'step_budget'
    assert not result['test_evaluated']
    assert (tmp_path/'run'/'last.pt').is_file()
    assert not (tmp_path/'test.jsonl').exists()
