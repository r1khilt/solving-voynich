import json

import numpy as np

from voynich.context_analysis import sample_contexts
from voynich.context_controls import frequency_adapt, make_conditions, score_baselines, section
from voynich.runtime import PageWindows
from voynich.synthetic import prepare


def fixture_data(tmp_path):
    prepare(tmp_path, 'cycle_null', counts=(4, 2, 2))
    for split in ['train', 'validation']:
        path = tmp_path/f'{split}.jsonl'
        pages = [json.loads(line) for line in path.read_text().splitlines()]
        for i, p in enumerate(pages):
            p['metadata']['page_variables'] = {'I': 'H' if i % 2 else 'B'}
        path.write_text(''.join(json.dumps(p)+'\n' for p in pages))
    return PageWindows(tmp_path, 'train', 256), PageWindows(tmp_path, 'validation', 256)


def test_donors_and_matching_preserve_registered_constraints(tmp_path):
    train, val = fixture_data(tmp_path)
    examples = sample_contexts(val, per_page=2)
    sections = {p['page_id']: section(p) for p in val.pages}
    conditions, donors = make_conditions(examples, train, sections, repeats=2)
    assert len(conditions) == 11
    for rows in conditions.values():
        for repetition in rows:
            for row, example in zip(repetition, examples, strict=True):
                np.testing.assert_array_equal(row[-16:], example['prefix'][-16:])
    by_key = {(d['example'], d['repeat'], d['condition']): d for d in donors}
    for record in donors:
        e = examples[record['example']]
        assert record['donor']['page_id'] in train.sequences
        assert record['donor']['page_id'] not in val.sequences
        same = record['condition'].startswith('same')
        assert (record['donor_section'] == sections[e['page_id']]) == same
        if record['condition'].endswith('matched'):
            random = by_key[(record['example'], record['repeat'], record['condition'].replace('matched', 'random'))]
            assert record['histogram_total_variation'] <= random['histogram_total_variation']
    for name in ['same_random', 'same_matched', 'different_random', 'different_matched']:
        np.testing.assert_array_equal(np.sort(conditions[name+'_ordered'], axis=-1),
                                      np.sort(conditions[name+'_shuffled'], axis=-1))


def test_histogram_predictor_is_normalized_order_invariant_and_identity_at_zero():
    p = np.array([.5, .3, .2])
    prior = np.array([.2, .3, .5])
    prefix = [0, 0, 1, 2, 0]
    np.testing.assert_allclose(frequency_adapt(p, prefix, prior, 0), p)
    for alpha in [.5, 1]:
        a = frequency_adapt(p, prefix, prior, alpha)
        assert np.all(a > 0)
        np.testing.assert_allclose(a.sum(), 1)
        np.testing.assert_array_equal(a, frequency_adapt(p, prefix[::-1], prior, alpha))
        assert a[0] > p[0]


def test_all_baselines_score_same_targets_without_neural_outputs(tmp_path):
    train, val = fixture_data(tmp_path)
    examples = sample_contexts(val, per_page=2)
    sections = {p['page_id']: section(p) for p in val.pages}
    result = score_baselines(examples, train, sections)
    assert len(result['losses_bits']) == 4
    for losses in result['losses_bits'].values():
        assert len(losses) == 4
        assert np.isfinite(losses).all()
