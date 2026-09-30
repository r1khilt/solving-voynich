from collections import Counter
import hashlib

import pytest

from voynich.fresh_reader_panel import allocate_windows, encode_known, reader_gate, shuffled_plaintext


def records():
    texts = ['a' * 223, 'abcd' * 257, 'bcad' * 333]
    return [{'id': str(i), 'text': text, 'sha256': hashlib.sha256(text.encode()).hexdigest()}
            for i, text in enumerate(texts)]


def test_windows_reproducible_contained_nonoverlapping_and_no_short_segment():
    rows = records()
    first = allocate_windows(rows, count=5, seed=33)
    assert first == allocate_windows(rows, count=5, seed=33)
    assert all(w['record_index'] != 0 and len(w['text']) == 224 and w['offset'] % 512 == 0 for w in first)
    assert len({(w['record_index'], w['offset']) for w in first}) == 5
    for window in first:
        row = rows[window['record_index']]
        assert window['text'] == row['text'][window['offset']:window['offset'] + 224]
        assert window['record_sha256'] == row['sha256']


def test_insufficient_windows_and_overlapping_stride_fail_without_redraw():
    with pytest.raises(ValueError, match='Insufficient'):
        allocate_windows(records(), count=100)
    with pytest.raises(ValueError):
        allocate_windows(records(), count=2, stride=223)


def test_plaintext_permutation_null_preserves_counts_and_encoded_length():
    text = 'aaabbcd' * 20
    shuffled = shuffled_plaintext(text, 33)
    assert shuffled != text and Counter(shuffled) == Counter(text)
    assert shuffled == shuffled_plaintext(text, 33)
    units = ['x', 'xx', 'y', 'yx']
    assert len(encode_known(text, 'abcd', units)) == len(encode_known(shuffled, 'abcd', units))


def test_reader_gate_exact_boundaries_and_zero_baseline():
    good = reader_gate([8] * 16, [12] * 16)
    assert good['pass']
    assert not reader_gate([9] * 16, [12] * 16)['pass']  #144/7168 exceeds2%.
    assert not reader_gate([23] + [0] * 15, [100] * 16)['pass']
    assert not reader_gate([8] * 16, [10] * 16)['pass']
    perfect = reader_gate([0] * 16, [0] * 16)
    assert perfect['pass'] and perfect['both_perfect'] and perfect['relative_edit_reduction'] is None
    with pytest.raises(ValueError):
        reader_gate([0] * 15, [0] * 16)
