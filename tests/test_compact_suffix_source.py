import itertools

import numpy as np
import pytest

from voynich.compact_suffix_source import CompactSuffixSource, fit_compact, lookup
from voynich.recurrent_latin_source import chunks
from voynich.sparse_suffix_source import SuffixSource, collect_counts


def literal_counts(source):
    result = {}
    alphabet = source.alphabet
    for depth, level in enumerate(source.levels):
        for code, count in zip(level['joints'], level['frequencies']):
            value = int(code)
            letters = []
            for _ in range(depth + 1):
                value, char = divmod(value, len(alphabet))
                letters.append(alphabet[char])
            text = ''.join(reversed(letters))
            result.setdefault(text[:-1], {})[text[-1]] = int(count)
    return result


@pytest.mark.parametrize('order', [0, 1, 3, 8, 12])
@pytest.mark.parametrize('minimum', [1, 4, 10])
def test_compact_all_counts_and_all_scores_match_literal_reference(order, minimum):
    training = ['aaabcaaabcaaabc', 'bbbcbbbc', 'aa', 'cb']
    reference = {c: row for c, row in collect_counts(training, 'abc', order).items()
                 if not c or sum(row.values()) >= minimum}
    compact = fit_compact(training, 'abc', order, minimum)
    assert literal_counts(compact) == reference
    texts = [''.join(chars) for n in range(1, 5) for chars in itertools.product('abc', repeat=n)]
    for row in compact.score(texts, [.25, 16., 64., 1024.], length=3):
        source = SuffixSource('abc', order, row['tau'], reference)
        expected = source.bits([s for _, s in chunks(texts, 3)])
        assert row['selection_bits'] == pytest.approx(expected, abs=1e-10, rel=0)
        assert row['characters'] == sum(map(len, texts))


def test_archive_exact_roundtrip_and_no_overwrite(tmp_path):
    source = fit_compact(['abcabcabc'], 'abc', 12, 2)
    path = tmp_path / 'counts.npz'
    source.save(path)
    restored = CompactSuffixSource.load(path)
    assert literal_counts(source) == literal_counts(restored)
    assert source.score(['cbacba'], [64.]) == restored.score(['cbacba'], [64.])
    with pytest.raises(FileExistsError):
        source.save(path)


def test_lookup_absent_zero_and_maximum_keys():
    keys = np.array([0, 7, 2**64 - 1], dtype=np.uint64)
    values = np.array([6, 2, 3], dtype=np.uint64)
    assert lookup(keys, values, np.array([0, 1, 7, 8, 2**64 - 1], dtype=np.uint64)).tolist() == [6, 0, 2, 0, 3]
    assert lookup(keys[:0], values[:0], [1, 3]).tolist() == [0, 0]


def test_high_code_leading_zero_and_unknown_next_letter():
    alphabet = 'abcdefghiklmnopqrstuxyz'
    text = 'z' * 14 + 'a' * 14
    model = fit_compact([text] * 4, alphabet, 12, 4)
    assert literal_counts(model) == collect_counts([text] * 4, alphabet, 12)
    assert np.isfinite(model.score(['yyyyyy', 'a', 'zzzz'], [64.])[0]['selection_bits'])


def test_count_corruption_and_caps_fail():
    source = fit_compact(['abcabcabc'], 'abc', 3, 2)
    source.levels[0]['totals'][0] += 1
    with pytest.raises(ValueError, match='totals'):
        source.validate()
    with pytest.raises(RuntimeError, match='resource cap'):
        fit_compact(['abcabcabc'], 'abc', 3, 1, max_contexts=2)
    with pytest.raises(RuntimeError, match='resource cap'):
        fit_compact(['abcabcabc'], 'abc', 3, 1, max_array_bytes=8)
    for records, alphabet, order in [([], 'ab', 1), (['ab'], 'aa', 1), (['ab'], 'ab', 13)]:
        with pytest.raises(ValueError):
            fit_compact(records, alphabet, order)
