import pytest

from voynich.latin_source_corpus import extract, normalize, slice_records, remove_overlaps


def xml(body):
    return ('<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>'
            '<div type="edition" xml:lang="lat">' + body + '</div></body></text></TEI>').encode()


def test_preserves_inline_words_omits_notes_and_never_double_counts_verse():
    records, audit = extract(xml('<p>Ro<hi>ma</hi><note>English editor</note> fuit.'
                                 '<quote><l>arma</l><l>uirumque</l></quote></p>'), work_id='test')
    assert [r['text'] for r in records] == ['romafuitarmauirumque']
    assert audit['counts']['retained_units'] == 1


@pytest.mark.parametrize('element', ['gap', 'foreign', 'unclear', 'add', 'del', 'choice', 'app', 'abbr'])
def test_quarantines_whole_uncertain_unit_and_resets_context(element):
    records, audit = extract(xml(f'<p>arma</p><p>a<{element}>foo</{element}>b</p><p>cano</p>'), work_id='x')
    assert [r['text'] for r in records] == ['arma', 'cano']
    assert audit['counts']['quarantined_units'] == 1


def test_structural_boundaries_and_orphan_quote_are_accounted():
    body = '<div subtype="book"><head>Title</head><p>arma</p><quote>cano</quote></div>'
    body += '<div subtype="book"><p>roma</p></div>'
    records, _ = extract(xml(body), work_id='x')
    assert [r['text'] for r in records] == ['armacano', 'roma']
    with pytest.raises(ValueError, match='Unaccounted'):
        extract(xml('<div>unwrapped text</div>'), work_id='x')


def test_unknown_markup_is_never_silently_kept():
    records, audit = extract(xml('<p>arma<mystery>abc</mystery></p>'), work_id='x')
    assert records == [] and audit['counts']['reason_mystery'] == 1
    with pytest.raises(ValueError, match='Unhandled'):
        extract(xml('<mystery><p>abc</p></mystery>'), work_id='x')


def test_normalization_has_explicit_rejections_and_numeric_policy():
    assert normalize('Jūlius VĪVIT Æ Œ 123 x4') == ('iuliusuiuitaeoe', 2)
    for text in ['abc [def]', 'λόγος', 'word']:
        with pytest.raises(ValueError):
            normalize(text)


def test_dtd_entities_and_wrong_language_fail():
    with pytest.raises(ValueError, match='DTD'):
        extract(b'<!DOCTYPE TEI [<!ENTITY text "hello">]>' + xml('<p>abc</p>'), work_id='x')
    with pytest.raises(ValueError, match='Latin'):
        extract(xml('<p>abc</p>').replace(b'"lat"', b'"eng"'), work_id='x')


def test_prefix_does_not_join_boundaries_or_silently_shorten():
    rows, _ = extract(xml('<div subtype="book"><p>abc</p></div><p>defghi</p>'), work_id='x')
    sliced = slice_records(rows, 5)
    assert [r['text'] for r in sliced] == ['abc', 'de']
    assert all(sum(r['unit_lengths']) == len(r['text']) for r in sliced)
    with pytest.raises(ValueError, match='enough'):
        slice_records(rows, 50)


def test_overlap_touching_two_units_removes_both_without_joining_neighbors():
    rows, _ = extract(xml('<p>ab</p><p>cdef</p><p>ghik</p><p>lm</p>'), work_id='x')
    kept, removed = remove_overlaps(rows, {'fghi'}, 4)
    assert [r['text'] for r in kept] == ['ab', 'lm']
    assert removed[0]['removed_characters'] == 8
    assert len(removed[0]['removed_unit_paths']) == 2


def test_overlap_at_unit_end_does_not_remove_following_unit():
    rows, _ = extract(xml('<p>abcdef</p><p>ghik</p>'), work_id='x')
    kept, _ = remove_overlaps(rows, {'cdef'}, 4)
    assert [r['text'] for r in kept] == ['ghik']


def test_incorrect_coverage_and_duplicate_width_rejected():
    rows, _ = extract(xml('<p>abcdef</p>'), work_id='x')
    with pytest.raises(ValueError):
        remove_overlaps(rows, {'short'}, 4)
    rows[0]['unit_lengths'] = [5]
    with pytest.raises(ValueError, match='coverage'):
        remove_overlaps(rows, set(), 4)
