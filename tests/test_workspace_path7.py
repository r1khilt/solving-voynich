"""Panel and frozen selection checks for PATH-0007."""

from voynich.workspace.path7_campaign import choose, counts
from voynich.workspace.path7_tasks import path7_tasks


def test_balanced_styles_and_inverse_pairs():
    first, second = path7_tasks('discovery'), path7_tasks('confirmation')
    assert len(first) == len(second) == 72
    assert len({row['prompt'] for row in first + second}) == 144
    for group in (first, second):
        lookup = {row['prompt']: row for row in group}
        for style in ('colon', 'arrow'):
            chosen = [row for row in group if row['style'] == style]
            assert sum(row['family'] == 'composed' for row in chosen) == 24
            assert sum(row['family'] == 'copy' for row in chosen) == 12
        for row in group:
            opposite = lookup[row['donor_prompt']]
            assert opposite['donor_prompt'] == row['prompt']
            assert opposite['answer'] == row['donor_answer']


def test_selection_precedence():
    rows = [{'style': 'colon', 'family': 'composed', 'source_correct': True,
             'donor_correct': True, 'first_token_differs': True}]
    assert counts(rows, 'colon')['eligible'] == 1
    assert choose({'colon': {'eligible': 1, 'source_copy': 0},
                   'arrow': {'eligible': 1, 'source_copy': 0}}) == 'colon'
