import numpy as np

from voynich.order_scale import make_conditions, matched_group_corruptions, nonidentity_permutation


def test_group_and_character_controls_match_exact_changed_positions():
    # Leading/trailing groups are fragments, complete groups vary only within equal lengths.
    row = np.array([8, 8, 0, 1, 2, 0, 3, 4, 0, 5, 6, 0, 1, 3, 0, 9, 9])
    grouped, scrambled, stats = matched_group_corruptions(row, {0}, np.random.default_rng(1))
    assert stats['changed_fraction'] > 0
    np.testing.assert_array_equal(grouped[:3], row[:3])
    np.testing.assert_array_equal(grouped[-3:], row[-3:])
    np.testing.assert_array_equal(grouped == 0, row == 0)
    np.testing.assert_array_equal(scrambled == 0, row == 0)
    np.testing.assert_array_equal(grouped != row, scrambled != row)
    np.testing.assert_array_equal(np.sort(grouped), np.sort(row))
    np.testing.assert_array_equal(np.sort(scrambled), np.sort(row))
    originals = sorted(tuple(row[i:i+2]) for i in [3, 6, 9, 12])
    assert originals == sorted(tuple(grouped[i:i+2]) for i in [3, 6, 9, 12])


def test_no_eligible_groups_remains_identity_not_selected_out():
    row = np.array([1, 2, 0, 3, 0, 4, 5, 6])
    grouped, scrambled, stats = matched_group_corruptions(row, {0}, np.random.default_rng(2))
    np.testing.assert_array_equal(grouped, row)
    np.testing.assert_array_equal(scrambled, row)
    assert stats['changed_fraction'] == 0


def test_all_scale_corruptions_preserve_tokens_and_local_context():
    examples = [{'prefix': (list(range(14))*10)[:128]}]
    conditions, exposure = make_conditions(examples, {0}, repeats=2)
    for rows in conditions.values():
        for repeat in rows:
            np.testing.assert_array_equal(np.sort(repeat[0]), np.sort(examples[0]['prefix']))
            np.testing.assert_array_equal(repeat[0, -16:], examples[0]['prefix'][-16:])
    assert len(exposure) == 2
    for size in [2, 4, 7]:
        order = nonidentity_permutation(size, np.random.default_rng(1))
        assert not np.array_equal(order, np.arange(size))
        np.testing.assert_array_equal(np.sort(order), np.arange(size))
