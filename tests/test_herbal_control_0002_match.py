"""Synthetic controls for the frozen historical-image matching rule."""

from scripts.herbal_control_0002_match import CLASSES, MANUSCRIPTS, score
from scripts.herbal_control_0002_audit import recompute


def test_three_way_assignment_recovers_a_pairwise_mistake():
    rows = [{"manuscript": manuscript, "chapter_class": name}
            for manuscript in MANUSCRIPTS for name in CLASSES]
    matrix = [[0.0 if i == j else
               0.2 if rows[i]["chapter_class"] == rows[j]["chapter_class"] else 1.0
               for j in range(18)] for i in range(18)]
    # One unusually close *wrong* BnF/Egerton pair fools the local rule in
    # both directions; the one-to-one three-way constraint should reject it.
    matrix[0][7] = matrix[7][0] = 0.05

    result = score(rows, matrix)

    assert result["raw_pairwise_correct_out_of_36"] == 34
    assert result["induced_correct_out_of_36"] == 36
    assert result["complete_triplets_out_of_6"] == 6
    assert result["triplets"] == [[i, i + 6, i + 12] for i in range(6)]
    assert result["exact_null_permutations"] == 518_400
    assert result["one_sided_exact_p"] == 1 / 518_400
    assert result["registered_fresh_panel_feasibility_pass"] is True
    audit = recompute(rows, matrix)
    assert audit["triplets"] == result["triplets"]
    assert audit["raw_pairwise_correct_out_of_36"] == result["raw_pairwise_correct_out_of_36"]
    assert audit["induced_correct_out_of_36"] == result["induced_correct_out_of_36"]
    assert audit["exact_null_complete_triplet_histogram"] == result["exact_null_complete_triplet_histogram"]
    assert audit["one_sided_exact_p"] == result["one_sided_exact_p"]


def test_no_cross_manuscript_signal_fails_the_gate():
    # With all cross-manuscript distances equal, the lexicographic matching
    # rule cannot exploit labels or infer a chapter correspondence.
    orders = (range(6), (3, 4, 5, 0, 1, 2), (1, 0, 3, 2, 5, 4))
    rows = [{"manuscript": manuscript, "chapter_class": CLASSES[index]}
            for manuscript, order in zip(MANUSCRIPTS, orders, strict=True)
            for index in order]
    matrix = [[0.0 if i == j else 1.0 for j in range(18)] for i in range(18)]

    result = score(rows, matrix)

    assert result["second_minus_best_cost"] == 0
    assert result["raw_pairwise_correct_out_of_36"] == 0
    assert result["complete_triplets_out_of_6"] == 0
    assert result["registered_fresh_panel_feasibility_pass"] is False
