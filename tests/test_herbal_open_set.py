"""Known-answer controls for reject-capable one-to-one image matching."""

from itertools import permutations
import random

import pytest

from voynich.herbal_open_set import (
    alternate_prediction_margin,
    batch_predictions,
    candidate_thresholds,
    class_bootstrap,
    choose_primary,
    directional_balanced_accuracy,
    independent_predictions,
    prepare_direction_panels,
    rectangular_assignment,
)


def test_rectangular_assignment_agrees_with_bruteforce() -> None:
    rng = random.Random(3003)
    for rows, columns in ((2, 3), (3, 5), (4, 6)):
        for _ in range(8):
            costs = [[rng.randrange(-20, 40) / 10 for _ in range(columns)]
                     for _ in range(rows)]
            assignment, objective = rectangular_assignment(costs)
            brute = min(sum(costs[i][j] for i, j in enumerate(cols))
                        for cols in permutations(range(columns), rows))
            assert len(set(assignment)) == rows
            assert objective == pytest.approx(brute, abs=1e-10)


def test_batch_nulls_reject_unknown_and_avoid_many_to_one_collapse() -> None:
    costs = [[0.10, 0.20], [0.11, 0.90], [0.85, 0.80]]
    assert independent_predictions(costs, 0.50) == [0, 0, None]
    predictions, objective = batch_predictions(costs, 0.50)
    assert predictions == [1, 0, None]
    assert objective == pytest.approx(0.81)
    assert directional_balanced_accuracy(predictions, [1, 0, None], 2, 1) == 1.0


def test_alternate_margin_ignores_null_permutations_and_matches_bruteforce() -> None:
    costs = [[0.1, 0.8], [0.2, 0.9], [0.9, 0.8]]
    threshold = 0.5
    result = alternate_prediction_margin(costs, threshold)
    rows = len(costs)
    matrix = [[threshold] * rows + row for row in costs]
    patterns = {}
    for columns in permutations(range(rows + len(costs[0])), rows):
        prediction = tuple(None if col < rows else col - rows for col in columns)
        score = sum(matrix[i][col] for i, col in enumerate(columns))
        patterns[prediction] = min(patterns.get(prediction, float("inf")), score)
    ordered = sorted(patterns.items(), key=lambda item: item[1])
    assert result["optimum_cost"] == pytest.approx(ordered[0][1])
    assert result["alternate_cost"] == pytest.approx(ordered[1][1])
    assert result["prediction_gap"] == pytest.approx(ordered[1][1] - ordered[0][1])
    assert result["near_tie_1e_minus_9"] is False


def test_alternate_margin_reports_distinct_near_tied_predictions() -> None:
    result = alternate_prediction_margin([[0.5]], 0.5)
    assert result["prediction_gap"] == 0.0
    assert result["near_tie_1e_minus_9"] is True
    assert result["optimum_predictions"] != result["alternate_predictions"]


def test_primary_selection_can_prefer_strong_simple_method() -> None:
    first = {"bnf": [[0.1, 0.9], [0.9, 0.1], [0.8, 0.9]]}
    second = {"bnf": [[0.9, 0.1], [0.1, 0.9], [0.1, 0.2]]}
    truth = {"bnf": [0, 1, None]}
    selection = choose_primary(first, second, truth, 2, 1)
    assert selection["primary_method"] == "single_first"
    assert selection["methods"]["single_first"]["development_macro_ba"] == 1.0
    assert len(candidate_thresholds(first)) <= 203


def test_directional_metric_rejects_wrong_unknown_count() -> None:
    with pytest.raises(ValueError, match="wrong number"):
        directional_balanced_accuracy([0, None], [0, 1], 1, 1)


def test_evaluation_uses_development_medians_and_disjoint_rows() -> None:
    def make_rows(split: str) -> list[dict]:
        return [{"role": f"{split}_{role}", "chapter_class": chapter,
                 "manuscript": manuscript}
                for role, chapters in (("known", ("a", "b")), ("unknown", ("c",)))
                for chapter in chapters
                for manuscript in ("bnf", "egerton", "casanatense")]

    rows = make_rows("development")
    distance = [[0.0 if i == j else abs(i - j) + 1.0 for j in range(9)]
                for i in range(9)]
    first, second, truth, medians = prepare_direction_panels(
        distance, rows, ["a", "b"], ["c"], "development")
    assert len(medians) == 6
    assert all(len(first[m]) == len(second[m]) == 3 for m in first)
    assert truth["bnf"] == [0, 1, None]

    evaluation_rows = make_rows("evaluation")
    multiplied = [[value * 100 for value in row] for row in distance]
    evaluation_first, _, _, frozen = prepare_direction_panels(
        multiplied, evaluation_rows, ["a", "b"], ["c"], "evaluation", medians)
    assert frozen == medians
    assert evaluation_first["bnf"][0][0] == pytest.approx(first["bnf"][0][0] * 100)
    with pytest.raises(ValueError, match="frozen split"):
        prepare_direction_panels(distance, rows, ["a", "b"], ["c"], "evaluation", medians)


def test_class_bootstrap_keeps_correlated_directions_and_strongest_comparator() -> None:
    directions = ("bnf", "egerton", "casanatense")
    truth = {m: [0, 1, None] for m in directions}
    perfect = {m: [0, 1, None] for m in directions}
    reject_all = {m: [None, None, None] for m in directions}
    result = class_bootstrap(perfect, truth, 2, 1, [reject_all], replicates=200)
    assert result["primary_ci_95"] == [1.0, 1.0]
    assert result["paired_improvement_ci_95"] == [0.5, 0.5]
    conservative = class_bootstrap(perfect, truth, 2, 1,
                                   [reject_all, perfect], replicates=200)
    assert conservative["paired_improvement_ci_95"] == [0.0, 0.0]
