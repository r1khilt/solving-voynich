"""Reject-capable historical chapter matching for HERBAL-CONTROL-0003.

This module is deliberately independent of image extraction and manuscript
text. The registered study feeds it label-free distance matrices after the
crop/source freeze. Development alone selects the threshold and cost family.
"""

from __future__ import annotations

from math import floor, inf, isfinite
import random
from statistics import median
from typing import TypeAlias


CostMatrix: TypeAlias = list[list[float]]
Prediction: TypeAlias = int | None  # gallery class index, or unknown
METHODS = ("single_first", "single_second", "two_mean", "two_min")
MANUSCRIPTS = ("bnf", "egerton", "casanatense")


def _validate_rectangular(cost: CostMatrix) -> tuple[int, int]:
    if not cost or not cost[0]:
        raise ValueError("cost matrix must be nonempty")
    columns = len(cost[0])
    if any(len(row) != columns for row in cost):
        raise ValueError("cost matrix is not rectangular")
    if any(not isfinite(value) for row in cost for value in row):
        raise ValueError("cost matrix contains nonfinite values")
    return len(cost), columns


def rectangular_assignment(cost: CostMatrix) -> tuple[list[int], float]:
    """Minimum-cost row→distinct-column assignment, `rows <= columns`.

    Uses the rectangular primal-dual Hungarian algorithm. Input column order
    is the deterministic tie order; callers place null columns first.
    """
    rows, columns = _validate_rectangular(cost)
    if rows > columns:
        raise ValueError("more rows than columns")
    u = [0.0] * (rows + 1)
    v = [0.0] * (columns + 1)
    matched_row = [0] * (columns + 1)
    predecessor = [0] * (columns + 1)
    for row_index in range(1, rows + 1):
        matched_row[0] = row_index
        min_reduced_cost = [inf] * (columns + 1)
        used = [False] * (columns + 1)
        col0 = 0
        while True:
            used[col0] = True
            current_row = matched_row[col0]
            delta = inf
            col1 = 0
            for col in range(1, columns + 1):
                if used[col]:
                    continue
                reduced = cost[current_row - 1][col - 1] - u[current_row] - v[col]
                if reduced < min_reduced_cost[col]:
                    min_reduced_cost[col] = reduced
                    predecessor[col] = col0
                if min_reduced_cost[col] < delta:
                    delta = min_reduced_cost[col]
                    col1 = col
            if not isfinite(delta):
                raise ValueError("no finite assignment")
            for col in range(columns + 1):
                if used[col]:
                    u[matched_row[col]] += delta
                    v[col] -= delta
                else:
                    min_reduced_cost[col] -= delta
            col0 = col1
            if matched_row[col0] == 0:
                break
        while True:
            col1 = predecessor[col0]
            matched_row[col0] = matched_row[col1]
            col0 = col1
            if col0 == 0:
                break
    assignment = [-1] * rows
    for col in range(1, columns + 1):
        if matched_row[col]:
            assignment[matched_row[col] - 1] = col - 1
    if any(col < 0 for col in assignment):
        raise AssertionError("not every row was assigned")
    return assignment, sum(cost[row][col] for row, col in enumerate(assignment))


def batch_predictions(class_costs: CostMatrix, rejection_cost: float) -> tuple[list[Prediction], float]:
    """Decode 36 queries against 24 classes plus one null slot per query."""
    rows, classes = _validate_rectangular(class_costs)
    if not isfinite(rejection_cost):
        raise ValueError("rejection cost must be finite")
    matrix = [[rejection_cost] * rows + row for row in class_costs]
    assigned_columns, total = rectangular_assignment(matrix)
    predictions = [None if col < rows else col - rows for col in assigned_columns]
    named = [class_id for class_id in predictions if class_id is not None]
    if len(set(named)) != len(named) or len(named) > classes:
        raise AssertionError("invalid class reuse")
    return predictions, total


def alternate_prediction_margin(class_costs: CostMatrix, rejection_cost: float) -> dict:
    """Find the cheapest assignment that changes any class/reject prediction.

    Different null-column permutations are deliberately ignored. For each
    query, forbid its chosen named class or require a named class if rejected;
    the minimum across those constrained assignments is the exact runner-up
    *prediction pattern* because every different pattern changes some row.
    """
    rows, classes = _validate_rectangular(class_costs)
    predictions, best_cost = batch_predictions(class_costs, rejection_cost)
    if classes <= 0:
        raise ValueError("at least one gallery class is required")
    biggest = max(abs(rejection_cost), *(abs(value) for row in class_costs for value in row))
    forbidden = (2 * rows + classes + 1) * (biggest + 1.0) * 10.0
    original = [[rejection_cost] * rows + row for row in class_costs]
    second_cost = inf
    second_predictions: list[Prediction] | None = None
    for changed_row, prediction in enumerate(predictions):
        altered = [row[:] for row in original]
        if prediction is None:
            altered[changed_row][:rows] = [forbidden] * rows
        else:
            altered[changed_row][rows + prediction] = forbidden
        columns, objective = rectangular_assignment(altered)
        if any(altered[row][col] == forbidden for row, col in enumerate(columns)):
            raise AssertionError("alternate pattern should be feasible without forbidden cells")
        candidate = [None if col < rows else col - rows for col in columns]
        if candidate == predictions:
            raise AssertionError("constraint did not change prediction pattern")
        if objective < second_cost:
            second_cost, second_predictions = objective, candidate
    if second_predictions is None:
        raise AssertionError("no alternate prediction pattern")
    return {
        "optimum_cost": best_cost,
        "alternate_cost": second_cost,
        "prediction_gap": max(0.0, second_cost - best_cost),
        "optimum_predictions": predictions,
        "alternate_predictions": second_predictions,
        "near_tie_1e_minus_9": second_cost - best_cost <= 1e-9,
    }


def independent_predictions(class_costs: CostMatrix, rejection_cost: float,
                            tie_tolerance: float = 1e-7) -> list[Prediction]:
    _validate_rectangular(class_costs)
    results = []
    for row in class_costs:
        best = min(row)
        if best >= rejection_cost or sum(abs(value - best) <= tie_tolerance for value in row) != 1:
            results.append(None)
        else:
            results.append(row.index(best))
    return results


def fuse_costs(first: CostMatrix, second: CostMatrix, method: str) -> CostMatrix:
    n1, k1 = _validate_rectangular(first)
    n2, k2 = _validate_rectangular(second)
    if (n1, k1) != (n2, k2):
        raise ValueError("reference cost matrices differ in shape")
    if method not in METHODS:
        raise ValueError(f"unknown cost family: {method}")
    if method == "single_first":
        return [row[:] for row in first]
    if method == "single_second":
        return [row[:] for row in second]
    if method == "two_mean":
        return [[(a + b) / 2 for a, b in zip(x, y, strict=True)]
                for x, y in zip(first, second, strict=True)]
    return [[min(a, b) for a, b in zip(x, y, strict=True)]
            for x, y in zip(first, second, strict=True)]


def prepare_direction_panels(
    distance_matrix: CostMatrix,
    row_order: list[dict],
    known_chapters: list[str],
    unknown_chapters: list[str],
    split: str,
    normalizers: dict[str, float] | None = None,
) -> tuple[dict[str, CostMatrix], dict[str, CostMatrix],
           dict[str, list[Prediction]], dict[str, float]]:
    """Make direction-specific costs using only the named split's image rows.

    When `normalizers` is absent, derive all six ordered manuscript-pair
    medians from this panel (development only). Evaluation must supply the
    saved development medians.
    """
    if split not in ("development", "evaluation"):
        raise ValueError("split must be development or evaluation")
    rows, columns = _validate_rectangular(distance_matrix)
    if rows != columns or rows != len(row_order):
        raise ValueError("distance matrix/row-order shape mismatch")
    expected_rows = len(MANUSCRIPTS) * (len(known_chapters) + len(unknown_chapters))
    if rows != expected_rows or len(set(known_chapters + unknown_chapters)) != len(known_chapters) + len(unknown_chapters):
        raise ValueError("split classes or image row count mismatch")
    index = {}
    for i, row in enumerate(row_order):
        key = (row["role"], row["chapter_class"], row["manuscript"])
        if key in index:
            raise ValueError(f"duplicate feature row: {key}")
        index[key] = i
    expected_keys = {(f"{split}_known", chapter, manuscript)
                     for chapter in known_chapters for manuscript in MANUSCRIPTS}
    expected_keys.update((f"{split}_unknown", chapter, manuscript)
                         for chapter in unknown_chapters for manuscript in MANUSCRIPTS)
    if set(index) != expected_keys:
        raise ValueError("feature rows do not match the frozen split")
    query_chapters = known_chapters + unknown_chapters
    truths = {query: list(range(len(known_chapters))) + [None] * len(unknown_chapters)
              for query in MANUSCRIPTS}
    if normalizers is None:
        normalizers = {}
        for query in MANUSCRIPTS:
            query_rows = [index[(f"{split}_{'known' if chapter in known_chapters else 'unknown'}",
                                 chapter, query)] for chapter in query_chapters]
            for source in MANUSCRIPTS:
                if source == query:
                    continue
                source_rows = [index[(f"{split}_known", chapter, source)] for chapter in known_chapters]
                scale = median(distance_matrix[i][j] for i in query_rows for j in source_rows)
                if not isfinite(scale) or scale <= 0:
                    raise ValueError(f"invalid development median for {query}->{source}")
                normalizers[f"{query}->{source}"] = scale
    else:
        expected_pairs = {f"{query}->{source}" for query in MANUSCRIPTS
                          for source in MANUSCRIPTS if query != source}
        if set(normalizers) != expected_pairs or any(not isfinite(x) or x <= 0 for x in normalizers.values()):
            raise ValueError("invalid frozen development normalizers")
        normalizers = dict(normalizers)
    first = {}
    second = {}
    for query in MANUSCRIPTS:
        sources = [source for source in MANUSCRIPTS if source != query]
        matrices = []
        for source in sources:
            scale = normalizers[f"{query}->{source}"]
            values = []
            for chapter in query_chapters:
                role = f"{split}_{'known' if chapter in known_chapters else 'unknown'}"
                i = index[(role, chapter, query)]
                values.append([distance_matrix[i][index[(f"{split}_known", ref, source)]] / scale
                               for ref in known_chapters])
            matrices.append(values)
        first[query], second[query] = matrices
    return first, second, truths, normalizers


def candidate_thresholds(cost_panels: dict[str, CostMatrix]) -> list[float]:
    values = sorted(value for matrix in cost_panels.values() for row in matrix for value in row)
    if not values or not all(isfinite(value) for value in values):
        raise ValueError("missing or nonfinite development class cost")
    positions = []
    for k in range(201):
        pos = (len(values) - 1) * k / 200
        low = floor(pos)
        alpha = pos - low
        positions.append(values[low] * (1 - alpha) + values[min(low + 1, len(values) - 1)] * alpha)
    below = 0.0 if values[0] > 0 else values[0] - 1.0
    return sorted({below, values[-1] + 1.0, *positions})


def directional_balanced_accuracy(predictions: list[Prediction],
                                  truth: list[Prediction], known_count: int,
                                  unknown_count: int) -> float:
    if len(predictions) != len(truth) or len(truth) != known_count + unknown_count:
        raise ValueError("prediction/truth length or class count mismatch")
    if sum(label is None for label in truth) != unknown_count:
        raise ValueError("wrong number of unknown truth rows")
    known_correct = sum(pred == gold for pred, gold in zip(predictions, truth, strict=True)
                        if gold is not None)
    unknown_rejected = sum(pred is None for pred, gold in zip(predictions, truth, strict=True)
                           if gold is None)
    return 0.5 * (known_correct / known_count + unknown_rejected / unknown_count)


def choose_threshold(cost_panels: dict[str, CostMatrix], truths: dict[str, list[Prediction]],
                     known_count: int, unknown_count: int, *, batch: bool) -> tuple[float, float]:
    if set(cost_panels) != set(truths) or not cost_panels:
        raise ValueError("development direction sets differ")
    best_threshold = 0.0
    best_score = -1.0
    for threshold in candidate_thresholds(cost_panels):
        accuracies = []
        for direction, matrix in cost_panels.items():
            predictions = (batch_predictions(matrix, threshold)[0] if batch
                           else independent_predictions(matrix, threshold))
            accuracies.append(directional_balanced_accuracy(
                predictions, truths[direction], known_count, unknown_count))
        score = sum(accuracies) / len(accuracies)
        if score > best_score + 1e-12:
            best_threshold, best_score = threshold, score
    return best_threshold, best_score


def choose_primary(first: dict[str, CostMatrix], second: dict[str, CostMatrix],
                   truths: dict[str, list[Prediction]], known_count: int,
                   unknown_count: int) -> dict:
    if set(first) != set(second):
        raise ValueError("reference directions differ")
    methods = {}
    for method in METHODS:
        costs = {direction: fuse_costs(first[direction], second[direction], method)
                 for direction in first}
        threshold, score = choose_threshold(costs, truths, known_count, unknown_count, batch=True)
        methods[method] = {"threshold": threshold, "development_macro_ba": score}
    selected = max(METHODS, key=lambda name: methods[name]["development_macro_ba"])
    return {"primary_method": selected, "methods": methods}


def _sampled_macro_ba(predictions: dict[str, list[Prediction]],
                      truths: dict[str, list[Prediction]], known_indices: list[int],
                      unknown_indices: list[int]) -> float:
    known_count = len(known_indices)
    unknown_count = len(unknown_indices)
    direction_scores = []
    for direction in truths:
        pred = predictions[direction]
        gold = truths[direction]
        correct_known = sum(pred[i] == gold[i] for i in known_indices)
        rejected_unknown = sum(pred[i] is None for i in unknown_indices)
        direction_scores.append(0.5 * (correct_known / known_count
                                       + rejected_unknown / unknown_count))
    return sum(direction_scores) / len(direction_scores)


def _percentile(sorted_values: list[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    low = floor(position)
    alpha = position - low
    return sorted_values[low] * (1 - alpha) + sorted_values[min(low + 1, len(sorted_values) - 1)] * alpha


def class_bootstrap(
    primary: dict[str, list[Prediction]],
    truths: dict[str, list[Prediction]],
    known_count: int,
    unknown_count: int,
    comparators: list[dict[str, list[Prediction]]] | None = None,
    *,
    replicates: int = 10_000,
    seed: int = 3003,
) -> dict:
    """Stratified class bootstrap preserving each class's three directions.

    When comparators are provided, their best BA within each replicate is the
    conservative comparator for the primary's paired improvement.
    """
    if not primary or set(primary) != set(truths):
        raise ValueError("prediction/truth direction sets differ")
    if comparators is None:
        comparators = []
    if any(set(comp) != set(truths) for comp in comparators):
        raise ValueError("comparator direction sets differ")
    if known_count <= 0 or unknown_count <= 0 or replicates <= 0:
        raise ValueError("bootstrap counts must be positive")
    expected_truth = list(range(known_count)) + [None] * unknown_count
    for direction, truth in truths.items():
        if truth != expected_truth:
            raise ValueError(f"unexpected class order for {direction}")
        if len(primary[direction]) != len(truth) or any(len(comp[direction]) != len(truth)
                                                      for comp in comparators):
            raise ValueError(f"prediction length differs for {direction}")
    known_indices = list(range(known_count))
    unknown_indices = list(range(known_count, known_count + unknown_count))
    observed_primary = _sampled_macro_ba(primary, truths, known_indices, unknown_indices)
    observed_comparator = (max(_sampled_macro_ba(comp, truths, known_indices, unknown_indices)
                               for comp in comparators) if comparators else None)
    rng = random.Random(seed)
    primary_samples = []
    difference_samples = []
    for _ in range(replicates):
        known = [rng.randrange(known_count) for _ in range(known_count)]
        unknown = [known_count + rng.randrange(unknown_count) for _ in range(unknown_count)]
        score = _sampled_macro_ba(primary, truths, known, unknown)
        primary_samples.append(score)
        if comparators:
            competitor = max(_sampled_macro_ba(comp, truths, known, unknown)
                             for comp in comparators)
            difference_samples.append(score - competitor)
    primary_samples.sort()
    result = {
        "primary_observed_macro_ba": observed_primary,
        "primary_ci_95": [_percentile(primary_samples, 0.025),
                          _percentile(primary_samples, 0.975)],
        "replicates": replicates,
        "seed": seed,
    }
    if comparators:
        difference_samples.sort()
        result["better_comparator_observed_macro_ba"] = observed_comparator
        result["paired_improvement_observed"] = observed_primary - observed_comparator
        result["paired_improvement_ci_95"] = [
            _percentile(difference_samples, 0.025), _percentile(difference_samples, 0.975)]
    return result
