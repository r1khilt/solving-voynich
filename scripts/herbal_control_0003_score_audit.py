"""Independently replay the HERBAL-CONTROL-0003 development score.

This auditor imports neither the production scorer nor its Hungarian solver.
It uses maximum-weight *partial* matching from NetworkX: with 36 queries,
unlimited rejects at cost tau, and one-use known classes, maximizing the
weights tau - class_cost on positive query/class edges is exactly the same
objective as minimizing the registered assignment cost.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import floor, isfinite
from pathlib import Path
from statistics import median

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/manifests/herbal_control_0003_panel_amended.json"
SOURCES = ROOT / "data/manifests/herbal_control_0003_sources.json"
DEVELOPMENT = ROOT / "data/manifests/herbal_control_0003_development_images.json"
EVALUATION = ROOT / "data/manifests/herbal_control_0003_evaluation_images.json"
FREEZE = ROOT / "data/manifests/herbal_control_0003_input_freeze.json"
PANEL_SHA256 = "388fe569beb08bb46bb11cad175c96846b57e7b80cd3e7c209fa86079bddc832"
MANUSCRIPTS = ("bnf", "egerton", "casanatense")
METHODS = ("single_first", "single_second", "two_mean", "two_min")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def close(observed: float, expected: float, field: str, tolerance: float = 1e-8) -> None:
    if not isfinite(observed) or abs(observed - expected) > tolerance:
        raise ValueError(f"{field} differs: {observed} versus {expected}")


def partial_matching(costs: list[list[float]], tau: float, *,
                     force_known_row: int | None = None,
                     forbid_pair: tuple[int, int] | None = None) -> tuple[list[int | None], float]:
    """Independent maximum-weight formulation of reject-capable assignment."""
    if not costs or not costs[0] or not isfinite(tau):
        raise ValueError("empty costs or nonfinite rejection cost")
    classes = len(costs[0])
    if any(len(row) != classes or any(not isfinite(value) for value in row)
           for row in costs):
        raise ValueError("ragged or nonfinite class costs")
    if force_known_row is not None and not 0 <= force_known_row < len(costs):
        raise ValueError("invalid forced query row")
    if forbid_pair is not None and not (0 <= forbid_pair[0] < len(costs)
                                        and 0 <= forbid_pair[1] < classes):
        raise ValueError("invalid forbidden query/class edge")
    max_weight_magnitude = max(abs(tau - value) for row in costs for value in row)
    forced_bonus = (len(costs) + classes + 1) * (max_weight_magnitude + 1) * 2
    graph = nx.Graph()
    graph.add_nodes_from(("q", i) for i in range(len(costs)))
    graph.add_nodes_from(("c", j) for j in range(classes))
    for i, row in enumerate(costs):
        for j, value in enumerate(row):
            if (i, j) == forbid_pair:
                continue
            if value < tau or i == force_known_row:
                bonus = forced_bonus if i == force_known_row else 0
                graph.add_edge(("q", i), ("c", j), weight=tau - value + bonus)
    predictions: list[int | None] = [None] * len(costs)
    for left, right in nx.max_weight_matching(graph, maxcardinality=False, weight="weight"):
        if left[0] == "c":
            left, right = right, left
        predictions[left[1]] = right[1]
    if force_known_row is not None and predictions[force_known_row] is None:
        raise AssertionError("forced known row was rejected")
    if forbid_pair is not None and predictions[forbid_pair[0]] == forbid_pair[1]:
        raise AssertionError("forbidden edge was used")
    objective = sum(tau if predicted is None else costs[i][predicted]
                    for i, predicted in enumerate(predictions))
    return predictions, objective


def next_pattern_cost(costs: list[list[float]], tau: float,
                      optimum: list[int | None]) -> float:
    """Cheapest different named/reject pattern, ignoring null-column permutations."""
    if len(optimum) != len(costs):
        raise ValueError("optimum/query count differs")
    candidates = []
    for row, selected in enumerate(optimum):
        alternative, objective = (
            partial_matching(costs, tau, force_known_row=row) if selected is None
            else partial_matching(costs, tau, forbid_pair=(row, selected)))
        if alternative == optimum:
            raise AssertionError("constrained matching did not change the prediction pattern")
        candidates.append(objective)
    return min(candidates)


def independent(costs: list[list[float]], tau: float) -> list[int | None]:
    predictions = []
    for row in costs:
        lowest = min(row)
        matches = [j for j, value in enumerate(row) if abs(value - lowest) <= 1e-7]
        predictions.append(matches[0] if lowest < tau and len(matches) == 1 else None)
    return predictions


def accuracy(predictions: list[int | None]) -> tuple[int, int, float]:
    if len(predictions) != 36:
        raise ValueError("expected 24 known and 12 unknown queries")
    known = sum(predictions[i] == i for i in range(24))
    rejected = sum(predictions[i] is None for i in range(24, 36))
    return known, rejected, 0.5 * (known / 24 + rejected / 12)


def thresholds(panels: dict[str, list[list[float]]]) -> list[float]:
    values = sorted(value for matrix in panels.values() for row in matrix for value in row)
    if not values or any(not isfinite(value) for value in values):
        raise ValueError("missing or nonfinite class costs")
    candidates = set()
    for k in range(201):
        position = (len(values) - 1) * k / 200
        index = floor(position)
        fraction = position - index
        candidates.add((1 - fraction) * values[index]
                       + fraction * values[min(index + 1, len(values) - 1)])
    candidates.add(0.0 if values[0] > 0 else values[0] - 1)
    candidates.add(values[-1] + 1)
    return sorted(candidates)


def select_threshold(panels: dict[str, list[list[float]]], *, batch: bool) -> tuple[float, float, int]:
    best_tau = None
    best_ba = -1.0
    candidates = thresholds(panels)
    for tau in candidates:
        scores = []
        for manuscript in MANUSCRIPTS:
            predictions = (partial_matching(panels[manuscript], tau)[0] if batch
                           else independent(panels[manuscript], tau))
            scores.append(accuracy(predictions)[2])
        ba = sum(scores) / 3
        if ba > best_ba + 1e-12:
            best_tau, best_ba = tau, ba
    assert best_tau is not None
    return best_tau, best_ba, len(candidates)


def class_panels(distance: list[list[float]], rows: list[dict],
                 known: list[str], unknown: list[str]) -> tuple[dict[str, tuple], dict[str, float]]:
    index = {(row["role"], row["chapter_class"], row["manuscript"]): i
             for i, row in enumerate(rows)}
    expected = {(role, chapter, manuscript)
                for role, names in (("development_known", known),
                                    ("development_unknown", unknown))
                for chapter in names for manuscript in MANUSCRIPTS}
    if set(index) != expected or len(index) != 108:
        raise ValueError("feature rows differ from fixed development classes")
    medians = {}
    panels = {}
    for query in MANUSCRIPTS:
        sources = [source for source in MANUSCRIPTS if source != query]
        source_matrices = []
        for source in sources:
            query_indices = [index[(f"development_{kind}", chapter, query)]
                             for kind, names in (("known", known), ("unknown", unknown))
                             for chapter in names]
            references = [index[("development_known", chapter, source)] for chapter in known]
            scale = median(distance[i][j] for i in query_indices for j in references)
            if not isfinite(scale) or scale <= 0:
                raise ValueError(f"invalid distance median for {query}->{source}")
            medians[f"{query}->{source}"] = scale
            source_matrices.append([[distance[i][j] / scale for j in references]
                                    for i in query_indices])
        panels[query] = tuple(source_matrices)
    return panels, medians


def fused(first: list[list[float]], second: list[list[float]], method: str) -> list[list[float]]:
    if method == "single_first":
        return first
    if method == "single_second":
        return second
    if method == "two_mean":
        return [[(a + b) / 2 for a, b in zip(x, y, strict=True)]
                for x, y in zip(first, second, strict=True)]
    if method == "two_min":
        return [[min(a, b) for a, b in zip(x, y, strict=True)]
                for x, y in zip(first, second, strict=True)]
    raise ValueError(f"unknown method {method}")


def _load_checked_inputs(feature_path: Path, score_path: Path) -> tuple[dict, dict, dict, dict]:
    if digest(PANEL) != PANEL_SHA256:
        raise ValueError("panel changed")
    panel = json.loads(PANEL.read_text())
    freeze = json.loads(FREEZE.read_text())
    if (freeze.get("id") != "HERBAL-CONTROL-0003-input-freeze"
            or freeze.get("status") != "complete-pre-score-inputs"
            or freeze.get("panel_manifest_sha256") != PANEL_SHA256
            or freeze.get("source_pages") != 216 or freeze.get("crop_pages") != 216
            or freeze.get("sources_manifest_sha256") != digest(SOURCES)
            or freeze.get("development_images_sha256") != digest(DEVELOPMENT)
            or freeze.get("evaluation_images_sha256") != digest(EVALUATION)):
        raise ValueError("complete source/crop input freeze absent or changed")
    images = json.loads(DEVELOPMENT.read_text())
    if (images.get("id") != "HERBAL-CONTROL-0003-development"
            or images.get("status") != "complete-pre-score-development-crops"
            or images.get("panel_manifest_sha256") != PANEL_SHA256
            or images.get("rendered_crops") != 108 or len(images.get("rows", [])) != 108):
        raise ValueError("development crop panel is incomplete")
    for row in images["rows"]:
        if digest(ROOT / row["crop_file"]) != row["crop_sha256"]:
            raise ValueError(f"crop changed: {row['crop_file']}")
    features = json.loads(feature_path.read_text())
    if (features.get("id") != "HERBAL-CONTROL-0003-development"
            or features.get("feature_method") !=
            "Apple Vision VNGenerateImageFeaturePrintRequest revision 2, scaleFit"
            or features.get("input_manifest_sha256") != digest(DEVELOPMENT)
            or features.get("full_input_freeze_sha256") != digest(FREEZE)):
        raise ValueError("feature extraction inputs differ from freeze")
    rows = features.get("row_order", [])
    if len(rows) != 108:
        raise ValueError("wrong feature row count")
    for actual, expected in zip(rows, images["rows"], strict=True):
        if any(actual.get(field) != expected[field] for field in
               ("role", "chapter_class", "manuscript", "crop_file", "crop_sha256")):
            raise ValueError("feature row order/crop differs from freeze")
    matrix = features.get("distance_matrix", [])
    if len(matrix) != 108 or any(len(row) != 108 for row in matrix):
        raise ValueError("feature distance matrix not 108 square")
    for i in range(108):
        for j in range(108):
            value = matrix[i][j]
            if not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
                raise ValueError("invalid feature distance")
            if i == j and abs(value) > 1e-5:
                raise ValueError("nonzero feature diagonal")
            if j < i and abs(value - matrix[j][i]) > 1e-5:
                raise ValueError("asymmetric feature matrix")
    score = json.loads(score_path.read_text())
    if (score.get("id") != "HERBAL-CONTROL-0003-development-selection"
            or score.get("status") != "development-score-produced-awaiting-audit-and-push"
            or score.get("panel_manifest_sha256") != PANEL_SHA256
            or score.get("full_input_freeze_sha256") != digest(FREEZE)
            or score.get("development_images_sha256") != digest(DEVELOPMENT)
            or score.get("development_features_sha256") != digest(feature_path)):
        raise ValueError("stored development score provenance differs")
    return panel, features, score, images


def audit(feature_path: Path, score_path: Path) -> dict:
    panel, features, score, _ = _load_checked_inputs(feature_path, score_path)
    known = [row["chapter"] for row in panel["roles"]["development_known"]]
    unknown = [row["chapter"] for row in panel["roles"]["development_unknown"]]
    if (len(known) != 24 or len(unknown) != 12 or len(set(known + unknown)) != 36
            or score.get("known_classes") != known or score.get("unknown_classes") != unknown):
        raise ValueError("development class identity/order differs")
    pair_panels, medians = class_panels(features["distance_matrix"],
                                       features["row_order"], known, unknown)
    if set(score.get("normalization_medians", {})) != set(medians):
        raise ValueError("missing distance normalizers")
    for pair, value in medians.items():
        close(score["normalization_medians"][pair], value, pair)
    if set(score.get("methods", {})) != set(METHODS):
        raise ValueError("stored method set differs")
    method_scores = {}
    candidate_counts = {}
    for method in METHODS:
        panels = {query: fused(*pair_panels[query], method) for query in MANUSCRIPTS}
        tau, macro_ba, candidates = select_threshold(panels, batch=True)
        independent_tau, independent_ba, _ = select_threshold(panels, batch=False)
        stored = score["methods"][method]
        close(stored["batch_threshold"], tau, f"{method} batch threshold")
        close(stored["batch_macro_ba"], macro_ba, f"{method} batch macro BA")
        close(stored["independent_threshold"], independent_tau,
              f"{method} independent threshold")
        close(stored["independent_macro_ba"], independent_ba,
              f"{method} independent macro BA")
        for query in MANUSCRIPTS:
            predicted, objective = partial_matching(panels[query], tau)
            known_ok, unknown_ok, ba = accuracy(predicted)
            recorded = stored["batch_by_direction"][query]
            if (recorded["predictions"] != predicted or recorded["known_correct"] != known_ok
                    or recorded["unknown_rejected"] != unknown_ok):
                raise ValueError(f"{method}/{query} batch predictions differ")
            close(recorded["balanced_accuracy"], ba, f"{method}/{query} batch BA")
            close(recorded["assignment_margin"]["optimum_cost"], objective,
                  f"{method}/{query} matching objective")
            if recorded["assignment_margin"]["optimum_predictions"] != predicted:
                raise ValueError(f"{method}/{query} optimum pattern differs")
            alternate_cost = next_pattern_cost(panels[query], tau, predicted)
            close(recorded["assignment_margin"]["alternate_cost"], alternate_cost,
                  f"{method}/{query} runner-up pattern objective")
            gap = max(0.0, alternate_cost - objective)
            close(recorded["assignment_margin"]["prediction_gap"], gap,
                  f"{method}/{query} runner-up pattern gap")
            alternate = recorded["assignment_margin"]["alternate_predictions"]
            if (not isinstance(alternate, list) or len(alternate) != 36
                    or alternate == predicted
                    or any(value is not None and
                           (not isinstance(value, int) or not 0 <= value < 24)
                           for value in alternate)
                    or len({value for value in alternate if value is not None})
                    != sum(value is not None for value in alternate)):
                raise ValueError(f"{method}/{query} invalid alternate pattern")
            alternate_objective = sum(tau if label is None else panels[query][i][label]
                                      for i, label in enumerate(alternate))
            close(alternate_objective, alternate_cost,
                  f"{method}/{query} stored alternate pattern objective")
            if recorded["assignment_margin"]["near_tie_1e_minus_9"] != (
                    alternate_cost - objective <= 1e-9):
                raise ValueError(f"{method}/{query} near-tie flag differs")
            control = independent(panels[query], independent_tau)
            c_known, c_unknown, c_ba = accuracy(control)
            recorded_control = stored["independent_by_direction"][query]
            if (recorded_control["predictions"] != control
                    or recorded_control["known_correct"] != c_known
                    or recorded_control["unknown_rejected"] != c_unknown):
                raise ValueError(f"{method}/{query} independent predictions differ")
            close(recorded_control["balanced_accuracy"], c_ba,
                  f"{method}/{query} independent BA")
        method_scores[method] = macro_ba
        candidate_counts[method] = candidates
    selected = max(METHODS, key=lambda method: method_scores[method])
    if score.get("primary_method") != selected:
        raise ValueError("primary method differs")
    return {
        "id": "HERBAL-CONTROL-0003-independent-development-score-audit",
        "status": "pass",
        "feature_sha256": digest(feature_path),
        "score_sha256": digest(score_path),
        "full_input_freeze_sha256": digest(FREEZE),
        "matcher": "NetworkX maximum-weight partial matching on positive tau-minus-cost edges",
        "primary_method": selected,
        "batch_macro_ba": method_scores,
        "threshold_candidate_counts": candidate_counts,
        "limits": "Development-only scoring and alternate-pattern audit; evaluation remains sealed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("score", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.features, args.score)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "primary_method": result["primary_method"]}))


if __name__ == "__main__":
    main()
