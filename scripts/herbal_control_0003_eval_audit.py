"""Independent replay of the frozen HERBAL-CONTROL-0003 evaluation score.

Uses the NetworkX partial-matching formulation from the independent
development auditor, not the production Hungarian solver or evaluation
scorer. It does not select a method, threshold or distance normalizer.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import floor, isfinite
from pathlib import Path
import random

if __package__:
    from .herbal_control_0003_score_audit import (
        accuracy,
        fused,
        independent,
        next_pattern_cost,
        partial_matching,
    )
else:
    from herbal_control_0003_score_audit import (
        accuracy,
        fused,
        independent,
        next_pattern_cost,
        partial_matching,
    )


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/manifests/herbal_control_0003_panel_amended.json"
SOURCES = ROOT / "data/manifests/herbal_control_0003_sources.json"
DEVELOPMENT_IMAGES = ROOT / "data/manifests/herbal_control_0003_development_images.json"
EVALUATION_IMAGES = ROOT / "data/manifests/herbal_control_0003_evaluation_images.json"
FREEZE = ROOT / "data/manifests/herbal_control_0003_input_freeze.json"
PANEL_SHA256 = "388fe569beb08bb46bb11cad175c96846b57e7b80cd3e7c209fa86079bddc832"
MANUSCRIPTS = ("bnf", "egerton", "casanatense")
METHODS = ("single_first", "single_second", "two_mean", "two_min")
FEATURE_METHOD = "Apple Vision VNGenerateImageFeaturePrintRequest revision 2, scaleFit"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def close(actual: float, expected: float, label: str, tolerance: float = 1e-8) -> None:
    if not isfinite(actual) or abs(actual - expected) > tolerance:
        raise ValueError(f"{label} differs: {actual} versus {expected}")


def _load_inputs(feature_path: Path, selection_path: Path,
                 development_audit_path: Path, score_path: Path) -> tuple[dict, dict, dict, dict]:
    if digest(PANEL) != PANEL_SHA256:
        raise ValueError("fixed panel changed")
    panel = json.loads(PANEL.read_text())
    freeze = json.loads(FREEZE.read_text())
    if (freeze.get("id") != "HERBAL-CONTROL-0003-input-freeze"
            or freeze.get("status") != "complete-pre-score-inputs"
            or freeze.get("panel_manifest_sha256") != PANEL_SHA256
            or freeze.get("source_pages") != 216 or freeze.get("crop_pages") != 216
            or freeze.get("sources_manifest_sha256") != digest(SOURCES)
            or freeze.get("development_images_sha256") != digest(DEVELOPMENT_IMAGES)
            or freeze.get("evaluation_images_sha256") != digest(EVALUATION_IMAGES)):
        raise ValueError("complete input freeze missing or changed")
    selection = json.loads(selection_path.read_text())
    development_audit = json.loads(development_audit_path.read_text())
    if (selection.get("id") != "HERBAL-CONTROL-0003-development-selection"
            or selection.get("status") != "development-score-produced-awaiting-audit-and-push"
            or selection.get("panel_manifest_sha256") != PANEL_SHA256
            or selection.get("full_input_freeze_sha256") != digest(FREEZE)
            or development_audit.get("id") !=
            "HERBAL-CONTROL-0003-independent-development-score-audit"
            or development_audit.get("status") != "pass"
            or development_audit.get("score_sha256") != digest(selection_path)
            or development_audit.get("full_input_freeze_sha256") != digest(FREEZE)
            or development_audit.get("feature_sha256") !=
            selection.get("development_features_sha256")
            or development_audit.get("primary_method") != selection.get("primary_method")):
        raise ValueError("development selection/audit changed")
    images = json.loads(EVALUATION_IMAGES.read_text())
    if (images.get("id") != "HERBAL-CONTROL-0003-evaluation"
            or images.get("status") != "complete-pre-score-evaluation-crops"
            or images.get("panel_manifest_sha256") != PANEL_SHA256
            or images.get("rendered_crops") != 108 or len(images.get("rows", [])) != 108):
        raise ValueError("evaluation crop manifest incomplete")
    for row in images["rows"]:
        if digest(ROOT / row["crop_file"]) != row["crop_sha256"]:
            raise ValueError(f"evaluation crop changed: {row['crop_file']}")
    features = json.loads(feature_path.read_text())
    if (features.get("id") != "HERBAL-CONTROL-0003-evaluation"
            or features.get("feature_method") != FEATURE_METHOD
            or features.get("input_manifest_sha256") != digest(EVALUATION_IMAGES)
            or features.get("full_input_freeze_sha256") != digest(FREEZE)
            or features.get("development_selection_sha256") != digest(selection_path)
            or features.get("development_audit_sha256") != digest(development_audit_path)):
        raise ValueError("evaluation feature provenance changed")
    rows = features.get("row_order", [])
    if len(rows) != 108:
        raise ValueError("evaluation feature row count differs")
    for actual, expected in zip(rows, images["rows"], strict=True):
        if any(actual.get(field) != expected[field] for field in
               ("role", "chapter_class", "manuscript", "crop_file", "crop_sha256")):
            raise ValueError("evaluation feature row order/crop differs")
    matrix = features.get("distance_matrix", [])
    if len(matrix) != 108 or any(len(row) != 108 for row in matrix):
        raise ValueError("evaluation feature matrix not 108 square")
    for i in range(108):
        for j in range(108):
            value = matrix[i][j]
            if not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
                raise ValueError("invalid evaluation feature distance")
            if i == j and abs(value) > 1e-5:
                raise ValueError("nonzero evaluation feature diagonal")
            if j < i and abs(value - matrix[j][i]) > 1e-5:
                raise ValueError("asymmetric evaluation feature matrix")
    score = json.loads(score_path.read_text())
    if (score.get("id") != "HERBAL-CONTROL-0003-evaluation-score"
            or score.get("status") != "evaluation-score-produced-awaiting-independent-audit"
            or score.get("panel_manifest_sha256") != PANEL_SHA256
            or score.get("full_input_freeze_sha256") != digest(FREEZE)
            or score.get("development_selection_sha256") != digest(selection_path)
            or score.get("development_audit_sha256") != digest(development_audit_path)
            or score.get("evaluation_images_sha256") != digest(EVALUATION_IMAGES)
            or score.get("evaluation_features_sha256") != digest(feature_path)):
        raise ValueError("stored evaluation score provenance changed")
    return panel, features, selection, score


def _panels(matrix: list[list[float]], rows: list[dict],
            known: list[str], unknown: list[str],
            scales: dict[str, float]) -> dict[str, tuple]:
    index = {(row["role"], row["chapter_class"], row["manuscript"]): i
             for i, row in enumerate(rows)}
    expected = {(f"evaluation_{kind}", chapter, manuscript)
                for kind, names in (("known", known), ("unknown", unknown))
                for chapter in names for manuscript in MANUSCRIPTS}
    if set(index) != expected or len(index) != 108:
        raise ValueError("evaluation feature rows differ from fixed class split")
    pairs = {f"{query}->{source}" for query in MANUSCRIPTS for source in MANUSCRIPTS
             if query != source}
    if set(scales) != pairs or any(not isfinite(value) or value <= 0
                                   for value in scales.values()):
        raise ValueError("invalid frozen development scales")
    panels = {}
    for query in MANUSCRIPTS:
        sources = [source for source in MANUSCRIPTS if source != query]
        matrices = []
        for source in sources:
            query_indices = [index[(f"evaluation_{kind}", chapter, query)]
                             for kind, names in (("known", known), ("unknown", unknown))
                             for chapter in names]
            gallery = [index[("evaluation_known", chapter, source)] for chapter in known]
            scale = scales[f"{query}->{source}"]
            matrices.append([[matrix[i][j] / scale for j in gallery]
                             for i in query_indices])
        panels[query] = tuple(matrices)
    return panels


def _edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, letter in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(current[j - 1] + 1, previous[j] + 1,
                               previous[j - 1] + int(letter != other)))
        previous = current
    return previous[-1]


def _near_name(known: list[str], unknown: list[str]) -> list[str]:
    results = []
    for raw in unknown:
        name = raw.casefold()
        if any(1 - _edit_distance(name, candidate.casefold()) /
               max(len(name), len(candidate.casefold())) > 0.5
               for candidate in known):
            results.append(raw)
    return results


def _percentile(values: list[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    index = floor(position)
    alpha = position - index
    return (1 - alpha) * values[index] + alpha * values[min(index + 1, len(values) - 1)]


def _class_rates(predictions: dict[str, list[int | None]]) -> tuple[list[float], list[float]]:
    known = [sum(predictions[direction][i] == i for direction in MANUSCRIPTS) / 3
             for i in range(24)]
    unknown = [sum(predictions[direction][24 + i] is None for direction in MANUSCRIPTS) / 3
               for i in range(12)]
    return known, unknown


def bootstrap(primary: dict[str, list[int | None]],
              comparators: list[dict[str, list[int | None]]] | None = None) -> dict:
    """Class-level bootstrap via per-class directional success rates."""
    comparators = comparators or []
    p_known, p_unknown = _class_rates(primary)
    competitors = [_class_rates(predictions) for predictions in comparators]

    def value(rates: tuple[list[float], list[float]],
              known_indices: list[int], unknown_indices: list[int]) -> float:
        known_rates, unknown_rates = rates
        return 0.5 * (sum(known_rates[i] for i in known_indices) / 24
                      + sum(unknown_rates[i] for i in unknown_indices) / 12)

    all_known = list(range(24))
    all_unknown = list(range(12))
    observed = value((p_known, p_unknown), all_known, all_unknown)
    observed_comp = max((value(rates, all_known, all_unknown) for rates in competitors),
                        default=None)
    rng = random.Random(3003)
    primary_values = []
    differences = []
    for _ in range(10_000):
        drawn_known = [rng.randrange(24) for _ in range(24)]
        drawn_unknown = [rng.randrange(12) for _ in range(12)]
        score = value((p_known, p_unknown), drawn_known, drawn_unknown)
        primary_values.append(score)
        if competitors:
            comparison = max(value(rates, drawn_known, drawn_unknown)
                             for rates in competitors)
            differences.append(score - comparison)
    primary_values.sort()
    result = {
        "primary_observed_macro_ba": observed,
        "primary_ci_95": [_percentile(primary_values, 0.025),
                          _percentile(primary_values, 0.975)],
        "replicates": 10_000,
        "seed": 3003,
    }
    if competitors:
        differences.sort()
        result["better_comparator_observed_macro_ba"] = observed_comp
        result["paired_improvement_observed"] = observed - observed_comp
        result["paired_improvement_ci_95"] = [
            _percentile(differences, 0.025), _percentile(differences, 0.975)]
    return result


def _compare_bootstrap(stored: dict, expected: dict, label: str) -> None:
    if set(stored) != set(expected):
        raise ValueError(f"{label} bootstrap keys differ")
    for key, value in expected.items():
        if isinstance(value, list):
            for i, number in enumerate(value):
                close(stored[key][i], number, f"{label} {key}[{i}]")
        elif isinstance(value, float):
            close(stored[key], value, f"{label} {key}")
        elif stored[key] != value:
            raise ValueError(f"{label} {key} differs")


def audit(feature_path: Path, selection_path: Path, development_audit_path: Path,
          score_path: Path) -> dict:
    panel, features, selection, score = _load_inputs(
        feature_path, selection_path, development_audit_path, score_path)
    known = [item["chapter"] for item in panel["roles"]["evaluation_known"]]
    unknown = [item["chapter"] for item in panel["roles"]["evaluation_unknown"]]
    if (len(known) != 24 or len(unknown) != 12 or len(set(known + unknown)) != 36
            or score.get("known_classes") != known or score.get("unknown_classes") != unknown
            or score.get("near_name_unknown_classes") != _near_name(known, unknown)
            or score.get("primary_method") != selection.get("primary_method")
            or set(score.get("methods", {})) != set(METHODS)):
        raise ValueError("evaluation class/method summary differs")
    scales = selection["normalization_medians"]
    if score.get("normalization_medians") != scales:
        raise ValueError("evaluation replaced frozen development normalizers")
    pairs = _panels(features["distance_matrix"], features["row_order"],
                    known, unknown, scales)
    batch_results = {}
    independent_results = {}
    macro = {}
    for method in METHODS:
        selected = selection["methods"][method]
        stored = score["methods"][method]
        batch_tau = selected["batch_threshold"]
        independent_tau = selected["independent_threshold"]
        if (stored["batch_threshold"] != batch_tau
                or stored["independent_threshold"] != independent_tau):
            raise ValueError(f"{method} evaluation threshold differs from development")
        panels = {query: fused(*pairs[query], method) for query in MANUSCRIPTS}
        batch_results[method] = {}
        independent_results[method] = {}
        batch_ba = []
        independent_ba = []
        for query in MANUSCRIPTS:
            costs = panels[query]
            predicted, objective = partial_matching(costs, batch_tau)
            control = independent(costs, independent_tau)
            batch_results[method][query] = predicted
            independent_results[method][query] = control
            known_correct, unknown_rejected, ba = accuracy(predicted)
            c_known, c_unknown, c_ba = accuracy(control)
            stored_batch = stored["batch_by_direction"][query]
            stored_control = stored["independent_by_direction"][query]
            if (stored_batch["predictions"] != predicted
                    or stored_batch["known_correct"] != known_correct
                    or stored_batch["unknown_rejected"] != unknown_rejected
                    or stored_control["predictions"] != control
                    or stored_control["known_correct"] != c_known
                    or stored_control["unknown_rejected"] != c_unknown):
                raise ValueError(f"{method}/{query} prediction or count differs")
            close(stored_batch["balanced_accuracy"], ba, f"{method}/{query} batch BA")
            close(stored_control["balanced_accuracy"], c_ba,
                  f"{method}/{query} independent BA")
            false_accepts = [
                {"query_chapter": unknown[i - 24], "predicted_chapter": known[value]}
                for i, value in enumerate(predicted) if i >= 24 and value is not None
            ]
            if stored_batch["unknown_false_accepts"] != false_accepts:
                raise ValueError(f"{method}/{query} unknown false accepts differ")
            margin = stored_batch["assignment_margin"]
            close(margin["optimum_cost"], objective, f"{method}/{query} optimum cost")
            if margin["optimum_predictions"] != predicted:
                raise ValueError(f"{method}/{query} optimum pattern differs")
            runner_up = next_pattern_cost(costs, batch_tau, predicted)
            close(margin["alternate_cost"], runner_up,
                  f"{method}/{query} alternate cost")
            close(margin["prediction_gap"], max(0.0, runner_up - objective),
                  f"{method}/{query} prediction gap")
            if margin["near_tie_1e_minus_9"] != (runner_up - objective <= 1e-9):
                raise ValueError(f"{method}/{query} near-tie flag differs")
            alternate = margin["alternate_predictions"]
            if (not isinstance(alternate, list) or len(alternate) != 36
                    or alternate == predicted
                    or any(value is not None and
                           (not isinstance(value, int) or not 0 <= value < 24)
                           for value in alternate)
                    or len({value for value in alternate if value is not None})
                    != sum(value is not None for value in alternate)):
                raise ValueError(f"{method}/{query} invalid alternate pattern")
            alternate_objective = sum(batch_tau if value is None else costs[i][value]
                                      for i, value in enumerate(alternate))
            close(alternate_objective, runner_up,
                  f"{method}/{query} stored alternate objective")
            batch_ba.append(ba)
            independent_ba.append(c_ba)
        macro[method] = sum(batch_ba) / 3
        close(stored["batch_macro_ba"], macro[method], f"{method} macro BA")
        close(stored["independent_macro_ba"], sum(independent_ba) / 3,
              f"{method} independent macro BA")
    primary = selection["primary_method"]
    primary_predictions = batch_results[primary]
    primary_ci = bootstrap(primary_predictions)
    for i, value in enumerate(primary_ci["primary_ci_95"]):
        close(score["primary_macro_ba_ci_95"][i], value, f"primary CI[{i}]")
    two_ci = bootstrap(primary_predictions, [batch_results["single_first"],
                                             batch_results["single_second"]]) if primary in (
                                                 "two_mean", "two_min") else None
    one_ci = bootstrap(primary_predictions, [independent_results[primary]])
    if two_ci is None:
        if score.get("two_source_comparison") is not None:
            raise ValueError("non-two-source primary has two-source comparison")
    else:
        _compare_bootstrap(score["two_source_comparison"], two_ci, "two-source")
    _compare_bootstrap(score["one_to_one_comparison"], one_ci, "one-to-one")
    feasibility = (macro[primary] >= 0.80 and all(
        accuracy(batch_results[primary][query])[0] >= 18
        and accuracy(batch_results[primary][query])[1] >= 10
        for query in MANUSCRIPTS))
    two_source = (two_ci is not None and macro[primary] >=
                  max(macro["single_first"], macro["single_second"]) + 0.05
                  and two_ci["paired_improvement_ci_95"][0] > 0)
    independent_macro = sum(accuracy(independent_results[primary][query])[2]
                            for query in MANUSCRIPTS) / 3
    one_to_one = (macro[primary] >= independent_macro + 0.05
                  and one_ci["paired_improvement_ci_95"][0] > 0)
    gates = {
        "historical_image_open_set_feasibility": feasibility,
        "two_source_benefit": two_source,
        "one_to_one_benefit": one_to_one,
    }
    if score.get("gates") != gates:
        raise ValueError("evaluation gate decision differs")
    return {
        "id": "HERBAL-CONTROL-0003-independent-evaluation-audit",
        "status": "pass",
        "evaluation_features_sha256": digest(feature_path),
        "development_selection_sha256": digest(selection_path),
        "development_audit_sha256": digest(development_audit_path),
        "evaluation_score_sha256": digest(score_path),
        "primary_method": primary,
        "method_macro_ba": macro,
        "gates": gates,
        "limits": "Independent arithmetic/provenance replay of chapter-metadata evaluation, not botanical or Voynich truth.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("development_selection", type=Path)
    parser.add_argument("development_audit", type=Path)
    parser.add_argument("evaluation_score", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.features, args.development_selection,
                   args.development_audit, args.evaluation_score)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
