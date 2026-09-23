"""Frozen metric and selection logic for TEACH-0013 causal discovery."""

from collections import defaultdict
from dataclasses import asdict, dataclass
import math


DISCOVERY_THRESHOLDS = {
    "item_accuracy": .75,
    "group_accuracy": .60,
    "non_injection": .90,
    "mean_probability_gain": .35,
}
DISCOVERY_STRATA = ("all", "marked", "marker_free", "first_half", "second_half")


@dataclass(frozen=True)
class EffectMetrics:
    items: int
    correct: int
    item_accuracy: float
    groups: int
    exact_groups: int
    group_accuracy: float
    changed_recipient_items: int
    non_injected: int
    non_injection: float
    mean_probability_gain: float


def _mean(values):
    return sum(values) / len(values) if values else math.nan


def effect_metrics(rows: list[dict]) -> EffectMetrics:
    """Score recipient-specific transfer rows; each group must contain recipients 0,1,2."""
    if not rows:
        raise ValueError("Cannot score an empty intervention cell")
    keys = {(row["group_id"], row["recipient"]) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("Duplicate group/recipient row")
    grouped = defaultdict(list)
    for row in rows:
        if row["recipient"] not in (0, 1, 2):
            raise ValueError("Recipient index outside 0,1,2")
        grouped[row["group_id"]].append(row)
    if any({row["recipient"] for row in group} != {0, 1, 2}
           for group in grouped.values()):
        raise ValueError("Every scored group must contain exactly three recipients")
    correct = sum(row["prediction"] == row["target"] for row in rows)
    exact = sum(all(row["prediction"] == row["target"] for row in group)
                for group in grouped.values())
    changed = [row for row in rows if row["recipient"] != 0]
    non_injected = sum(row["prediction"] != row["fixed_donor_answer"] for row in changed)
    return EffectMetrics(
        len(rows), correct, correct / len(rows), len(grouped), exact, exact / len(grouped),
        len(changed), non_injected, non_injected / len(changed),
        _mean([row["target_probability"] - row["base_target_probability"] for row in rows]))


def _stratum_rows(rows: list[dict], stratum: str) -> list[dict]:
    if stratum == "all":
        return rows
    if stratum in ("marked", "marker_free"):
        return [row for row in rows if row["render_stratum"] == stratum]
    if stratum in ("first_half", "second_half"):
        return [row for row in rows if row["position_stratum"] == stratum]
    raise ValueError(f"Unknown discovery stratum: {stratum}")


def candidate_metrics(rows: list[dict]) -> dict:
    """Score all registered strata for one seed/site candidate."""
    result = {}
    for stratum in DISCOVERY_STRATA:
        selected = _stratum_rows(rows, stratum)
        result[stratum] = asdict(effect_metrics(selected)) if selected else None
    return result


def metrics_qualify(metrics: dict, *, expected_groups_per_render: int) -> bool:
    if set(metrics) != set(DISCOVERY_STRATA):
        raise ValueError("Discovery strata are incomplete")
    complete = (metrics["all"] is not None
                and metrics["all"]["groups"] == 2 * expected_groups_per_render
                and metrics["marked"] is not None
                and metrics["marked"]["groups"] == expected_groups_per_render
                and metrics["marker_free"] is not None
                and metrics["marker_free"]["groups"] == expected_groups_per_render)
    return complete and all(
        row is not None
        and row["item_accuracy"] >= DISCOVERY_THRESHOLDS["item_accuracy"]
        and row["group_accuracy"] >= DISCOVERY_THRESHOLDS["group_accuracy"]
        and row["non_injection"] >= DISCOVERY_THRESHOLDS["non_injection"]
        and row["mean_probability_gain"] >= DISCOVERY_THRESHOLDS["mean_probability_gain"]
        for row in metrics.values())


def select_residual_site(rows: list[dict], semantic_label_order: tuple[str, ...], *,
                         expected_groups_per_render: int = 128) -> dict:
    """Apply the frozen two-seed earliest-cut/minimum-effect selection rule."""
    if not rows:
        raise ValueError("No discovery rows")
    label_index = {name: index for index, name in enumerate(semantic_label_order)}
    if len(label_index) != len(semantic_label_order):
        raise ValueError("Semantic label order contains duplicates")
    cells = defaultdict(list)
    for row in rows:
        if row["semantic_label"] not in label_index:
            raise ValueError(f"Unregistered semantic label: {row['semantic_label']}")
        cells[(row["replicate"], row["cut_index"], row["semantic_label"])].append(row)
    candidates = sorted({(cut, label) for _, cut, label in cells})
    table = {}
    qualified = []
    for cut, label in candidates:
        by_seed = {}
        for replicate in (0, 1):
            key = (replicate, cut, label)
            if key not in cells:
                raise ValueError(f"Missing seed cell: {key}")
            by_seed[str(replicate)] = candidate_metrics(cells[key])
        is_qualified = all(metrics_qualify(
            seed, expected_groups_per_render=expected_groups_per_render)
            for seed in by_seed.values())
        existing = [row for seed in by_seed.values() for row in seed.values()
                    if row is not None]
        minimum_group = min(row["group_accuracy"] for row in existing)
        minimum_item = min(row["item_accuracy"] for row in existing)
        key = f"{cut}:{label}"
        table[key] = {"cut_index": cut, "semantic_label": label,
                      "qualified": is_qualified, "by_seed": by_seed,
                      "minimum_group_accuracy": minimum_group,
                      "minimum_item_accuracy": minimum_item}
        if is_qualified:
            qualified.append(table[key])
    if not qualified:
        selection = None
    else:
        earliest = min(row["cut_index"] for row in qualified)
        selection = sorted(
            (row for row in qualified if row["cut_index"] == earliest),
            key=lambda row: (-row["minimum_group_accuracy"],
                             -row["minimum_item_accuracy"],
                             label_index[row["semantic_label"]]))[0]
        selection = {key: value for key, value in selection.items() if key != "by_seed"}
    return {"thresholds": DISCOVERY_THRESHOLDS, "strata": DISCOVERY_STRATA,
            "expected_groups_per_render": expected_groups_per_render,
            "semantic_label_order": semantic_label_order,
            "candidates": table, "selection": selection}


def rank_secondary_residual_sites(rows: list[dict], semantic_label_order: tuple[str, ...], *,
                                  expected_groups: int) -> dict:
    """Rank a registered secondary factor without borrowing the primary F thresholds."""
    if not rows or expected_groups <= 0:
        raise ValueError("Secondary residual ranking requires rows and positive group count")
    label_index = {name: index for index, name in enumerate(semantic_label_order)}
    cells = defaultdict(list)
    families = {row.get("family") for row in rows}
    if len(families) != 1 or None in families:
        raise ValueError("Rank exactly one named secondary family at a time")
    for row in rows:
        if row["semantic_label"] not in label_index:
            raise ValueError(f"Unregistered semantic label: {row['semantic_label']}")
        cells[(row["replicate"], row["cut_index"], row["semantic_label"])].append(row)
    candidates = sorted({(cut, label) for _, cut, label in cells})
    table = {}
    for cut, label in candidates:
        by_seed, metrics, complete = {}, [], True
        for replicate in (0, 1):
            key = (replicate, cut, label)
            if key not in cells:
                raise ValueError(f"Missing secondary seed cell: {key}")
            score = asdict(effect_metrics(cells[key]))
            complete = complete and score["groups"] == expected_groups
            by_seed[str(replicate)] = score
            metrics.append(score)
        entry = {
            "cut_index": cut, "semantic_label": label, "complete": complete,
            "by_seed": by_seed,
            "minimum_group_accuracy": min(row["group_accuracy"] for row in metrics),
            "minimum_item_accuracy": min(row["item_accuracy"] for row in metrics),
            "minimum_mean_probability_gain": min(
                row["mean_probability_gain"] for row in metrics),
        }
        table[f"{cut}:{label}"] = entry
    eligible = [row for row in table.values() if row["complete"]]
    if eligible:
        selected = sorted(
            eligible,
            key=lambda row: (-row["minimum_group_accuracy"],
                             -row["minimum_item_accuracy"],
                             -row["minimum_mean_probability_gain"],
                             row["cut_index"], label_index[row["semantic_label"]]))[0]
        selection = {key: value for key, value in selected.items() if key != "by_seed"}
    else:
        selection = None
    return {"family": next(iter(families)), "expected_groups": expected_groups,
            "semantic_label_order": semantic_label_order,
            "ranking_rule": ("max_min_group_then_item_then_probability_gain_then_earlier_cut_"
                             "then_semantic_label"),
            "candidates": table, "selection": selection}


def fresh_panel_decision(scores: dict) -> dict:
    """Conjunctive Stage-A confirmation competence gate for both raw-deep seeds."""
    required = {
        "composed_items": .90,
        "base_recipient_groups": .80,
        "donor_recipient_groups": .80,
        "marker_pairs": .85,
        "order_pairs": .85,
        "distractor_pairs": .85,
        "first_hop": .95,
        "direct": .95,
        "copy": .98,
    }
    clauses = {}
    for replicate in ("0", "1"):
        if replicate not in scores or set(scores[replicate]) != set(required):
            raise ValueError(f"Incomplete fresh-panel scores for seed {replicate}")
        clauses[replicate] = {f"{name}_at_least_{threshold:.2f}":
                              scores[replicate][name] >= threshold
                              for name, threshold in required.items()}
    passed = all(all(seed.values()) for seed in clauses.values())
    return {"label": "pass" if passed else "inconclusive_fresh_panel_competence",
            "thresholds": required, "clauses_by_seed": clauses}


def recipient_transfer_decision(scores: dict, *, discovery_selected: bool,
                                numerical_qualified: bool) -> dict:
    """Conjunctive Stage-C recipient-key transfer verdict."""
    floors = {
        "clean_base": .95, "clean_donor": .95, "input_f_replacement": .95,
        "sufficiency_items_forward": .85, "sufficiency_groups_forward": .70,
        "sufficiency_items_reverse": .85, "sufficiency_groups_reverse": .70,
        "changed_non_injection_forward": .95, "changed_non_injection_reverse": .95,
        "necessity_items": .80, "necessity_groups": .65, "rescue": .95,
        "same_key_format": .95, "same_key_order": .95, "same_key_distractor": .95,
        "minimum_control_advantage": .40, "mean_probability_gain": .40,
        "final_answer_injection": .95,
    }
    ceilings = {"direct_loss": .05, "copy_loss": .05}
    clauses = {}
    for replicate in ("0", "1"):
        if replicate not in scores:
            raise ValueError(f"Missing recipient-transfer seed {replicate}")
        row = scores[replicate]
        missing = (set(floors) | set(ceilings)) - set(row)
        if missing:
            raise ValueError(f"Missing recipient-transfer scores: {sorted(missing)}")
        clauses[replicate] = {
            **{f"{name}_at_least_{threshold:.2f}": row[name] >= threshold
               for name, threshold in floors.items()},
            **{f"{name}_at_most_{threshold:.2f}": row[name] <= threshold
               for name, threshold in ceilings.items()},
        }
    supported = discovery_selected and numerical_qualified and all(
        all(seed.values()) for seed in clauses.values())
    return {"label": "recipient_key_transfer_supported" if supported else "not_supported",
            "discovery_selected": discovery_selected,
            "numerical_qualified": numerical_qualified,
            "floor_thresholds": floors, "ceiling_thresholds": ceilings,
            "clauses_by_seed": clauses}
