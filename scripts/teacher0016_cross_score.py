"""No-model TEACH-0016 group-level scoring on an already audited archive."""

import hashlib
import json

import numpy as np

from scripts.teacher0016_suite_audit import SURFACES, _episode


def _interval(per_group: list[tuple[int, int]], identity: list[str]) -> list[float]:
    counts = np.asarray(per_group, dtype=np.int64)
    if counts.shape[1:] != (2,) or counts[:, 1].sum() <= 0:
        raise ValueError("Cross-order bootstrap counts invalid")
    token = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(counts), size=(4000, len(counts)))
    totals = counts[draws].sum(axis=1)
    ratios = np.divide(totals[:, 0], totals[:, 1],
                       out=np.zeros(4000), where=totals[:, 1] > 0)
    valid = ratios[totals[:, 1] > 0]
    return np.quantile(valid, [.025, .975]).tolist() if len(valid) else [0.0, 0.0]


def score_split(manifest: dict, surfaces: list[list[dict]],
                identity: list[str]) -> dict:
    """Score all groups; no thresholds are fitted from the observed rows."""
    groups = manifest["groups"]
    if len(surfaces) != len(groups) * len(SURFACES):
        raise ValueError("Cross-order surface coverage differs")
    names = (
        "transfer", "triples", "same_key", "non_injection",
        "reverse_eligible", "reverse_full", "rich", "free",
        "source_order_0", "source_order_1", "wrong_key", "distinct",
        "random", "pointer", "base_clean", "target_clean",
        "full_transfer", "full_same_key",
    )
    per_group = {name: [] for name in names}
    matrix = {(a, b): [0, 0] for a in range(3) for b in range(3)}
    full_matrix = {(a, b): [0, 0] for a in range(3) for b in range(3)}
    order_matrix = {(order, a, b): [0, 0]
                    for order in range(2) for a in range(3) for b in range(3)}
    full_order_matrix = {(order, a, b): [0, 0]
                         for order in range(2) for a in range(3)
                         for b in range(3)}
    shifted_pairs = 0
    shifted_offdiag = 0
    for group_index, group in enumerate(groups):
        counts = {name: [0, 0] for name in names}

        def add(name: str, condition: bool) -> None:
            counts[name][0] += int(condition)
            counts[name][1] += 1

        for surface_index, (d, marked, source_order) in enumerate(SURFACES):
            rows = surfaces[group_index * 8 + surface_index]
            if len(rows) != 9:
                raise ValueError("Cross-order nine-pair surface incomplete")
            shifted = rows[0]["shifted_slot"]
            if any(row["shifted_slot"] != shifted for row in rows):
                raise ValueError("Cross-order shifted flag differs within surface")
            shifted_pairs += int(shifted)
            if shifted:
                for a in range(3):
                    add("triples", all(rows[a * 3 + b][
                        "transfer_prediction"] == rows[a * 3 + b][
                        "target_answer"] for b in range(3)))
            for row in rows:
                a, b = row["source_g"], row["recipient_g"]
                target = row["target_answer"]
                transfer_hit = row["transfer_prediction"] == target
                add("full_transfer", transfer_hit)
                add("full_same_key", row["same_key_prediction"] == target)
                full_matrix[a, b][0] += int(transfer_hit)
                full_matrix[a, b][1] += 1
                full_order_matrix[source_order, a, b][0] += int(transfer_hit)
                full_order_matrix[source_order, a, b][1] += 1
                if not shifted or a == b:
                    continue
                shifted_offdiag += 1
                add("transfer", transfer_hit)
                add("same_key", row["same_key_prediction"] == target)
                add("non_injection", row["transfer_prediction"] !=
                    row["source_answer"])
                add("rich" if marked else "free", transfer_hit)
                add(f"source_order_{source_order}", transfer_hit)
                add("wrong_key", row["wrong_key_prediction"] == target)
                add("distinct", row["distinct_prediction"] == target)
                add("random", row["random_prediction"] == target)
                source_slot = row["source_slot"]
                base = _episode(group, 0, b, d, marked, 1 - source_order)
                add("pointer", base["serialized_rows"][source_slot][1] == target)
                add("base_clean", row["base_prediction"] == row["base_answer"])
                add("target_clean", row["target_prediction"] == target)
                eligible = (row["base_prediction"] == row["base_answer"] and
                            row["target_prediction"] == target)
                if eligible:
                    add("reverse_eligible", row["reverse_prediction"] ==
                        row["base_answer"])
                add("reverse_full", row["reverse_prediction"] ==
                    row["base_answer"])
                matrix[a, b][0] += int(transfer_hit)
                matrix[a, b][1] += 1
                order_matrix[source_order, a, b][0] += int(transfer_hit)
                order_matrix[source_order, a, b][1] += 1
        for name in names:
            per_group[name].append(tuple(counts[name]))
    scores = {}
    for name, grouped in per_group.items():
        hit = sum(count for count, _ in grouped)
        total = sum(count for _, count in grouped)
        scores[name] = {"correct": hit, "total": total,
                        "accuracy": hit / total if total else 0.0,
                        "group_bootstrap_95": (_interval(grouped, identity + [name])
                                               if total else None)}
    if (len(groups) == 128 and
            (scores["full_transfer"]["total"] != 9216 or
             scores["triples"]["total"] != shifted_pairs * 3 or
             scores["transfer"]["total"] != shifted_offdiag or
             shifted_offdiag != shifted_pairs * 6)):
        raise ValueError("Cross-order fixed denominator differs")
    transfer = scores["transfer"]["accuracy"]
    control_margins = {name: transfer - scores[name]["accuracy"]
                       for name in ("wrong_key", "distinct", "random", "pointer")}
    positive = scores["same_key"]["accuracy"] >= .90
    qualified = (
        positive and transfer >= .70 and scores["triples"]["accuracy"] >= .55
        and scores["non_injection"]["accuracy"] >= .90
        and scores["reverse_eligible"]["total"] > 0
        and scores["reverse_eligible"]["accuracy"] >= .65
        and scores["rich"]["accuracy"] >= .65
        and scores["free"]["accuracy"] >= .65
        and scores["source_order_0"]["accuracy"] >= .65
        and scores["source_order_1"]["accuracy"] >= .65
        and all(margin >= .35 for margin in control_margins.values())
    )

    def compact(pair: list[int]) -> dict:
        return {"correct": pair[0], "total": pair[1],
                "accuracy": pair[0] / pair[1] if pair[1] else 0.0}

    return {
        "groups": len(groups), "shifted_surface_pairs": shifted_pairs,
        "shifted_offdiag_attempts": shifted_offdiag,
        "scores": scores, "control_margins": control_margins,
        "per_source_recipient": {f"{a},{b}": compact(matrix[a, b])
                                 for a in range(3) for b in range(3)},
        "full_per_source_recipient": {
            f"{a},{b}": compact(full_matrix[a, b])
            for a in range(3) for b in range(3)},
        "per_order_source_recipient": {
            f"{order},{a},{b}": compact(order_matrix[order, a, b])
            for order in range(2) for a in range(3) for b in range(3)},
        "full_per_order_source_recipient": {
            f"{order},{a},{b}": compact(full_order_matrix[order, a, b])
            for order in range(2) for a in range(3) for b in range(3)},
        "same_key_positive_control_pass": positive,
        "candidate_order_robust_state_pending_replay": qualified,
        "interpretation": ("candidate_pending_replay" if qualified else
                           "inconclusive_cross_order_interface" if not positive
                           else "registered_conjunction_not_met"),
    }
