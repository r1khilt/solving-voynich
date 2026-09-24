#!/usr/bin/env python3
"""Independent no-model audit of TEACH-0013 Stage-D confirmation.

The auditor imports neither a model nor any generator, intervention, campaign, or
experiment scoring module.  It treats the compact rows, tensor payloads, and report
as untrusted and reconstructs their registered relationships from the frozen suite.
An otherwise well-formed legacy confirmation remains ``fail_closed`` until the
finite cross-seed transport, physical-order oracle, retained equal-energy states,
complete decisions, and independent decision evidence are present.
"""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess

import torch


ROOT = Path(__file__).resolve().parents[1]
SYMBOL_START = 16
CONTROL_COUNT = 32
CONTROL_SEEDS = {"content": 73411, "binding": 73421, "order": 73431}
CONTROL_SURFACES = {
    "content": ("marked",),
    "binding": ("marked",),
    "order": ("f0_marked", "f1_marked"),
}
FACTOR_PAIRS = {
    "content": (
        ("marked", "base", "donor"),
        ("marker_free", "marker_free_base", "marker_free_donor"),
        ("reordered", "reordered_base", "reordered_donor"),
        ("format", "format_base", "format_donor"),
        ("distractor", "distractor_base", "distractor_donor"),
    ),
    "binding": (
        ("marked", "binding_base", "binding_donor"),
        ("marker_free", "binding_marker_free_base", "binding_marker_free_donor"),
        ("reordered", "binding_reordered_base", "binding_reordered_donor"),
        ("format", "binding_format_base", "binding_format_donor"),
        ("distractor", "binding_distractor_base", "binding_distractor_donor"),
    ),
    "order": (
        ("f0_marked", "base", "reordered_base"),
        ("f1_marked", "donor", "reordered_donor"),
        ("f0_marker_free", "marker_free_base", "marker_free_reordered_base"),
        ("f1_marker_free", "marker_free_donor", "marker_free_reordered_donor"),
        ("f0_format", "format_base", "format_reordered_base"),
        ("f1_format", "format_donor", "format_reordered_donor"),
        ("f0_distractor", "distractor_base", "distractor_reordered_base"),
        ("f1_distractor", "distractor_donor", "distractor_reordered_donor"),
    ),
}
MAX_STAGE_C_E_SECONDS = 5400.0
MAX_CAMPAIGN_SECONDS = 14_400.0
MAX_CURRENT_BYTES = 24 * 1024**3
MAX_TRAFFIC_BYTES = 300 * 1024**3
MAX_OUTPUT_BYTES = 20 * 1024**3
MAX_RESULT_BYTES = 2 * 1024**3


class AuditError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise AuditError(message)


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) \
        if path.exists() else 0


def artifact_path(metadata: dict, description: str) -> Path:
    need(set(metadata) >= {"path", "sha256", "bytes"},
         f"Malformed {description} metadata")
    path = (ROOT / metadata["path"]).resolve()
    need(path.is_relative_to(ROOT.resolve()) and path.is_file()
         and path.stat().st_size == metadata["bytes"]
         and file_digest(path) == metadata["sha256"], f"{description} mismatch")
    return path


def load_rows(metadata: dict) -> list[dict]:
    path = artifact_path(metadata, "compact confirmation rows")
    with gzip.open(path, "rt") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    need(len(rows) == metadata["rows"], "Compact confirmation row count mismatch")
    return rows


def validate_row(row: dict, group_ids: set[str]) -> None:
    required = {
        "condition", "direction", "logical_group_id", "group_id", "recipient",
        "target", "prediction", "base_answer", "clean_prediction",
        "fixed_donor_answer", "candidate_member", "wrong_destination",
        "target_probability", "base_target_probability", "target_logit",
        "base_target_logit", "prediction_logit", "item_id",
    }
    need(required <= set(row), "Compact confirmation row fields are incomplete")
    need(isinstance(row["item_id"], str) and row["item_id"], "Missing item identity")
    need(row["logical_group_id"] in group_ids and row["recipient"] in (0, 1, 2),
         "Unknown confirmation group or recipient")
    need(row["group_id"] == f"{row['logical_group_id']}:{row['direction']}",
         "Grouped confirmation identifier changed")
    need(row["wrong_destination"] in {
        "target", "base_answer", "fixed_donor_answer", "other_recipient_target",
        "other_legal_candidate", "outside_legal_candidates",
    }, "Unknown wrong-destination label")
    for name in ("target_probability", "base_target_probability"):
        need(type(row[name]) in (int, float) and math.isfinite(row[name])
             and 0 <= row[name] <= 1, f"Malformed probability: {name}")
    for name in ("target_logit", "base_target_logit", "prediction_logit"):
        need(type(row[name]) in (int, float) and math.isfinite(row[name]),
             f"Malformed logit: {name}")


def _row_lookup(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        need(row["item_id"] not in result, "Duplicate compact item ID")
        result[row["item_id"]] = row
    return result


def verify_sharded_logits(metadata: dict, rows: list[dict]) -> dict:
    """Verify every shard, exact float32 value, and clean-logit reuse."""
    need(metadata.get("format") == "TEACH-0013-sharded-symbol-logits-v1"
         and isinstance(metadata.get("shards"), list)
         and metadata.get("records") == len(metadata["shards"]),
         "Unknown sharded exact-logit format")
    compact = _row_lookup(rows)
    seen, clean_by_render, native_errors = set(), {}, []
    total_rows = 0
    for index, shard in enumerate(metadata["shards"]):
        path = artifact_path(shard, f"exact-logit shard {index}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        need(payload.get("format") == "TEACH-0013-symbol-logit-shard-v1"
             and set(payload) == {"format", "record"}, "Unknown exact-logit shard")
        record = payload["record"]
        required = {"condition", "direction", "logical_group_ids", "recipients",
                    "targets", "item_ids", "edited_symbol_logits",
                    "clean_symbol_logits"}
        need(set(record) == required, "Exact-logit record fields changed")
        edited, clean = record["edited_symbol_logits"], record["clean_symbol_logits"]
        item_ids = tuple(record["item_ids"])
        count = len(item_ids)
        need(shard.get("rows") == count and isinstance(edited, torch.Tensor)
             and isinstance(clean, torch.Tensor) and edited.shape == clean.shape
             and edited.ndim == 2 and edited.shape[0] == count
             and edited.dtype == clean.dtype == torch.float32
             and torch.isfinite(edited).all().item() and torch.isfinite(clean).all().item(),
             "Malformed exact float32 logits")
        need(len(set(item_ids)) == count and not seen.intersection(item_ids),
             "Exact-logit item IDs are duplicated")
        need(all(item_id in compact for item_id in item_ids),
             "Exact-logit item is absent from compact rows")
        selected = [compact[item_id] for item_id in item_ids]
        need(tuple(row["condition"] for row in selected) == (record["condition"],) * count
             and tuple(row["direction"] for row in selected) == (record["direction"],) * count
             and tuple(row["logical_group_id"] for row in selected)
             == tuple(record["logical_group_ids"])
             and tuple(row["recipient"] for row in selected) == tuple(record["recipients"])
             and tuple(row["target"] for row in selected) == tuple(record["targets"]),
             "Exact-logit metadata does not bind to compact rows")
        probability, clean_probability = edited.softmax(-1), clean.softmax(-1)
        predictions = edited.argmax(-1) + SYMBOL_START
        clean_predictions = clean.argmax(-1) + SYMBOL_START
        for row_index, row in enumerate(selected):
            need(SYMBOL_START <= row["target"] < SYMBOL_START + edited.shape[1]
                 and SYMBOL_START <= row["prediction"] < SYMBOL_START + edited.shape[1]
                 and SYMBOL_START <= row["clean_prediction"] < SYMBOL_START + edited.shape[1],
                 "Compact symbol lies outside the stored vocabulary")
            target_index = row["target"] - SYMBOL_START
            prediction_index = row["prediction"] - SYMBOL_START
            need(int(predictions[row_index]) == row["prediction"]
                 and int(clean_predictions[row_index]) == row["clean_prediction"],
                 "Compact prediction disagrees with exact logits")
            need(math.isclose(float(probability[row_index, target_index]),
                              row["target_probability"], abs_tol=1e-6)
                 and math.isclose(float(clean_probability[row_index, target_index]),
                                  row["base_target_probability"], abs_tol=1e-6)
                 and math.isclose(float(edited[row_index, target_index]),
                                  row["target_logit"], abs_tol=1e-7)
                 and math.isclose(float(clean[row_index, target_index]),
                                  row["base_target_logit"], abs_tol=1e-7)
                 and math.isclose(float(edited[row_index, prediction_index]),
                                  row["prediction_logit"], abs_tol=1e-7),
                 "Compact probabilities or logits disagree with exact tensors")
            render_id = row.get("base_render_id")
            if render_id is not None:
                previous = clean_by_render.setdefault(render_id, clean[row_index].clone())
                need(torch.equal(previous, clean[row_index]),
                     "Clean logits changed for an identical frozen base rendering")
            if row["condition"].startswith("content_native_restoration"):
                native_errors.append(float((edited[row_index] - clean[row_index]).abs().max()))
        seen.update(item_ids)
        total_rows += count
    need(seen == set(compact) and total_rows == metadata.get("rows") == len(rows),
         "Exact-logit shards do not exhaust the compact rows")
    return {"records": len(metadata["shards"]), "rows": total_rows,
            "max_native_restoration_logit_error": (
                None if not native_errors else max(native_errors))}


def _episode_id(row: dict) -> str:
    return row["render_id"]


def expected_transfer_items(groups: list[dict], factor: str, prefix: str,
                            surfaces=None) -> dict[str, dict]:
    expected = {}
    for surface, left_name, right_name in FACTOR_PAIRS[factor]:
        if surfaces is not None and surface not in surfaces:
            continue
        for direction, base_name, donor_name in (
                ("forward", left_name, right_name), ("reverse", right_name, left_name)):
            condition = f"{prefix}_{surface}"
            for group in groups:
                bases, donors = group[base_name], group[donor_name]
                for recipient, base in enumerate(bases):
                    donor = donors[0] if factor in ("content", "binding") else donors[recipient]
                    item_id = f"{factor}:{condition}:{direction}:{group['group_id']}:{recipient}"
                    expected[item_id] = {
                        "condition": condition, "direction": direction,
                        "logical_group_id": group["group_id"], "recipient": recipient,
                        "target": donors[recipient]["answer"],
                        "fixed_donor_answer": donors[0]["answer"],
                        "base_render_id": _episode_id(base),
                        "donor_render_id": _episode_id(donor),
                    }
    return expected


def expected_core_grid(groups: list[dict], selections: dict) -> dict[str, dict]:
    expected = {}
    selected = [factor for factor in ("content", "binding", "order")
                if selections[factor]["selection"] is not None]
    for factor in selected:
        for prefix in (f"{factor}_selected", f"{factor}_complement"):
            expected.update(expected_transfer_items(groups, factor, prefix))
        for family in ("haar", "deranged", "equal_energy"):
            for index in range(CONTROL_COUNT):
                expected.update(expected_transfer_items(
                    groups, factor, f"{factor}_{family}_{index:02d}",
                    CONTROL_SURFACES[factor]))
    if "content" in selected and "binding" in selected:
        for factor in ("content", "binding"):
            expected.update(expected_transfer_items(
                groups, factor, f"content_binding_on_{factor}"))
    if "binding" in selected and "order" in selected:
        expected.update(expected_transfer_items(
            groups, "binding", "binding_order_control", ("marked",)))
    if set(selected) == {"content", "binding", "order"}:
        for factor in selected:
            expected.update(expected_transfer_items(groups, factor, f"all_factors_on_{factor}"))
    if "content" in selected:
        for group in groups:
            for assignment, name in (("f0", "base"), ("f1", "donor")):
                for recipient, episode in enumerate(group[name]):
                    for condition in ("content_mean_ablation",
                                      "content_equal_norm_corruption",
                                      "content_native_restoration"):
                        item_id = (f"content:necessity:{assignment}:{group['group_id']}:"
                                   f"{recipient}:{condition}")
                        expected[item_id] = {
                            "condition": f"{condition}_{assignment}",
                            "direction": "necessity",
                            "logical_group_id": group["group_id"], "recipient": recipient,
                            "target": episode["answer"],
                            "base_render_id": episode["render_id"],
                            "donor_render_id": episode["render_id"],
                        }
            for task, base_name, donor_name in (
                    ("first_hop", "first_hop_base", "first_hop_donor"),
                    ("direct", "direct_base", "direct_donor")):
                base, donor = group[base_name], group[donor_name]
                item_id = f"content:{task}:{group['group_id']}:0"
                expected[item_id] = {
                    "condition": f"content_{task}", "direction": "specificity",
                    "logical_group_id": group["group_id"], "recipient": 0,
                    "target": donor["answer"], "base_render_id": base["render_id"],
                    "donor_render_id": donor["render_id"],
                }
            base, donor = group["copy_control"], group["donor"][0]
            expected[f"content:copy:{group['group_id']}:0"] = {
                "condition": "content_copy", "direction": "specificity",
                "logical_group_id": group["group_id"], "recipient": 0,
                "target": base["answer"], "base_render_id": base["render_id"],
                "donor_render_id": donor["render_id"],
            }
    if "binding" in selected:
        for direction, base_assignment, donor_assignment, reference_name, donor_name in (
                ("forward", "base", "donor", "binding_base", "binding_donor"),
                ("reverse", "donor", "base", "binding_donor", "binding_base")):
            for task in ("first_hop", "direct"):
                base_name = f"{task}_{base_assignment}"
                target_name = f"{task}_{donor_assignment}" if task == "first_hop" else base_name
                for group in groups:
                    base, target, donor = (group[base_name], group[target_name],
                                           group[donor_name][0])
                    item_id = f"binding:{task}:{direction}:{group['group_id']}:0"
                    expected[item_id] = {
                        "condition": f"binding_{task}", "direction": direction,
                        "logical_group_id": group["group_id"], "recipient": 0,
                        "target": target["answer"], "base_render_id": base["render_id"],
                        "donor_render_id": donor["render_id"],
                    }
    return expected


def expected_cross_seed_items(groups: list[dict], target_seed: str) -> dict[str, dict]:
    source_seed = "1" if target_seed == "0" else "0"
    expected = {}
    for map_name in ("shared", "deranged"):
        prefix = "content_cross_seed"
        for item_id, fields in expected_transfer_items(
                groups, "content", prefix, ("marked",)).items():
            factor_direction = fields["direction"]
            rewritten_id = (f"content:cross_seed:{map_name}:{source_seed}_to_{target_seed}:"
                            f"{factor_direction}:{fields['logical_group_id']}:"
                            f"{fields['recipient']}")
            expected[rewritten_id] = {
                **fields,
                "condition": f"content_cross_seed_{map_name}_marked",
                "direction": f"{source_seed}_to_{target_seed}_{factor_direction}",
            }
    return expected


def verify_core_grid(rows: list[dict], groups: list[dict], selections: dict,
                     seed: str, binding_errors=(), physical_records=()) -> dict:
    actual, expected = _row_lookup(rows), expected_core_grid(groups, selections)
    expected.update(expected_cross_seed_items(groups, seed))
    error_cells = {(row["task"], row["direction"]) for row in binding_errors}
    need(len(error_cells) == len(binding_errors)
         and all(task in ("first_hop", "direct")
                 and direction in ("forward", "reverse")
                 and isinstance(row.get("reason"), str)
                 for row in binding_errors for task, direction in (
                     (row.get("task"), row.get("direction")),)),
         "Binding-specificity error records are malformed")
    for item_id, fields in tuple(expected.items()):
        if fields["condition"].startswith("binding_") \
                and (fields["condition"].removeprefix("binding_"),
                     fields["direction"]) in error_cells:
            expected.pop(item_id)
    for record in physical_records:
        if record["oracle_valid"]:
            expected[record["item_id"]] = {
                "condition": record["condition"], "direction": record["direction"],
                "logical_group_id": record["logical_group_id"],
                "recipient": record["recipient"], "target": record["oracle_target"],
                "base_render_id": record["base_render_id"],
                "donor_render_id": record["donor_render_id"],
            }
    missing = sorted(set(expected) - set(actual))
    need(not missing, f"Confirmation core grid is incomplete: {missing[:3]}")
    extra = sorted(set(actual) - set(expected))
    need(not extra, f"Confirmation contains unregistered rows: {extra[:3]}")
    for item_id, fields in expected.items():
        row = actual[item_id]
        need(all(row.get(name) == value for name, value in fields.items()),
             f"Confirmation row is not bound to frozen suite item {item_id}")
    return {"required_core_rows": len(expected), "extension_rows": 0}


def summarize(rows: list[dict]) -> dict:
    need(rows, "Cannot summarize an empty confirmation cell")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    need(all(len(cell) == 3 and {row["recipient"] for row in cell} == {0, 1, 2}
             for cell in grouped.values()), "Incomplete three-recipient confirmation cell")
    correct = sum(row["prediction"] == row["target"] for row in rows)
    exact = sum(all(row["prediction"] == row["target"] for row in cell)
                for cell in grouped.values())
    changed = [row for row in rows if row["recipient"] != 0]
    non_injected = sum(row["prediction"] != row["fixed_donor_answer"] for row in changed)
    return {"items": len(rows), "correct": correct, "item_accuracy": correct / len(rows),
            "groups": len(grouped), "exact_groups": exact,
            "group_accuracy": exact / len(grouped),
            "changed_recipient_items": len(changed), "non_injected": non_injected,
            "non_injection": non_injected / len(changed),
            "mean_probability_gain": sum(
                row["target_probability"] - row["base_target_probability"] for row in rows)
            / len(rows)}


def score_factor(rows: list[dict], factor: str) -> dict:
    def cells(prefix):
        result = defaultdict(list)
        for row in rows:
            if row["condition"].startswith(prefix):
                result[(row["condition"], row["direction"])].append(row)
        return {f"{condition}:{direction}": summarize(cell)
                for (condition, direction), cell in result.items()}
    return {"primary": cells(f"{factor}_selected_"),
            "controls": {family: cells(f"{factor}_{family}_")
                         for family in ("haar", "deranged", "equal_energy")}}


def score_cross_seed(rows: list[dict]) -> dict:
    cells = defaultdict(list)
    for row in rows:
        if row["condition"].startswith("content_cross_seed_"):
            cells[(row["condition"], row["direction"])].append(row)
    return {f"{condition}:{direction}": summarize(cell)
            for (condition, direction), cell in sorted(cells.items())}


def recursively_close(actual, expected, path="value"):
    if isinstance(expected, dict):
        need(isinstance(actual, dict) and actual.keys() == expected.keys(),
             f"{path} keys do not recompute")
        for key in expected:
            recursively_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        need(isinstance(actual, list) and len(actual) == len(expected),
             f"{path} list does not recompute")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            recursively_close(left, right, f"{path}[{index}]")
    elif type(expected) is float:
        need(type(actual) in (int, float) and math.isclose(
            actual, expected, rel_tol=1e-10, abs_tol=1e-10), f"{path} does not recompute")
    else:
        need(actual == expected, f"{path} does not recompute")


def _orthogonalize(raw, spectra):
    result, assigned = {}, torch.empty(raw["content"].shape[0], 0, dtype=torch.double)
    for name in ("content", "binding", "order"):
        weighted = raw[name].double() * spectra[name].double()[
            :raw[name].shape[1]].clamp_min(0).sqrt().unsqueeze(0)
        remainder = weighted - assigned @ (assigned.T @ weighted) \
            if assigned.shape[1] else weighted
        u, singular, _ = torch.linalg.svd(remainder, full_matrices=False)
        keep = singular > 1e-8 * max(float(singular[0]), 1.0) \
            if singular.numel() else torch.zeros(0, dtype=torch.bool)
        result[name] = u[:, keep]
        assigned = torch.cat((assigned, result[name]), 1)
    return result


def _blocked_basis(states, factor, nuisance, blocks, maximum_rank=64):
    states = states.double()
    factors, nuisances, block_levels = (tuple(sorted(set(value), key=repr))
                                        for value in (factor, nuisance, blocks))
    contrasts = []
    for block in block_levels:
        marginalized = []
        for level in factors:
            cells = []
            for nuisance_cell in nuisances:
                selected = [index for index, labels in enumerate(zip(
                    factor, nuisance, blocks, strict=True))
                            if labels == (level, nuisance_cell, block)]
                need(len(selected) == 1, "Deranged panel is not a complete factorial")
                cells.append(states[selected].mean(0))
            marginalized.append(torch.stack(cells).mean(0))
        marginalized = torch.stack(marginalized)
        contrasts.extend(marginalized - marginalized.mean(0))
    contrasts = torch.stack(contrasts)
    covariance = contrasts.T @ contrasts / len(contrasts)
    values, vectors = torch.linalg.eigh(covariance)
    order = values.argsort(descending=True)
    values, vectors = values[order], vectors[:, order]
    rank = min(int((values > 1e-10 * max(float(values[0].abs()), 1.0)).sum()), maximum_rank)
    return vectors[:, :rank], values


def _haar(width, rank, count, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = []
    for _ in range(count):
        sample = torch.randn(width, rank, generator=generator, dtype=torch.double)
        q, r = torch.linalg.qr(sample, mode="reduced")
        result.append(q * torch.where(torch.diag(r) < 0, -1.0, 1.0))
    return tuple(result)


def _deranged(geometry, factor_name, count, seed):
    panel = geometry["panels"][factor_name]
    states = panel["states"].double()
    factor, nuisance, blocks = (tuple(panel[name]) for name in ("factor", "nuisance", "blocks"))
    levels = tuple(sorted(set(factor), key=repr))
    block_levels = tuple(sorted(set(blocks), key=repr))
    nuisances = tuple(sorted(set(nuisance), key=repr))
    cell = {labels: index for index, labels in enumerate(zip(
        blocks, nuisance, factor, strict=True))}
    need(len(levels) == 2 and len(cell) == len(block_levels) * len(nuisances) * 2,
         "Malformed deranged-control panel")
    rng, output = random.Random(seed), []
    for _ in range(count):
        donors = list(block_levels)
        for right in range(len(donors) - 1, 0, -1):
            left = rng.randrange(right)
            donors[left], donors[right] = donors[right], donors[left]
        need(all(left != right for left, right in zip(block_levels, donors, strict=True)),
             "Deranged control retained a donor block")
        changed = states.clone()
        for base_block, donor_block in zip(block_levels, donors, strict=True):
            for nuisance_cell in nuisances:
                changed[cell[(base_block, nuisance_cell, levels[1])]] = \
                    states[cell[(donor_block, nuisance_cell, levels[1])]]
        basis, eigenvalues = _blocked_basis(changed, factor, nuisance, blocks)
        raw = {name: geometry["geometry"][name]["basis"]
               for name in ("content", "binding", "order")}
        spectra = {name: geometry["geometry"][name]["eigenvalues"]
                   for name in ("content", "binding", "order")}
        raw[factor_name], spectra[factor_name] = basis, eigenvalues
        output.append((_orthogonalize(raw, spectra)[factor_name], tuple(donors)))
    return output


def _same_subspace(left, right, atol=1e-7):
    return left.shape == right.shape and torch.allclose(
        left.double() @ left.double().T, right.double() @ right.double().T,
        atol=atol, rtol=atol)


def _registered_oracle_kind(mediator):
    label = mediator["label"] if mediator["kind"] == "single" else mediator["source_label"]
    return "g_slot" if label.startswith("matched_g.") else "f_slot"


def _physical_oracle_fields(group, base_name, donor_name, recipient, kind):
    base, donor = group[base_name][recipient], group[donor_name][recipient]
    for name in ("logical_id", "task", "query", "answer", "f_rows", "g_rows",
                 "distractor_rows"):
        need(base[name] == donor[name], "Physical-order oracle graph changed")
    original_rows = tuple(map(tuple, base["serialized_rows"]))
    reordered_rows = tuple(map(tuple, donor["serialized_rows"]))
    need(reordered_rows == original_rows[1:] + original_rows[:1],
         "Physical-order donor is not the registered cyclic reorder")
    layout = group["semantic_layouts"][base_name][recipient]

    def slot(role):
        positions = [index for index, candidate in enumerate(layout["roles"])
                     if candidate == f"{role}.left"]
        need(len(positions) == 1, "Physical-order semantic role is not unique")
        return layout["row_indices"][positions[0]]

    f_slot, g_slot = slot("queried_f"), slot("matched_g")
    f_occupant, g_occupant = reordered_rows[f_slot], reordered_rows[g_slot]
    mapping = dict(map(tuple, base["g_rows"]))
    target = (mapping.get(f_occupant[1]) if kind == "f_slot"
              else g_occupant[1] if kind == "g_slot"
              else g_occupant[1] if f_occupant[1] == g_occupant[0] else None)
    invalid_reason = None
    if target is None:
        invalid_reason = ("queried_f_slot_rhs_is_not_a_g_key" if kind == "f_slot"
                          else "physical_slot_rows_do_not_compose")
    structurally_valid = target is not None
    collision = structurally_valid and target == base["answer"]
    return {
        "oracle_kind": kind, "oracle_target": target,
        "oracle_valid": structurally_valid and not collision,
        "oracle_structurally_valid": structurally_valid,
        "oracle_semantic_collision": collision,
        "oracle_invalid_reason": (
            "semantic_target_collision" if collision else invalid_reason),
        "physical_f_slot_index": f_slot, "physical_g_slot_index": g_slot,
        "physical_f_slot_key": f_occupant[1], "physical_g_slot_key": g_occupant[0],
        "physical_g_slot_value": g_occupant[1],
        "physical_slots_compose": f_occupant[1] == g_occupant[0],
    }


def expected_physical_order_records(groups, mediator, selected_components):
    kind = _registered_oracle_kind(mediator)
    surfaces = {
        "marked": (("f0", "base", "reordered_base"),
                   ("f1", "donor", "reordered_donor")),
        "marker_free": (("f0", "marker_free_base", "marker_free_reordered_base"),
                        ("f1", "marker_free_donor", "marker_free_reordered_donor")),
        "format": (("f0", "format_base", "format_reordered_base"),
                   ("f1", "format_donor", "format_reordered_donor")),
        "distractor": (("f0", "distractor_base", "distractor_reordered_base"),
                       ("f1", "distractor_donor", "distractor_reordered_donor")),
    }
    output = []
    for component in ("order", "binding"):
        if component not in selected_components:
            continue
        for surface, assignments in surfaces.items():
            for assignment, base_name, donor_name in assignments:
                cell = []
                for group in groups:
                    for recipient in range(3):
                        base, donor = group[base_name][recipient], group[donor_name][recipient]
                        fields = _physical_oracle_fields(
                            group, base_name, donor_name, recipient, kind)
                        cell.append({
                            "condition": f"physical_order_{component}_{surface}_{assignment}",
                            "direction": "forward", "component": component,
                            "surface": surface, "assignment": assignment,
                            "logical_group_id": group["group_id"], "recipient": recipient,
                            "item_id": (f"order:physical:{component}:{surface}:{assignment}:"
                                        f"{group['group_id']}:{recipient}"),
                            "semantic_target": base["answer"],
                            "base_render_id": base["render_id"],
                            "donor_render_id": donor["render_id"], **fields,
                        })
                grouped = defaultdict(list)
                for record in cell:
                    grouped[record["logical_group_id"]].append(record)
                denominator = {
                    "kind": kind, "all_items": len(cell),
                    "structurally_valid_items": sum(
                        row["oracle_structurally_valid"] for row in cell),
                    "semantic_collision_items": sum(
                        row["oracle_semantic_collision"] for row in cell),
                    "adversarially_eligible_items": sum(row["oracle_valid"] for row in cell),
                    "item_denominator_rule": (
                        "structurally_valid_and_target_differs_from_semantic_answer"),
                    "all_groups": len(grouped),
                    "adversarially_eligible_groups": sum(
                        len(rows) == 3 and all(row["oracle_valid"] for row in rows)
                        for rows in grouped.values()),
                    "group_denominator_rule": (
                        "exactly_three_recipients_and_every_item_adversarially_eligible"),
                }
                output.extend({**record, "cell_denominator": denominator} for record in cell)
    return output


def score_physical_order(rows, records, component):
    selected_records = [row for row in records if row["component"] == component]
    selected_rows = {row["item_id"]: row for row in rows
                     if row["condition"].startswith(f"physical_order_{component}_")}
    eligible = [row for row in selected_records if row["oracle_valid"]]
    need(set(selected_rows) == {row["item_id"] for row in eligible},
         "Physical-order rows do not match the frozen adversarial denominator")

    def cell_summary(cell):
        valid = [row for row in cell if row["oracle_valid"]]
        grouped = defaultdict(list)
        for row in cell:
            grouped[(row["condition"], row["logical_group_id"])].append(row)
        eligible_groups = [group for group in grouped.values()
                           if len(group) == 3 and all(row["oracle_valid"] for row in group)]
        correct = sum(selected_rows[row["item_id"]]["prediction"] == row["oracle_target"]
                      for row in valid)
        exact = sum(all(selected_rows[row["item_id"]]["prediction"]
                        == row["oracle_target"] for row in group)
                    for group in eligible_groups)
        return {
            "registered_items": len(cell), "eligible_items": len(valid),
            "eligible_item_coverage": len(valid) / len(cell), "correct_items": correct,
            "item_accuracy": None if not valid else correct / len(valid),
            "registered_groups": len(grouped), "eligible_groups": len(eligible_groups),
            "eligible_group_coverage": len(eligible_groups) / len(grouped),
            "exact_groups": exact,
            "group_accuracy": None if not eligible_groups else exact / len(eligible_groups),
            "semantic_collisions": sum(row["oracle_semantic_collision"] for row in cell),
            "structurally_invalid": sum(
                not row["oracle_structurally_valid"] for row in cell),
        }

    cells = defaultdict(list)
    for record in selected_records:
        cells[f"{record['surface']}:{record['assignment']}"].append(record)
    return {"oracle_kind": selected_records[0]["oracle_kind"],
            "coverage_threshold": .50, "overall": cell_summary(selected_records),
            "cells": {key: cell_summary(cell) for key, cell in sorted(cells.items())}}


def _finite_state_batch(record, names):
    tensors = [record.get(name) for name in names]
    need(all(isinstance(value, torch.Tensor) and value.ndim == 2
             and torch.isfinite(value).all().item() for value in tensors)
         and len({tuple(value.shape) for value in tensors}) == 1,
         "Malformed retained control-state batch")
    need(len(record.get("item_ids", ())) == tensors[0].shape[0]
         and len(set(record["item_ids"])) == len(record["item_ids"]),
         "Retained control-state item IDs are malformed")
    return tensors


def verify_equal_energy_controls(cell, factor, basis, groups):
    expected_ids, expected_seeds = {}, {}
    for index in range(CONTROL_COUNT):
        prefix = f"{factor}_equal_energy_{index:02d}"
        expected_items = expected_transfer_items(
            groups, factor, prefix, CONTROL_SURFACES[factor])
        for item_id, fields in expected_items.items():
            expected_ids[(fields["condition"], fields["direction"])].append(item_id) \
                if (fields["condition"], fields["direction"]) in expected_ids \
                else expected_ids.setdefault(
                    (fields["condition"], fields["direction"]), [item_id])
        batch_index = 0
        for surface, _, _ in FACTOR_PAIRS[factor]:
            if surface not in CONTROL_SURFACES[factor]:
                continue
            for direction in ("forward", "reverse"):
                expected_seeds[(f"{prefix}_{surface}", direction)] = (
                    CONTROL_SEEDS[factor] + 1000 + 20 * index + batch_index)
                batch_index += 1
    expected_records = sum(2 for _ in CONTROL_SURFACES[factor]) * CONTROL_COUNT
    need(isinstance(cell, (tuple, list)) and len(cell) == expected_records,
         "Equal-energy retained batch count changed")
    seen = set()
    for record in cell:
        key = (record.get("condition"), record.get("direction"))
        need(key in expected_ids and key not in seen
             and tuple(record.get("item_ids", ())) == tuple(expected_ids[key])
             and record.get("seed") == expected_seeds[key],
             "Equal-energy retained batch is not bound to the frozen grid")
        seen.add(key)
        base, donor, replacement = _finite_state_batch(
            record, ("base_states", "donor_states", "replacement_states"))
        local_basis = basis.double()
        selected = ((donor.double() - base.double()) @ local_basis) @ local_basis.T
        generator = torch.Generator(device="cpu").manual_seed(record["seed"])
        noise = torch.randn(base.shape, generator=generator, dtype=torch.double)
        noise = noise - (noise @ local_basis) @ local_basis.T
        need((noise.norm(dim=1) > 1e-12).all().item(),
             "Degenerate retained equal-energy noise")
        expected = base.double() + noise / noise.norm(dim=1, keepdim=True) \
            * selected.norm(dim=1, keepdim=True)
        need(torch.equal(replacement, expected.to(base.dtype)),
             "Equal-energy replacement tensor does not regenerate")
        perturbation = replacement.double() - base.double()
        need(torch.allclose(perturbation @ local_basis,
                            torch.zeros_like(perturbation @ local_basis), atol=1e-6)
             and torch.allclose(perturbation.norm(dim=1), selected.norm(dim=1),
                                atol=1e-6, rtol=1e-6),
             "Equal-energy replacement violates norm or off-subspace constraint")
    need(seen == set(expected_ids), "Equal-energy retained batches are incomplete")


def verify_necessity_controls(records, geometry, basis, groups):
    need(isinstance(records, (tuple, list)) and len(records) == 2,
         "Necessity control batches are incomplete")
    mean = geometry["geometry"]["content"]["mean"].double()
    expected_seed = CONTROL_SEEDS["content"] + 5000
    for offset, (assignment, name) in enumerate((("f0", "base"), ("f1", "donor"))):
        record = records[offset]
        expected_ids = tuple(f"content:necessity:{assignment}:{group['group_id']}:{recipient}"
                             for group in groups for recipient in range(3))
        need(record.get("assignment") == assignment
             and record.get("seed") == expected_seed + offset
             and tuple(record.get("item_ids", ())) == expected_ids,
             "Necessity control batch is not bound to the frozen grid")
        base, ablation, corruption, restoration = _finite_state_batch(record, (
            "base_states", "mean_ablation_states", "corruption_states",
            "restoration_states"))
        centered, local_basis = base.double() - mean, basis.double()
        expected_ablation = base.double() - (centered @ local_basis) @ local_basis.T
        generator = torch.Generator(device="cpu").manual_seed(record["seed"])
        noise = torch.randn(base.shape, generator=generator, dtype=torch.double)
        noise = noise - (noise @ local_basis) @ local_basis.T
        selected = (centered @ local_basis) @ local_basis.T
        complement = centered - selected
        expected_corruption = mean + complement + noise / noise.norm(
            dim=1, keepdim=True) * selected.norm(dim=1, keepdim=True)
        need(torch.equal(ablation, expected_ablation.to(base.dtype))
             and torch.equal(corruption, expected_corruption.to(base.dtype))
             and torch.equal(restoration, base),
             "Necessity control tensors do not independently regenerate")


def verify_projected_state_batches(records, expected_items, basis, *, rotation=None):
    need(isinstance(records, (tuple, list)) and records,
         "Retained projected-state batches are missing")
    seen = set()
    for record in records:
        key = (record.get("condition"), record.get("direction"))
        need(key in expected_items and key not in seen
             and tuple(record.get("item_ids", ())) == tuple(expected_items[key]),
             "Projected-state batch is not bound to the frozen grid")
        seen.add(key)
        if rotation is None:
            base, donor, replacement = _finite_state_batch(
                record, ("base_states", "donor_states", "replacement_states"))
            expected = base.double() + (((donor.double() - base.double()) @ basis.double())
                                        @ basis.double().T)
        else:
            source_base, source_donor, target_base, replacement = _finite_state_batch(
                record, ("source_base_states", "source_donor_states",
                         "target_base_states", "replacement_states"))
            expected = target_base.double() + (((source_donor.double()
                                                 - source_base.double()) @ basis.double())
                                               @ basis.double().T) @ rotation.double()
        need(torch.equal(replacement, expected.to(replacement.dtype)),
             "Retained projected replacement tensor does not recompute")
    need(seen == set(expected_items), "Projected-state batches are incomplete")


def verify_controls(metadata: dict, discovery: dict, seed: str, groups: list[dict],
                    mediator: dict) -> tuple[dict, list[dict]]:
    path = artifact_path(metadata, f"seed {seed} control tensor")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    need(payload.get("format") == "TEACH-0013-subspace-controls-v1"
         and set(payload) == {"format", "controls"}, "Unknown control tensor format")
    controls = payload["controls"]
    geometry_metadata = discovery["artifacts"][seed]["geometry"]
    geometry_path = artifact_path(geometry_metadata, f"seed {seed} geometry")
    geometry = torch.load(geometry_path, map_location="cpu", weights_only=True)
    expected_selected = {}
    for factor in ("content", "binding", "order"):
        rank = discovery["rank_selections"][factor]["selection"]
        if rank is not None:
            expected_selected[factor] = geometry["orthogonal"]["forward"][factor][:, :rank]
    need(set(controls.get("selected", {})) == set(expected_selected),
         "Selected control bases do not match frozen rank selections")
    for factor, expected in expected_selected.items():
        actual = controls["selected"][factor]
        need(torch.equal(actual, expected), "Selected basis changed after discovery")
        gram = actual.double().T @ actual.double()
        need(torch.allclose(gram, torch.eye(actual.shape[1], dtype=torch.double),
                            atol=1e-7, rtol=1e-7), "Selected basis is not orthonormal")
        cell = controls.get(factor, {})
        need(set(cell) >= {"haar", "deranged"}
             and len(cell["haar"]) == len(cell["deranged"]) == CONTROL_COUNT,
             "Control basis count changed")
        expected_haar = _haar(actual.shape[0], actual.shape[1], CONTROL_COUNT,
                              CONTROL_SEEDS[factor])
        need(all(torch.equal(left, right) for left, right in zip(
            cell["haar"], expected_haar, strict=True)), "Haar controls do not regenerate")
        expected_deranged = _deranged(
            geometry, factor, CONTROL_COUNT, CONTROL_SEEDS[factor])
        for stored, (basis, donor_blocks) in zip(
                cell["deranged"], expected_deranged, strict=True):
            need(tuple(stored["donor_blocks"]) == donor_blocks
                 and _same_subspace(stored["basis"], basis),
                 "Deranged-pair control does not independently regenerate")
        verify_equal_energy_controls(cell.get("equal_energy"), factor, actual, groups)
    verify_necessity_controls(
        controls.get("necessity"), geometry, expected_selected["content"], groups)
    source_seed = "1" if seed == "0" else "0"
    cross = controls.get("cross_seed")
    need(isinstance(cross, dict) and cross.get("source_seed") == source_seed
         and cross.get("target_seed") == seed,
         "Cross-seed control metadata is missing or misdirected")
    source_geometry_path = artifact_path(
        discovery["artifacts"][source_seed]["geometry"],
        f"cross-seed source geometry {source_seed}")
    source_geometry = torch.load(source_geometry_path, map_location="cpu", weights_only=True)
    content_rank = discovery["rank_selections"]["content"]["selection"]
    source_basis = source_geometry["orthogonal"]["forward"]["content"][:, :content_rank]
    need(torch.equal(cross.get("source_content_basis"), source_basis),
         "Cross-seed source basis differs from audited discovery")
    cross_path = artifact_path(discovery["artifacts"]["cross_seed"],
                               "shared cross-seed geometry")
    cross_geometry = torch.load(cross_path, map_location="cpu", weights_only=True)
    need(cross_geometry.get("format") == "TEACH-0013-cross-seed-geometry-v2",
         "Unknown shared cross-seed geometry format")
    transpose = source_seed == "1"
    expected_shared = cross_geometry["alignment"]["shared"]["rotation"]
    expected_deranged_rotation = cross_geometry[
        "alignment"]["shared_deranged_control"]["rotation"]
    if transpose:
        expected_shared = expected_shared.T
        expected_deranged_rotation = expected_deranged_rotation.T
    need(torch.equal(cross.get("shared_rotation"), expected_shared)
         and torch.equal(cross.get("deranged_rotation"), expected_deranged_rotation),
         "Finite cross-seed transport maps differ from audited discovery")
    expected_transport = {}
    for item_id, fields in expected_cross_seed_items(groups, seed).items():
        expected_transport.setdefault((fields["condition"], fields["direction"]), []).append(
            item_id)
    shared_items = {key: value for key, value in expected_transport.items()
                    if "_shared_" in key[0]}
    deranged_items = {key: value for key, value in expected_transport.items()
                      if "_deranged_" in key[0]}
    transport = cross.get("transport_batches")
    verify_projected_state_batches(
        [record for record in transport or () if record.get("map_name") == "shared"],
        shared_items, source_basis, rotation=expected_shared)
    verify_projected_state_batches(
        [record for record in transport or () if record.get("map_name") == "deranged"],
        deranged_items, source_basis, rotation=expected_deranged_rotation)
    records = expected_physical_order_records(groups, mediator, set(expected_selected))
    need(controls.get("physical_order_oracles") == records,
         "Physical-order oracle records do not independently reconstruct")
    state_batches = controls.get("physical_order_state_batches")
    for component in ("order", "binding"):
        if component not in expected_selected:
            continue
        expected_batches = defaultdict(list)
        for record in records:
            if record["component"] == component and record["oracle_valid"]:
                expected_batches[(record["condition"], record["direction"])].append(
                    record["item_id"])
        verify_projected_state_batches(
            [record for record in state_batches or () if record.get("component") == component],
            expected_batches, expected_selected[component])
    return {"selected_factors": sorted(expected_selected),
            "haar_per_factor": CONTROL_COUNT, "deranged_per_factor": CONTROL_COUNT,
            "cross_seed_source": source_seed,
            "binding_specificity_errors": controls.get("binding_specificity_errors", []),
            "equal_energy_state_tensors_retained": True}, records


def _row_cells(rows, prefix):
    cells = defaultdict(list)
    for row in rows:
        if row["condition"].startswith(prefix):
            cells[(row["condition"], row["direction"])].append(row)
    return cells


def _single_item_summary(rows):
    need(rows and len({row["item_id"] for row in rows}) == len(rows),
         "Single-item decision cell is empty or duplicated")
    correct = sum(row["prediction"] == row["target"] for row in rows)
    clean = sum(row["clean_prediction"] == row["target"] for row in rows)
    return {"items": len(rows), "correct": correct, "item_accuracy": correct / len(rows),
            "clean_correct": clean, "clean_accuracy": clean / len(rows),
            "accuracy_loss": (clean - correct) / len(rows)}


def _factor_decision(rows, factor, item_threshold, group_threshold):
    selected = _row_cells(rows, f"{factor}_selected_")
    complement = _row_cells(rows, f"{factor}_complement_")
    primary = {f"{condition}:{direction}": summarize(cell)
               for (condition, direction), cell in sorted(selected.items())}
    complement_scores = {
        f"{condition}:{direction}": {
            "items": len(cell),
            "base_preservation": sum(row["prediction"] == row["base_answer"]
                                     for row in cell) / len(cell),
            "donor_target_rate": sum(row["prediction"] == row["target"]
                                     for row in cell) / len(cell),
        } for (condition, direction), cell in sorted(complement.items())}
    expected_cells = 16 if factor == "order" else 10
    transfer_pass = len(primary) == expected_cells and all(
        cell["item_accuracy"] >= item_threshold
        and cell["group_accuracy"] >= group_threshold
        and (factor != "content" or cell["non_injection"] >= .90)
        for cell in primary.values())
    complement_pass = len(complement_scores) == expected_cells and all(
        cell["base_preservation"] >= .90 and cell["donor_target_rate"] <= .10
        for cell in complement_scores.values())
    advantages = {}
    for direction in ("forward", "reverse"):
        primary_key = (f"{factor}_selected_marked", direction)
        if primary_key not in selected:
            continue
        primary_accuracy = summarize(selected[primary_key])["item_accuracy"]
        control_scores = {}
        for family in ("haar", "deranged", "equal_energy"):
            matches = [summarize(cell)["item_accuracy"]
                       for (condition, local_direction), cell in _row_cells(
                           rows, f"{factor}_{family}_").items()
                       if condition.endswith("_marked") and local_direction == direction]
            if matches:
                control_scores[family] = max(matches)
        if factor == "binding":
            order_cell = _row_cells(rows, "binding_order_control_marked").get(
                ("binding_order_control_marked", direction))
            if order_cell:
                control_scores["order"] = summarize(order_cell)["item_accuracy"]
        required = ({"haar", "deranged", "equal_energy", "order"}
                    if factor == "binding" else {"haar", "deranged", "equal_energy"})
        best = max(control_scores.values()) if control_scores else None
        advantages[direction] = {
            "primary_item_accuracy": primary_accuracy, "controls": control_scores,
            "best_control_item_accuracy": best,
            "advantage": None if best is None else primary_accuracy - best,
            "pass": (set(control_scores) == required and best is not None
                     and primary_accuracy - best >= .35),
        }
    return {"primary": primary, "complement": complement_scores,
            "control_advantages": advantages, "transfer_pass": transfer_pass,
            "complement_pass": complement_pass,
            "control_pass": set(advantages) == {"forward", "reverse"}
            and all(cell["pass"] for cell in advantages.values())}


def recompute_decisions(rows_by_seed, controls_by_seed, discovery):
    gates = {"content": {}, "binding": {}, "cross_seed_content": {}, "order": {}}
    for seed, rows in rows_by_seed.items():
        if discovery["rank_selections"]["content"]["selection"] is not None:
            content = _factor_decision(rows, "content", .80, .60)
            necessity = {}
            for assignment in ("f0", "f1"):
                corruption = [row for row in rows if row["condition"]
                              == f"content_equal_norm_corruption_{assignment}"]
                restoration = [row for row in rows if row["condition"]
                               == f"content_native_restoration_{assignment}"]
                if corruption and restoration:
                    corrupt_score, restore_score = (_single_item_summary(value)
                                                     for value in (corruption, restoration))
                    necessity[assignment] = {
                        "corruption": corrupt_score, "restoration": restore_score,
                        "clean_to_corrupt_drop": (corrupt_score["clean_accuracy"]
                                                  - corrupt_score["item_accuracy"]),
                        "pass_without_exact_logit_audit": (
                            corrupt_score["clean_accuracy"]
                            - corrupt_score["item_accuracy"] >= .30
                            and restore_score["item_accuracy"] >= .95),
                    }
            specificity = {}
            for task in ("first_hop", "direct", "copy"):
                selected = [row for row in rows if row["condition"] == f"content_{task}"]
                if selected:
                    specificity[task] = _single_item_summary(selected)
            necessity_pass = set(necessity) == {"f0", "f1"} and all(
                cell["pass_without_exact_logit_audit"] for cell in necessity.values())
            specificity_pass = (set(specificity) == {"first_hop", "direct", "copy"}
                                and specificity["first_hop"]["item_accuracy"] >= .80
                                and specificity["direct"]["item_accuracy"] >= .80
                                and specificity["copy"]["accuracy_loss"] <= .05)
            content.update({"necessity": necessity, "specificity": specificity,
                            "necessity_pass_without_exact_logit_audit": necessity_pass,
                            "specificity_pass": specificity_pass,
                            "candidate_pass_without_artifact_audit": (
                                content["transfer_pass"] and content["complement_pass"]
                                and content["control_pass"] and necessity_pass
                                and specificity_pass)})
            gates["content"][seed] = content

        cross = score_cross_seed(rows)
        shared = {key: value for key, value in cross.items() if "_shared_" in key}
        deranged = {key: value for key, value in cross.items() if "_deranged_" in key}
        cross_cells = {}
        for key, score in shared.items():
            control = deranged.get(key.replace("_shared_", "_deranged_"))
            cross_cells[key] = {
                "shared": score, "deranged": control,
                "advantage": None if control is None else (
                    score["item_accuracy"] - control["item_accuracy"]),
                "pass": (control is not None and score["item_accuracy"] >= .80
                         and score["group_accuracy"] >= .60 and score["non_injection"] >= .90
                         and score["item_accuracy"] - control["item_accuracy"] >= .35),
            }
        gates["cross_seed_content"][seed] = {
            "cells": cross_cells, "candidate_pass_without_artifact_audit": (
                len(cross_cells) == 2 and all(cell["pass"] for cell in cross_cells.values()))}

        if discovery["rank_selections"]["binding"]["selection"] is not None:
            binding = _factor_decision(rows, "binding", .75, .55)
            specificity = {}
            for task in ("first_hop", "direct"):
                for direction in ("forward", "reverse"):
                    selected = [row for row in rows if row["condition"] == f"binding_{task}"
                                and row["direction"] == direction]
                    if selected:
                        specificity[f"{task}:{direction}"] = _single_item_summary(selected)
            errors = controls_by_seed[seed]["binding_specificity_errors"]
            specificity_pass = (len(specificity) == 4 and not errors and all(
                cell["item_accuracy"] >= .90 for cell in specificity.values()))
            binding.update({"specificity": specificity, "specificity_errors": errors,
                            "specificity_pass": specificity_pass,
                            "candidate_pass_without_artifact_audit": (
                                binding["transfer_pass"] and binding["control_pass"]
                                and specificity_pass)})
            gates["binding"][seed] = binding
        if discovery["rank_selections"]["order"]["selection"] is not None:
            order = score_physical_order(
                rows, controls_by_seed[seed]["physical_order_records"], "order")
            overall = order["overall"]
            order["coverage_pass"] = (overall["eligible_item_coverage"] >= .50
                                      and overall["eligible_group_coverage"] >= .50)
            gates["order"][seed] = order
    content_pass = len(gates["content"]) == 2 and all(
        row["candidate_pass_without_artifact_audit"] for row in gates["content"].values())
    cross_pass = len(gates["cross_seed_content"]) == 2 and all(
        row["candidate_pass_without_artifact_audit"]
        for row in gates["cross_seed_content"].values())
    binding_pass = len(gates["binding"]) == 2 and all(
        row["candidate_pass_without_artifact_audit"] for row in gates["binding"].values())
    coverage = len(gates["order"]) == 2 and all(
        row["coverage_pass"] for row in gates["order"].values())
    accuracies = [row["overall"]["item_accuracy"] for row in gates["order"].values()]
    binding_measured = len(gates["binding"]) == 2
    order_supported = (coverage and all(value is not None and value >= .75
                                        for value in accuracies)
                       and binding_measured and not binding_pass)
    binding_reordered = binding_measured and all(
        len([key for key in row["primary"] if "_reordered:" in key]) == 2
        and all(cell["item_accuracy"] >= .75 and cell["group_accuracy"] >= .55
                for key, cell in row["primary"].items() if "_reordered:" in key)
        for row in gates["binding"].values())
    insufficient = (coverage and all(value is not None and value <= .20
                                     for value in accuracies) and binding_reordered)
    labels = {
        "content": "CONTENT-SUBSPACE-SUPPORTED" if content_pass
        else "CONTENT-SUBSPACE-NOT-SUPPORTED",
        "cross_seed_content": "CROSS-SEED-CAUSAL-TRANSPORT-SUPPORTED" if cross_pass
        else "CROSS-SEED-CAUSAL-TRANSPORT-NOT-SUPPORTED",
        "binding": "BINDING-SUBSPACE-SUPPORTED" if binding_pass
        else "BINDING-SUBSPACE-NOT-SUPPORTED",
        "order": ("ORDER-SHORTCUT-SUPPORTED" if order_supported else
                  "ORDER COMPONENT NOT CAUSALLY SUFFICIENT" if insufficient else
                  "INCONCLUSIVE: PHYSICAL-ORDER COVERAGE" if gates["order"] and not coverage
                  else "INCONCLUSIVE: BINDING COMPARATOR" if gates["order"]
                  and not binding_measured else "ORDER-SHORTCUT-NOT-SUPPORTED"),
    }
    return {"status": "candidate_decisions_pending_independent_artifact_audit",
            "thresholds_frozen_before_confirmation": True,
            "exact_native_restoration_logit_error_required": "<1e-6",
            "gates": gates, "candidate_labels": labels}


def capability_requirements(report: dict, rows_by_seed: dict,
                            controls_by_seed: dict, *, decision_match=False,
                            restoration_exact=False) -> dict[str, bool]:
    all_rows = [row for rows in rows_by_seed.values() for row in rows]
    conditions = {row["condition"] for row in all_rows}
    directions = {row["direction"] for row in all_rows}
    decisions = report.get("decisions")
    return {
        "finite_cross_seed_shared_transport": any(
            "cross_seed" in condition and "deranged" not in condition
            for condition in conditions),
        "finite_cross_seed_deranged_map_control": any(
            "cross_seed" in condition and "deranged" in condition
            for condition in conditions),
        "both_cross_seed_directions": (
            any(any(token in direction for token in ("seed0_to_seed1", "0_to_1"))
                for direction in directions)
            and any(any(token in direction for token in ("seed1_to_seed0", "1_to_0"))
                    for direction in directions)),
        "physical_order_oracle_rows": any(
            condition.startswith("physical_order_") for condition in conditions),
        "retained_equal_energy_state_tensors": all(
            value["equal_energy_state_tensors_retained"]
            for value in controls_by_seed.values()),
        "independent_decision_object": isinstance(decisions, dict)
            and decisions.get("status")
            == "candidate_decisions_pending_independent_artifact_audit"
            and set(decisions.get("candidate_labels", {}))
            == {"content", "binding", "order", "cross_seed_content"},
        "independent_decision_recomputation": decision_match,
        "native_restoration_exact_logits_below_1e-6": restoration_exact,
        "final_confirmation_status": report.get("status")
            == "stage_d_confirmation_complete_pending_audit",
    }


def auditor_provenance() -> dict:
    relative = "scripts/teacher0013_subspace_confirmation_audit.py"
    status = subprocess.run(["git", "status", "--porcelain", "--", relative], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.strip()
    need(not status, "Commit the Stage-D confirmation auditor before audit")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    return {"revision": revision, "sha256": {relative: file_digest(ROOT / relative)}}


def audit(report_path: Path, output_path: Path) -> dict:
    report = json.loads(report_path.read_text())
    need(report.get("experiment") == "TEACH-0013"
         and report.get("mode") == "subspace_confirmation"
         and report.get("status") in {"stage_d_confirmation_core_complete",
                                      "stage_d_confirmation_complete_pending_audit",
                                      "stage_d_confirmation_complete",
                                      "stage_d_confirmation_decided"},
         "Not a recognized Stage-D confirmation report")
    result_dir = report_path.parent
    output_relative = Path(report.get("output_directory", ""))
    need(output_relative and not output_relative.is_absolute()
         and ".." not in output_relative.parts,
         "Confirmation output directory is not safely recorded")
    output_dir = (ROOT / output_relative).resolve()
    need(output_dir.is_relative_to(ROOT.resolve()),
         "Confirmation output directory escapes the repository")
    manifest_path = result_dir / "suite-manifest.json"
    discovery_path = result_dir / "subspace-discovery.json"
    discovery_audit_path = result_dir / "subspace-discovery-audit.json"
    need(manifest_path.is_file() and discovery_path.is_file()
         and discovery_audit_path.is_file(), "Stage-D prerequisites are missing")
    manifest = json.loads(manifest_path.read_text())
    discovery = json.loads(discovery_path.read_text())
    discovery_audit = json.loads(discovery_audit_path.read_text())
    need(file_digest(discovery_path) == report["subspace_discovery_sha256"]
         and file_digest(discovery_audit_path) == report["subspace_discovery_audit_sha256"]
         and discovery_audit.get("audit") == "pass"
         and discovery_audit.get("scope") == "stage_d_discovery"
         and discovery_audit.get("subspace_discovery_sha256")
         == report["subspace_discovery_sha256"]
         and discovery_audit.get("rank_selections") == discovery["rank_selections"]
         == report["rank_selections"], "Audited discovery prerequisite changed")
    need(manifest.get("status") == "suite_frozen"
         and report.get("suite_gzip_sha256") == manifest.get("suite_gzip_sha256")
         and report.get("confirmation_source_git_head") == manifest.get("source_git_head")
         and report.get("confirmation_source_sha256") == manifest.get("source_sha256"),
         "Confirmation is not bound to frozen source")
    suite_path = artifact_path({"path": manifest["suite_path"],
                                "sha256": manifest["suite_gzip_sha256"],
                                "bytes": manifest["suite_gzip_bytes"]}, "frozen suite")
    suite = json.loads(gzip.decompress(suite_path.read_bytes()))
    groups = suite["splits"]["confirmation"]
    need([group["group_id"] for group in groups] == manifest["group_ids"]["confirmation"],
         "Frozen confirmation group order changed")
    need(report.get("mediator") == discovery.get("mediator"),
         "Confirmation mediator differs from audited discovery")

    rows_by_seed, exact_by_seed, controls_by_seed, recomputed_scores = {}, {}, {}, {}
    need(set(report.get("artifacts", {})) == {"0", "1"},
         "Confirmation must retain artifacts for both seeds")
    for seed in ("0", "1"):
        seed_artifacts = report["artifacts"][seed]
        rows = load_rows(seed_artifacts["rows"])
        group_ids = {group["group_id"] for group in groups}
        for row in rows:
            validate_row(row, group_ids)
        controls, physical_records = verify_controls(
            seed_artifacts["controls"], discovery, seed, groups, report["mediator"])
        grid = verify_core_grid(
            rows, groups, report["rank_selections"], seed,
            binding_errors=controls["binding_specificity_errors"],
            physical_records=physical_records)
        exact = verify_sharded_logits(seed_artifacts["exact_symbol_logits"], rows)
        scores = {factor: score_factor(rows, factor)
                  for factor in ("content", "binding", "order")
                  if report["rank_selections"][factor]["selection"] is not None}
        if any(row["condition"].startswith("content_cross_seed_") for row in rows):
            scores["cross_seed_content"] = score_cross_seed(rows)
        for component in ("order", "binding"):
            if component in controls["selected_factors"]:
                scores[f"physical_order_{component}"] = score_physical_order(
                    rows, physical_records, component)
        recursively_close(report["scores"][seed], scores, f"scores.{seed}")
        controls["physical_order_records"] = physical_records
        rows_by_seed[seed], exact_by_seed[seed] = rows, {**exact, **grid}
        controls_by_seed[seed], recomputed_scores[seed] = controls, scores

    need(report["elapsed_seconds"] >= 0 and report["materialized_activation_bytes"] >= 0
         and math.isclose(report["cumulative_stage_c_e_seconds"],
                          discovery["cumulative_stage_c_e_seconds"]
                          + report["elapsed_seconds"], abs_tol=1e-6)
         and math.isclose(report["cumulative_campaign_seconds"],
                          discovery["cumulative_campaign_seconds"]
                          + report["elapsed_seconds"], abs_tol=1e-6)
         and report["cumulative_materialized_bytes"]
         == discovery["cumulative_materialized_bytes"]
         + report["materialized_activation_bytes"],
         "Confirmation cumulative accounting does not reconcile")
    need(report["cumulative_stage_c_e_seconds"] <= MAX_STAGE_C_E_SECONDS
         and report["cumulative_campaign_seconds"] <= MAX_CAMPAIGN_SECONDS
         and report["peak_sampled_current_allocated_bytes"] <= MAX_CURRENT_BYTES
         and report["cumulative_materialized_bytes"] <= MAX_TRAFFIC_BYTES
         and report["ignored_output_bytes"] <= MAX_OUTPUT_BYTES
         and report["tracked_result_bytes"] <= MAX_RESULT_BYTES
         and tree_bytes(output_dir) <= MAX_OUTPUT_BYTES
         and tree_bytes(result_dir) <= MAX_RESULT_BYTES,
         "Stage-D confirmation resource or artifact ceiling exceeded")

    recomputed_decisions = recompute_decisions(rows_by_seed, controls_by_seed, discovery)
    recursively_close(report.get("decisions"), recomputed_decisions, "decisions")
    restoration_exact = all(
        row["max_native_restoration_logit_error"] is not None
        and row["max_native_restoration_logit_error"] < 1e-6
        for row in exact_by_seed.values())
    requirements = capability_requirements(
        report, rows_by_seed, controls_by_seed, decision_match=True,
        restoration_exact=restoration_exact)
    passed = all(requirements.values())
    result = {
        "audit": "pass" if passed else "fail_closed",
        "experiment": "TEACH-0013", "scope": "stage_d_confirmation",
        "status": report["status"], "subspace_confirmation_sha256": file_digest(report_path),
        "rank_selections": report["rank_selections"],
        "recomputed_scores": recomputed_scores, "verified_exact_logits": exact_by_seed,
        "verified_controls": controls_by_seed, "capability_requirements": requirements,
        "missing_requirements": sorted(name for name, value in requirements.items() if not value),
        "recomputed_decisions": recomputed_decisions,
        "final_labels": recomputed_decisions["candidate_labels"] if passed else None,
        "auditor": auditor_provenance(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path,
                        default=ROOT / "results/TEACH-0013/subspace-confirmation.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/TEACH-0013/subspace-confirmation-audit.json")
    args = parser.parse_args()
    print(json.dumps(audit(args.report, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
