#!/usr/bin/env python3
"""Sealed TEACH-0013 Stage-D causal-subspace confirmation campaign."""

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import time

import torch

from voynich.workspace.teacher13_confirm import (
    MediatorSpec,
    cross_task_mediator_logits,
    diagnostic_rows,
    summarize_confirmation_rows,
)
from voynich.workspace.teacher13_discovery import episode_from_record
from voynich.workspace.teacher13_geometry import (
    deranged_blocked_bases,
    haar_random_bases,
    orthogonal_factor_geometry,
    orthonormal_union,
)
from voynich.workspace.teacher13_intervene import (
    load_raw_checkpoint,
    materialized_tensor_counter,
    raw_forward,
)
from voynich.workspace.teacher13_subspace import (
    equal_norm_component_corruption_states,
    factor_episode_batches,
    factor_equal_energy_rows,
    factor_transfer_rows,
    mean_ablation_states,
    mediator_endpoint_logits,
    mediator_state_pairs,
)
from voynich.workspace.teacher13_tasks import (
    physical_order_shortcut_denominators,
    physical_order_shortcut_oracle,
    registered_physical_order_oracle_kind,
)


ROOT = Path(__file__).resolve().parents[1]
CONTROL_COUNT = 32
CONTROL_SEEDS = {"content": 73411, "binding": 73421, "order": 73431}
CONTROL_SURFACES = {"content": ("marked",), "binding": ("marked",),
                    "order": ("f0_marked", "f1_marked")}
ORDER_SHORTCUT_SURFACES = {
    "marked": (("f0", "base", "reordered_base"),
               ("f1", "donor", "reordered_donor")),
    "marker_free": (("f0", "marker_free_base", "marker_free_reordered_base"),
                    ("f1", "marker_free_donor", "marker_free_reordered_donor")),
    "format": (("f0", "format_base", "format_reordered_base"),
               ("f1", "format_donor", "format_reordered_donor")),
    "distractor": (("f0", "distractor_base", "distractor_reordered_base"),
                   ("f1", "distractor_donor", "distractor_reordered_donor")),
}
MAX_STAGE_C_E_SECONDS = 5400.0
MAX_CAMPAIGN_SECONDS = 14_400.0
MAX_CURRENT_BYTES = 24 * 1024**3
MAX_TRAFFIC_BYTES = 300 * 1024**3
MAX_OUTPUT_BYTES = 20 * 1024**3
MAX_RESULT_BYTES = 2 * 1024**3


def _load_stage_c():
    spec = importlib.util.spec_from_file_location(
        "teacher0013_confirm_dependency", ROOT / "scripts/teacher0013_confirm.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load frozen Stage-C dependency")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ShardedLogitCapture:
    """Write each exact-logit record immediately to keep peak host memory bounded."""

    def __init__(self, directory: Path, stage_c):
        if directory.exists():
            raise FileExistsError("No automatic Stage-D exact-logit overwrite")
        directory.mkdir(parents=True)
        self.directory, self.stage_c, self.shards = directory, stage_c, []

    def __call__(self, condition, direction, metadata, logits, clean_logits):
        item_ids = tuple(row.get(
            "item_id", f"{condition}:{direction}:{row['logical_group_id']}:{row['recipient']}")
            for row in metadata)
        record = {"condition": condition, "direction": direction,
                  "logical_group_ids": tuple(row["logical_group_id"] for row in metadata),
                  "recipients": tuple(row["recipient"] for row in metadata),
                  "targets": tuple(row["target"] for row in metadata),
                  "item_ids": item_ids, "edited_symbol_logits": logits.contiguous(),
                  "clean_symbol_logits": clean_logits.contiguous()}
        path = self.directory / f"{len(self.shards):05d}.pt"
        torch.save({"format": "TEACH-0013-symbol-logit-shard-v1", "record": record}, path)
        self.shards.append({"path": str(path.relative_to(ROOT)),
                            "sha256": self.stage_c.sha_file(path),
                            "bytes": path.stat().st_size, "rows": len(item_ids)})

    def metadata(self):
        return {"format": "TEACH-0013-sharded-symbol-logits-v1",
                "shards": self.shards, "records": len(self.shards),
                "rows": sum(shard["rows"] for shard in self.shards)}


def prerequisite(result_dir: Path, stage_c):
    discovery_path = result_dir / "subspace-discovery.json"
    audit_path = result_dir / "subspace-discovery-audit.json"
    discovery, audit = (json.loads(path.read_text()) for path in (discovery_path, audit_path))
    if discovery.get("status") != "stage_d_discovery_complete" \
            or audit.get("audit") != "pass" or audit.get("scope") != "stage_d_discovery" \
            or audit.get("subspace_discovery_sha256") != stage_c.sha_file(discovery_path) \
            or audit.get("rank_selections") != discovery.get("rank_selections"):
        raise RuntimeError("Matching independently audited Stage-D discovery required")
    if discovery.get("rank_selections", {}).get("content", {}).get("selection") is None:
        raise RuntimeError("Stage-D confirmation requires a discovery-selected content rank")
    return discovery, audit, MediatorSpec(**discovery["mediator"])


def load_geometry(discovery, seed, stage_c):
    metadata = discovery["artifacts"][seed]["geometry"]
    path = ROOT / metadata["path"]
    if not path.is_file() or path.stat().st_size != metadata["bytes"] \
            or stage_c.sha_file(path) != metadata["sha256"]:
        raise RuntimeError("Stage-D geometry artifact mismatch")
    return torch.load(path, map_location="cpu", weights_only=True)


def load_cross_seed_geometry(discovery, stage_c):
    """Load the independently audited shared discovery alignment fail-closed."""
    metadata = discovery["artifacts"]["cross_seed"]
    path = ROOT / metadata["path"]
    if not path.is_file() or path.stat().st_size != metadata["bytes"] \
            or stage_c.sha_file(path) != metadata["sha256"]:
        raise RuntimeError("Stage-D cross-seed artifact mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "TEACH-0013-cross-seed-geometry-v2" \
            or set(payload.get("alignment", {})) < {"shared", "shared_deranged_control"}:
        raise RuntimeError("Unexpected Stage-D cross-seed artifact format")
    return payload


def selected_bases(discovery, geometry, seed):
    result = {}
    for factor in ("content", "binding", "order"):
        rank = discovery["rank_selections"][factor]["selection"]
        basis = geometry["orthogonal"]["forward"][factor]
        if rank is not None:
            if rank > basis.shape[1]:
                raise RuntimeError("Selected rank exceeds frozen factor basis")
            result[factor] = basis[:, :rank]
    return result


def deranged_controls(geometry, factor, rank):
    panel = geometry["panels"][factor]
    nulls = deranged_blocked_bases(
        panel["states"], panel["factor"], panel["nuisance"], panel["blocks"],
        count=CONTROL_COUNT, seed=CONTROL_SEEDS[factor])
    controls = []
    for null in nulls:
        raw = {name: geometry["geometry"][name]["basis"]
               for name in ("content", "binding", "order")}
        spectra = {name: geometry["geometry"][name]["eigenvalues"]
                   for name in ("content", "binding", "order")}
        raw[factor], spectra[factor] = null.basis, null.eigenvalues
        fitted = orthogonal_factor_geometry(
            raw["content"], raw["binding"], raw["order"], eigenvalues=spectra)
        basis = fitted.forward[factor]
        if basis.shape[1] < rank:
            raise RuntimeError("Deranged control rank fell below selected rank")
        controls.append({"basis": basis[:, :rank], "donor_blocks": null.donor_blocks})
    return tuple(controls)


def necessity_rows(net, groups, spec, basis, mean, capture, *, device, seed):
    rows, controls = [], []
    for assignment, name, assignment_seed in (("f0", "base", seed),
                                                ("f1", "donor", seed + 1)):
        episodes = tuple(episode_from_record(row) for group in groups for row in group[name])
        metadata = tuple({
            "logical_group_id": group["group_id"], "recipient": recipient,
            "target": episode.answer, "fixed_donor_answer": episode.answer,
            "base_render_id": episode.render_id, "donor_render_id": episode.render_id,
            "assignment": assignment,
            "item_id": f"content:necessity:{assignment}:{group['group_id']}:{recipient}",
        } for group in groups for recipient, episode in enumerate(
            tuple(episode_from_record(row) for row in group[name])))
        clean = raw_forward(net, episodes, device=device).logits
        pairs = mediator_state_pairs(net, episodes, episodes, spec, device=device)
        states = {
            "content_mean_ablation": mean_ablation_states(pairs.base, mean, basis),
            "content_equal_norm_corruption": equal_norm_component_corruption_states(
                pairs.base, mean, basis, seed=assignment_seed),
            "content_native_restoration": pairs.base,
        }
        controls.append({
            "assignment": assignment, "seed": assignment_seed,
            "item_ids": tuple(row["item_id"] for row in metadata),
            "base_states": pairs.base.detach().cpu(),
            "mean_ablation_states": states["content_mean_ablation"].detach().cpu(),
            "corruption_states": states["content_equal_norm_corruption"].detach().cpu(),
            "restoration_states": states["content_native_restoration"].detach().cpu(),
        })
        for condition, replacement in states.items():
            tagged_condition = f"{condition}_{assignment}"
            logits = mediator_endpoint_logits(net, episodes, spec, replacement, device=device)
            tagged = tuple({**row, "item_id": f"{row['item_id']}:{condition}"}
                           for row in metadata)
            rows.extend(diagnostic_rows(
                logits, clean, episodes, metadata=tagged, condition=tagged_condition,
                direction="necessity", logit_sink=capture))
    return rows, controls


def content_specificity_rows(net, groups, spec, basis, capture, *, device):
    rows = []
    for task, base_name, donor_name in (("first_hop", "first_hop_base", "first_hop_donor"),
                                        ("direct", "direct_base", "direct_donor")):
        bases = tuple(episode_from_record(group[base_name]) for group in groups)
        donors = tuple(episode_from_record(group[donor_name]) for group in groups)
        references = tuple(episode_from_record(group["donor"][0]) for group in groups)
        metadata = tuple({"logical_group_id": group["group_id"], "recipient": 0,
                          "target": donor.answer, "fixed_donor_answer": donor.answer,
                          "base_render_id": base.render_id, "donor_render_id": donor.render_id,
                          "item_id": f"content:{task}:{group['group_id']}:0"}
                         for group, base, donor in zip(groups, bases, donors, strict=True))
        clean = raw_forward(net, bases, device=device).logits
        edited = cross_task_mediator_logits(
            net, bases, donors, references, spec, basis=basis, device=device)
        rows.extend(diagnostic_rows(
            edited, clean, bases, metadata=metadata, condition=f"content_{task}",
            direction="specificity", logit_sink=capture))
    bases = tuple(episode_from_record(group["copy_control"]) for group in groups)
    donors = tuple(episode_from_record(group["donor"][0]) for group in groups)
    metadata = tuple({"logical_group_id": group["group_id"], "recipient": 0,
                      "target": base.answer, "fixed_donor_answer": donor.answer,
                      "base_render_id": base.render_id, "donor_render_id": donor.render_id,
                      "item_id": f"content:copy:{group['group_id']}:0"}
                     for group, base, donor in zip(groups, bases, donors, strict=True))
    clean = raw_forward(net, bases, device=device).logits
    edited = cross_task_mediator_logits(net, bases, donors, donors, spec, basis=basis, device=device)
    rows.extend(diagnostic_rows(
        edited, clean, bases, metadata=metadata, condition="content_copy",
        direction="specificity", logit_sink=capture))
    return rows


def binding_specificity_rows(net, groups, spec, basis, capture, *, device):
    """Check binding edits change the F lookup but preserve unrelated direct content."""
    rows, errors = [], []
    for direction, base_assignment, donor_assignment, reference_name, donor_name in (
            ("forward", "base", "donor", "binding_base", "binding_donor"),
            ("reverse", "donor", "base", "binding_donor", "binding_base")):
        for task in ("first_hop", "direct"):
            base_name = f"{task}_{base_assignment}"
            target_name = f"{task}_{donor_assignment}" if task == "first_hop" else base_name
            bases = tuple(episode_from_record(group[base_name]) for group in groups)
            targets = tuple(episode_from_record(group[target_name]) for group in groups)
            donors = tuple(episode_from_record(group[donor_name][0]) for group in groups)
            references = tuple(episode_from_record(group[reference_name][0]) for group in groups)
            metadata = tuple({
                "logical_group_id": group["group_id"], "recipient": 0,
                "target": target.answer, "fixed_donor_answer": donor.answer,
                "base_render_id": base.render_id, "donor_render_id": donor.render_id,
                "item_id": f"binding:{task}:{direction}:{group['group_id']}:0",
            } for group, base, target, donor in zip(
                groups, bases, targets, donors, strict=True))
            try:
                clean = raw_forward(net, bases, device=device).logits
                edited = cross_task_mediator_logits(
                    net, bases, donors, references, spec, basis=basis, device=device)
            except ValueError as exc:
                if "correspondence" not in str(exc):
                    raise
                errors.append({"task": task, "direction": direction,
                               "reason": str(exc)})
                continue
            rows.extend(diagnostic_rows(
                edited, clean, bases, metadata=metadata,
                condition=f"binding_{task}", direction=direction, logit_sink=capture))
    return rows, errors


def score_rows(rows, factor):
    selected = {}
    prefix = f"{factor}_selected_"
    for row in rows:
        if row["condition"].startswith(prefix):
            selected.setdefault((row["condition"], row["direction"]), []).append(row)
    primary = {f"{condition}:{direction}": summarize_confirmation_rows(cell)
               for (condition, direction), cell in selected.items()}
    controls = {}
    for family in ("haar", "deranged", "equal_energy"):
        cells = {}
        for row in rows:
            if row["condition"].startswith(f"{factor}_{family}_"):
                cells.setdefault((row["condition"], row["direction"]), []).append(row)
        controls[family] = {f"{condition}:{direction}": summarize_confirmation_rows(cell)
                            for (condition, direction), cell in cells.items()}
    return {"primary": primary, "controls": controls}


def cross_seed_content_rows(source_net, target_net, groups, spec, source_basis,
                            rotation, capture, *, source_seed, target_seed,
                            map_name, device, control_sink=None):
    """Transport a source-model content edit into the other model's residual gauge.

    The shared rotation is fitted only on discovery states.  Confirmation transports
    the finite selected-subspace *delta*, rather than an absolute source state, so the
    target model retains its own native base point and nuisance coordinates.
    """
    if source_seed == target_seed or map_name not in ("shared", "deranged"):
        raise ValueError("Cross-seed transport requires distinct seeds and a frozen map")
    width, rank = source_basis.shape
    if rotation.shape != (width, width) or rank <= 0:
        raise ValueError("Cross-seed rotation/basis shapes are incompatible")
    local_basis = source_basis.double()
    local_rotation = rotation.double()
    if not torch.allclose(local_rotation.T @ local_rotation,
                          torch.eye(width, dtype=torch.double), atol=1e-6, rtol=1e-6):
        raise ValueError("Cross-seed map is not orthogonal")

    rows = []
    batches = factor_episode_batches(
        groups, factor_name="content", condition_prefix="content_cross_seed",
        episode_loader=episode_from_record, surfaces=("marked",))
    for surface, direction, bases, donors, metadata in batches:
        source_pairs = mediator_state_pairs(
            source_net, bases, donors, spec, device=device)
        target_pairs = mediator_state_pairs(
            target_net, bases, bases, spec, device=device)
        source_base = source_pairs.base.detach().cpu().double()
        source_donor = source_pairs.donor.detach().cpu().double()
        target_base = target_pairs.base.detach().cpu().double()
        source_delta = source_donor - source_base
        selected_delta = (source_delta @ local_basis) @ local_basis.T
        replacement = target_base + selected_delta @ local_rotation
        if control_sink is not None:
            control_sink({
                "condition": f"content_cross_seed_{map_name}_{surface}",
                "direction": f"{source_seed}_to_{target_seed}_{direction}",
                "source_seed": source_seed, "target_seed": target_seed,
                "map_name": map_name,
                "item_ids": tuple(
                    f"content:cross_seed:{map_name}:{source_seed}_to_{target_seed}:"
                    f"{direction}:{row['logical_group_id']}:{row['recipient']}"
                    for row in metadata),
                "source_base_states": source_base.to(source_pairs.base.dtype),
                "source_donor_states": source_donor.to(source_pairs.donor.dtype),
                "target_base_states": target_base.to(target_pairs.base.dtype),
                "replacement_states": replacement.detach().cpu(),
            })
        clean = raw_forward(target_net, bases, device=device).logits
        edited = mediator_endpoint_logits(
            target_net, bases, spec, replacement.to(target_pairs.base.dtype), device=device)
        condition = f"content_cross_seed_{map_name}_{surface}"
        mapped_metadata = tuple({
            **row,
            "item_id": (f"content:cross_seed:{map_name}:{source_seed}_to_{target_seed}:"
                        f"{direction}:{row['logical_group_id']}:{row['recipient']}"),
            "source_seed": source_seed,
            "target_seed": target_seed,
        } for row in metadata)
        rows.extend(diagnostic_rows(
            edited, clean, bases, metadata=mapped_metadata, condition=condition,
            direction=f"{source_seed}_to_{target_seed}_{direction}",
            logit_sink=capture))
    return rows


def score_cross_seed_rows(rows):
    cells = {}
    for row in rows:
        if row["condition"].startswith("content_cross_seed_"):
            cells.setdefault((row["condition"], row["direction"]), []).append(row)
    return {f"{condition}:{direction}": summarize_confirmation_rows(cell)
            for (condition, direction), cell in sorted(cells.items())}


def order_shortcut_rows(net, groups, spec, bases, capture, *, device):
    """Test selected order/binding components against a frozen physical-slot oracle."""
    oracle_kind = registered_physical_order_oracle_kind(
        mediator_kind=spec.kind,
        **({"label": spec.label} if spec.kind == "single"
           else {"source_label": spec.source_label}))
    component_bases = {name: bases[name] for name in ("order", "binding") if name in bases}
    rows, records, state_batches = [], [], []
    for component, basis in component_bases.items():
        for surface, assignments in ORDER_SHORTCUT_SURFACES.items():
            for assignment, base_name, donor_name in assignments:
                episodes, donors, metadata, cell_oracles = [], [], [], []
                for group in groups:
                    base_rows = tuple(episode_from_record(row) for row in group[base_name])
                    donor_rows = tuple(episode_from_record(row) for row in group[donor_name])
                    for recipient, (base, donor) in enumerate(zip(
                            base_rows, donor_rows, strict=True)):
                        oracle = physical_order_shortcut_oracle(base, donor)
                        fields = oracle.compact_fields(oracle_kind)
                        item_id = (f"order:physical:{component}:{surface}:{assignment}:"
                                   f"{group['group_id']}:{recipient}")
                        record = {
                            "condition": f"physical_order_{component}_{surface}_{assignment}",
                            "direction": "forward", "component": component,
                            "surface": surface, "assignment": assignment,
                            "logical_group_id": group["group_id"],
                            "recipient": recipient, "item_id": item_id,
                            "semantic_target": base.answer,
                            "base_render_id": base.render_id,
                            "donor_render_id": donor.render_id, **fields,
                        }
                        records.append(record)
                        cell_oracles.append(oracle)
                        if fields["oracle_valid"]:
                            episodes.append(base)
                            donors.append(donor)
                            metadata.append({
                                "logical_group_id": group["group_id"],
                                "recipient": recipient, "target": fields["oracle_target"],
                                "fixed_donor_answer": donor.answer,
                                "base_render_id": base.render_id,
                                "donor_render_id": donor.render_id,
                                "item_id": item_id, **fields,
                            })
                denominator = physical_order_shortcut_denominators(
                    cell_oracles, kind=oracle_kind,
                    group_ids=[record["logical_group_id"]
                               for record in records[-len(cell_oracles):]])
                for record in records[-len(cell_oracles):]:
                    record["cell_denominator"] = denominator
                if not episodes:
                    continue
                episode_tuple, donor_tuple = tuple(episodes), tuple(donors)
                clean = raw_forward(net, episode_tuple, device=device).logits
                pairs = mediator_state_pairs(
                    net, episode_tuple, donor_tuple, spec, device=device)
                local_basis = basis.detach().cpu().double()
                local_base = pairs.base.detach().cpu().double()
                local_donor = pairs.donor.detach().cpu().double()
                delta = local_donor - local_base
                replacement = local_base + (delta @ local_basis) @ local_basis.T
                state_batches.append({
                    "condition": f"physical_order_{component}_{surface}_{assignment}",
                    "direction": "forward", "component": component,
                    "item_ids": tuple(row["item_id"] for row in metadata),
                    "base_states": pairs.base.detach().cpu(),
                    "donor_states": pairs.donor.detach().cpu(),
                    "replacement_states": replacement.detach().cpu(),
                })
                edited = mediator_endpoint_logits(
                    net, episode_tuple, spec, replacement.to(pairs.base.dtype), device=device)
                rows.extend(diagnostic_rows(
                    edited, clean, episode_tuple, metadata=tuple(metadata),
                    condition=f"physical_order_{component}_{surface}_{assignment}",
                    direction="forward", logit_sink=capture))
    return rows, records, state_batches


def score_order_shortcut(rows, records, component):
    """Score physical targets while keeping all exclusions and coverage explicit."""
    selected_records = [row for row in records if row["component"] == component]
    selected_rows = {row["item_id"]: row for row in rows
                     if row["condition"].startswith(f"physical_order_{component}_")}
    eligible = [row for row in selected_records if row["oracle_valid"]]
    if set(selected_rows) != {row["item_id"] for row in eligible}:
        raise RuntimeError("Physical-order compact rows do not match the frozen denominator")

    def summarize(record_cell):
        valid = [row for row in record_cell if row["oracle_valid"]]
        correct = sum(selected_rows[row["item_id"]]["prediction"] == row["oracle_target"]
                      for row in valid)
        grouped = {}
        for row in record_cell:
            grouped.setdefault((row["condition"], row["logical_group_id"]), []).append(row)
        eligible_groups = [group for group in grouped.values()
                           if len(group) == 3 and all(row["oracle_valid"] for row in group)]
        exact = sum(all(selected_rows[row["item_id"]]["prediction"]
                        == row["oracle_target"] for row in group)
                    for group in eligible_groups)
        return {
            "registered_items": len(record_cell), "eligible_items": len(valid),
            "eligible_item_coverage": len(valid) / len(record_cell),
            "correct_items": correct,
            "item_accuracy": None if not valid else correct / len(valid),
            "registered_groups": len(grouped), "eligible_groups": len(eligible_groups),
            "eligible_group_coverage": len(eligible_groups) / len(grouped),
            "exact_groups": exact,
            "group_accuracy": None if not eligible_groups else exact / len(eligible_groups),
            "semantic_collisions": sum(row["oracle_semantic_collision"]
                                       for row in record_cell),
            "structurally_invalid": sum(not row["oracle_structurally_valid"]
                                        for row in record_cell),
        }

    cells = {}
    for record in selected_records:
        key = f"{record['surface']}:{record['assignment']}"
        cells.setdefault(key, []).append(record)
    return {"oracle_kind": selected_records[0]["oracle_kind"],
            "coverage_threshold": .50, "overall": summarize(selected_records),
            "cells": {key: summarize(cell) for key, cell in sorted(cells.items())}}


def _row_cells(rows, prefix):
    cells = {}
    for row in rows:
        if row["condition"].startswith(prefix):
            cells.setdefault((row["condition"], row["direction"]), []).append(row)
    return cells


def _single_item_summary(rows):
    if not rows or len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError("Single-item confirmation cells require unique nonempty item IDs")
    correct = sum(row["prediction"] == row["target"] for row in rows)
    clean = sum(row["clean_prediction"] == row["target"] for row in rows)
    return {"items": len(rows), "correct": correct, "item_accuracy": correct / len(rows),
            "clean_correct": clean, "clean_accuracy": clean / len(rows),
            "accuracy_loss": (clean - correct) / len(rows)}


def _factor_decision(rows, factor, *, item_threshold, group_threshold):
    selected = _row_cells(rows, f"{factor}_selected_")
    complement = _row_cells(rows, f"{factor}_complement_")
    primary = {f"{condition}:{direction}": summarize_confirmation_rows(cell)
               for (condition, direction), cell in sorted(selected.items())}
    complement_scores = {}
    for key, cell in sorted(complement.items()):
        complement_scores[f"{key[0]}:{key[1]}"] = {
            "items": len(cell),
            "base_preservation": sum(row["prediction"] == row["base_answer"]
                                     for row in cell) / len(cell),
            "donor_target_rate": sum(row["prediction"] == row["target"]
                                     for row in cell) / len(cell),
        }
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
        primary_accuracy = summarize_confirmation_rows(selected[primary_key])["item_accuracy"]
        control_scores = {}
        for family in ("haar", "deranged", "equal_energy"):
            family_cells = _row_cells(rows, f"{factor}_{family}_")
            matches = [summarize_confirmation_rows(cell)["item_accuracy"]
                       for (condition, local_direction), cell in family_cells.items()
                       if condition.endswith("_marked") and local_direction == direction]
            if matches:
                control_scores[family] = max(matches)
        if factor == "binding":
            order_cell = _row_cells(rows, "binding_order_control_marked").get(
                ("binding_order_control_marked", direction))
            if order_cell:
                control_scores["order"] = summarize_confirmation_rows(order_cell)[
                    "item_accuracy"]
        best = max(control_scores.values()) if control_scores else None
        required_control_families = ({"haar", "deranged", "equal_energy", "order"}
                                     if factor == "binding"
                                     else {"haar", "deranged", "equal_energy"})
        advantages[direction] = {
            "primary_item_accuracy": primary_accuracy, "controls": control_scores,
            "best_control_item_accuracy": best,
            "advantage": None if best is None else primary_accuracy - best,
            "pass": (set(control_scores) == required_control_families
                     and best is not None and primary_accuracy - best >= .35),
        }
    control_pass = set(advantages) == {"forward", "reverse"} \
        and all(cell["pass"] for cell in advantages.values())
    return {"primary": primary, "complement": complement_scores,
            "control_advantages": advantages, "transfer_pass": transfer_pass,
            "complement_pass": complement_pass, "control_pass": control_pass}


def decide_confirmation(rows_by_seed, controls_by_seed, discovery):
    """Compute every preregistered gate; final labels remain audit-contingent."""
    gates = {"content": {}, "binding": {}, "cross_seed_content": {}, "order": {}}
    for seed, rows in rows_by_seed.items():
        if discovery["rank_selections"]["content"]["selection"] is not None:
            content = _factor_decision(
                rows, "content", item_threshold=.80, group_threshold=.60)
            necessity = {}
            for assignment in ("f0", "f1"):
                corruption = [row for row in rows if row["condition"]
                              == f"content_equal_norm_corruption_{assignment}"]
                restoration = [row for row in rows if row["condition"]
                               == f"content_native_restoration_{assignment}"]
                if corruption and restoration:
                    corrupt_score = _single_item_summary(corruption)
                    restore_score = _single_item_summary(restoration)
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
                task_rows = [row for row in rows if row["condition"] == f"content_{task}"]
                if task_rows:
                    specificity[task] = _single_item_summary(task_rows)
            specificity_pass = (set(specificity) == {"first_hop", "direct", "copy"}
                                and specificity["first_hop"]["item_accuracy"] >= .80
                                and specificity["direct"]["item_accuracy"] >= .80
                                and specificity["copy"]["accuracy_loss"] <= .05)
            necessity_pass = (set(necessity) == {"f0", "f1"}
                              and all(cell["pass_without_exact_logit_audit"]
                                      for cell in necessity.values()))
            content.update({"necessity": necessity, "specificity": specificity,
                            "necessity_pass_without_exact_logit_audit": necessity_pass,
                            "specificity_pass": specificity_pass,
                            "candidate_pass_without_artifact_audit": (
                                content["transfer_pass"] and content["complement_pass"]
                                and content["control_pass"] and necessity_pass
                                and specificity_pass)})
            gates["content"][seed] = content

        cross = score_cross_seed_rows(rows)
        shared = {key: value for key, value in cross.items()
                  if "content_cross_seed_shared_marked" in key}
        deranged = {key: value for key, value in cross.items()
                    if "content_cross_seed_deranged_marked" in key}
        cross_cells = {}
        for key, score in shared.items():
            control_key = key.replace("_shared_", "_deranged_")
            control = deranged.get(control_key)
            cross_cells[key] = {
                "shared": score, "deranged": control,
                "advantage": None if control is None else (
                    score["item_accuracy"] - control["item_accuracy"]),
                "pass": (control is not None and score["item_accuracy"] >= .80
                         and score["group_accuracy"] >= .60
                         and score["non_injection"] >= .90
                         and score["item_accuracy"] - control["item_accuracy"] >= .35),
            }
        gates["cross_seed_content"][seed] = {
            "cells": cross_cells,
            "candidate_pass_without_artifact_audit": (
                len(cross_cells) == 2 and all(cell["pass"] for cell in cross_cells.values())),
        }

        if discovery["rank_selections"]["binding"]["selection"] is not None:
            binding = _factor_decision(
                rows, "binding", item_threshold=.75, group_threshold=.55)
            specificity = {}
            for task in ("first_hop", "direct"):
                for direction in ("forward", "reverse"):
                    task_rows = [row for row in rows if row["condition"] == f"binding_{task}"
                                 and row["direction"] == direction]
                    if task_rows:
                        specificity[f"{task}:{direction}"] = _single_item_summary(task_rows)
            errors = controls_by_seed[seed].get("binding_specificity_errors", [])
            specificity_pass = (len(specificity) == 4 and not errors
                                and all(cell["item_accuracy"] >= .90
                                        for cell in specificity.values()))
            binding.update({"specificity": specificity, "specificity_errors": errors,
                            "specificity_pass": specificity_pass,
                            "candidate_pass_without_artifact_audit": (
                                binding["transfer_pass"] and binding["control_pass"]
                                and specificity_pass)})
            gates["binding"][seed] = binding

        if discovery["rank_selections"]["order"]["selection"] is not None:
            order = score_order_shortcut(
                rows, controls_by_seed[seed]["physical_order_oracles"], "order")
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
    order_coverage = len(gates["order"]) == 2 and all(
        row["coverage_pass"] for row in gates["order"].values())
    order_accuracies = [row["overall"]["item_accuracy"] for row in gates["order"].values()]
    binding_measured = len(gates["binding"]) == 2
    order_supported = (order_coverage and all(value is not None and value >= .75
                                              for value in order_accuracies)
                       and binding_measured and not binding_pass)
    binding_reordered = len(gates["binding"]) == 2 and all(
        len([key for key in row["primary"] if "_reordered:" in key]) == 2
        and all(cell["item_accuracy"] >= .75 and cell["group_accuracy"] >= .55
                for key, cell in row["primary"].items() if "_reordered:" in key)
        for row in gates["binding"].values())
    order_insufficient = (order_coverage and all(
        value is not None and value <= .20 for value in order_accuracies)
        and binding_reordered)
    candidate_labels = {
        "content": ("CONTENT-SUBSPACE-SUPPORTED" if content_pass
                    else "CONTENT-SUBSPACE-NOT-SUPPORTED"),
        "cross_seed_content": ("CROSS-SEED-CAUSAL-TRANSPORT-SUPPORTED" if cross_pass
                               else "CROSS-SEED-CAUSAL-TRANSPORT-NOT-SUPPORTED"),
        "binding": ("BINDING-SUBSPACE-SUPPORTED" if binding_pass
                    else "BINDING-SUBSPACE-NOT-SUPPORTED"),
        "order": ("ORDER-SHORTCUT-SUPPORTED" if order_supported else
                  "ORDER COMPONENT NOT CAUSALLY SUFFICIENT" if order_insufficient else
                  "INCONCLUSIVE: PHYSICAL-ORDER COVERAGE" if gates["order"]
                  and not order_coverage else
                  "INCONCLUSIVE: BINDING COMPARATOR" if gates["order"]
                  and not binding_measured else "ORDER-SHORTCUT-NOT-SUPPORTED"),
    }
    return {"status": "candidate_decisions_pending_independent_artifact_audit",
            "thresholds_frozen_before_confirmation": True,
            "exact_native_restoration_logit_error_required": "<1e-6",
            "gates": gates, "candidate_labels": candidate_labels}


def run_seed(net, groups, spec, discovery, geometry, seed, capture, *, device,
             probe=lambda: None):
    bases = selected_bases(discovery, geometry, seed)
    rows, controls = [], {}
    for factor, basis in bases.items():
        rows.extend(factor_transfer_rows(
            net, groups, spec, basis, factor_name=factor,
            condition_prefix=f"{factor}_selected", episode_loader=episode_from_record,
            device=device, logit_sink=capture))
        probe()
        rows.extend(factor_transfer_rows(
            net, groups, spec, basis, factor_name=factor,
            condition_prefix=f"{factor}_complement", component="complement",
            episode_loader=episode_from_record, device=device, logit_sink=capture))
        rank = basis.shape[1]
        haar = haar_random_bases(basis.shape[0], rank, CONTROL_COUNT,
                                 seed=CONTROL_SEEDS[factor])
        deranged = deranged_controls(geometry, factor, rank)
        controls[factor] = {"haar": haar, "deranged": deranged, "equal_energy": []}
        for index, control in enumerate(haar):
            rows.extend(factor_transfer_rows(
                net, groups, spec, control, factor_name=factor,
                condition_prefix=f"{factor}_haar_{index:02d}",
                surfaces=CONTROL_SURFACES[factor], episode_loader=episode_from_record,
                device=device, logit_sink=capture))
            probe()
        for index, control in enumerate(deranged):
            rows.extend(factor_transfer_rows(
                net, groups, spec, control["basis"], factor_name=factor,
                condition_prefix=f"{factor}_deranged_{index:02d}",
                surfaces=CONTROL_SURFACES[factor], episode_loader=episode_from_record,
                device=device, logit_sink=capture))
            probe()
        for index in range(CONTROL_COUNT):
            rows.extend(factor_equal_energy_rows(
                net, groups, spec, basis, factor_name=factor,
                condition_prefix=f"{factor}_equal_energy_{index:02d}",
                surfaces=CONTROL_SURFACES[factor], episode_loader=episode_from_record,
                seed=CONTROL_SEEDS[factor] + 1000 + 20 * index,
                device=device, logit_sink=capture,
                control_sink=controls[factor]["equal_energy"].append))
            probe()
    if "content" in bases and "binding" in bases:
        union = orthonormal_union(bases["content"], bases["binding"])
        for factor in ("content", "binding"):
            rows.extend(factor_transfer_rows(
                net, groups, spec, union, factor_name=factor,
                condition_prefix=f"content_binding_on_{factor}",
                episode_loader=episode_from_record, device=device, logit_sink=capture))
    if "binding" in bases and "order" in bases:
        rows.extend(factor_transfer_rows(
            net, groups, spec, bases["order"], factor_name="binding",
            condition_prefix="binding_order_control", surfaces=("marked",),
            episode_loader=episode_from_record, device=device, logit_sink=capture))
    if set(bases) == {"content", "binding", "order"}:
        union = orthonormal_union(bases["content"], bases["binding"], bases["order"])
        for factor in ("content", "binding", "order"):
            rows.extend(factor_transfer_rows(
                net, groups, spec, union, factor_name=factor,
                condition_prefix=f"all_factors_on_{factor}",
                episode_loader=episode_from_record, device=device, logit_sink=capture))
    content = bases["content"]
    necessity, necessity_controls = necessity_rows(
        net, groups, spec, content, geometry["geometry"]["content"]["mean"], capture,
        device=device, seed=CONTROL_SEEDS["content"] + 5000)
    rows.extend(necessity)
    controls["necessity"] = necessity_controls
    rows.extend(content_specificity_rows(net, groups, spec, content, capture, device=device))
    if "binding" in bases:
        binding_rows, binding_errors = binding_specificity_rows(
            net, groups, spec, bases["binding"], capture, device=device)
        rows.extend(binding_rows)
        controls["binding_specificity_errors"] = binding_errors
    shortcut_rows, shortcut_records, shortcut_state_batches = order_shortcut_rows(
        net, groups, spec, bases, capture, device=device)
    rows.extend(shortcut_rows)
    controls["physical_order_oracles"] = shortcut_records
    controls["physical_order_state_batches"] = shortcut_state_batches
    controls["selected"] = bases
    return rows, controls


def confirmation(result_dir: Path, output_dir: Path, device: str):
    report_path = result_dir / "subspace-confirmation.json"
    if report_path.exists():
        raise FileExistsError("No automatic Stage-D confirmation overwrite")
    stage_c = _load_stage_c()
    manifest = stage_c.load_manifest(result_dir)
    provenance = stage_c.verify_frozen_source(manifest)
    discovery, discovery_audit, spec = prerequisite(result_dir, stage_c)
    suite = stage_c.load_confirmation_suite(output_dir, manifest)
    groups = suite["splits"]["confirmation"]
    start, traffic = time.monotonic(), {"bytes": 0}
    peak = {"current": 0, "driver": 0}
    artifacts, scores = {}, {}

    def count(value):
        traffic["bytes"] += int(value)

    def probe():
        if device == "mps":
            peak["current"] = max(peak["current"], int(torch.mps.current_allocated_memory()))
            peak["driver"] = max(peak["driver"], int(torch.mps.driver_allocated_memory()))
        elapsed = time.monotonic() - start
        if discovery["cumulative_stage_c_e_seconds"] + elapsed > MAX_STAGE_C_E_SECONDS \
                or discovery["cumulative_campaign_seconds"] + elapsed > MAX_CAMPAIGN_SECONDS \
                or peak["current"] > MAX_CURRENT_BYTES \
                or discovery["cumulative_materialized_bytes"] + traffic["bytes"] \
                > MAX_TRAFFIC_BYTES:
            raise RuntimeError("Stage-D confirmation resource ceiling exceeded")

    nets, geometries, captures, rows_by_seed, controls_by_seed = {}, {}, {}, {}, {}
    for seed in ("0", "1"):
        net, metadata = load_raw_checkpoint(stage_c.checkpoint_path(manifest, seed), device=device)
        if metadata["arm"] != "raw_deep" or metadata["step"] != 8000:
            raise RuntimeError("Stage-D confirmation requires final raw-deep checkpoints")
        geometry = load_geometry(discovery, seed, stage_c)
        capture = ShardedLogitCapture(
            output_dir / f"subspace-confirmation-logits-rep{seed}", stage_c)
        with materialized_tensor_counter(count):
            rows, controls = run_seed(
                net, groups, spec, discovery, geometry, seed, capture,
                device=device, probe=probe)
        nets[seed], geometries[seed], captures[seed] = net, geometry, capture
        rows_by_seed[seed], controls_by_seed[seed] = rows, controls
        probe()

    cross_seed = load_cross_seed_geometry(discovery, stage_c)["alignment"]
    shared_rotation = cross_seed["shared"]["rotation"]
    deranged_rotation = cross_seed["shared_deranged_control"]["rotation"]
    for source_seed, target_seed, transpose in (("0", "1", False), ("1", "0", True)):
        source_basis = selected_bases(
            discovery, geometries[source_seed], source_seed)["content"]
        rotations = {
            "shared": shared_rotation.T if transpose else shared_rotation,
            "deranged": deranged_rotation.T if transpose else deranged_rotation,
        }
        transport_controls = []
        with materialized_tensor_counter(count):
            for map_name, rotation in rotations.items():
                rows_by_seed[target_seed].extend(cross_seed_content_rows(
                    nets[source_seed], nets[target_seed], groups, spec, source_basis,
                    rotation, captures[target_seed], source_seed=source_seed,
                    target_seed=target_seed, map_name=map_name, device=device,
                    control_sink=transport_controls.append))
                probe()
        controls_by_seed[target_seed]["cross_seed"] = {
            "source_seed": source_seed,
            "target_seed": target_seed,
            "source_content_basis": source_basis,
            "shared_rotation": rotations["shared"],
            "deranged_rotation": rotations["deranged"],
            "transport_batches": transport_controls,
        }

    for seed in ("0", "1"):
        rows, controls, capture = (
            rows_by_seed[seed], controls_by_seed[seed], captures[seed])
        compact = result_dir / f"subspace-confirmation-rows-rep{seed}.jsonl.gz"
        controls_path = output_dir / f"subspace-confirmation-controls-rep{seed}.pt"
        if compact.exists() or controls_path.exists():
            raise FileExistsError("No automatic Stage-D compact/control overwrite")
        torch.save({"format": "TEACH-0013-subspace-controls-v1",
                    "controls": controls}, controls_path)
        scores[seed] = {factor: score_rows(rows, factor)
                        for factor in ("content", "binding", "order")
                        if discovery["rank_selections"][factor]["selection"] is not None}
        scores[seed]["cross_seed_content"] = score_cross_seed_rows(rows)
        for component in ("order", "binding"):
            if component in controls["selected"]:
                scores[seed][f"physical_order_{component}"] = score_order_shortcut(
                    rows, controls["physical_order_oracles"], component)
        artifacts[seed] = {
            "rows": {"path": str(compact.relative_to(ROOT)),
                     "sha256": stage_c.write_rows(compact, rows),
                     "bytes": compact.stat().st_size, "rows": len(rows)},
            "exact_symbol_logits": capture.metadata(),
            "controls": {"path": str(controls_path.relative_to(ROOT)),
                         "sha256": stage_c.sha_file(controls_path),
                         "bytes": controls_path.stat().st_size},
        }
        probe()
    if device == "mps":
        torch.mps.synchronize()
        peak["current"] = max(peak["current"], int(torch.mps.current_allocated_memory()))
        peak["driver"] = max(peak["driver"], int(torch.mps.driver_allocated_memory()))
        nets.clear()
        torch.mps.empty_cache()
    elapsed = time.monotonic() - start
    cumulative_time = discovery["cumulative_stage_c_e_seconds"] + elapsed
    cumulative_campaign = discovery["cumulative_campaign_seconds"] + elapsed
    cumulative_traffic = discovery["cumulative_materialized_bytes"] + traffic["bytes"]
    if cumulative_time > MAX_STAGE_C_E_SECONDS or cumulative_campaign > MAX_CAMPAIGN_SECONDS \
            or peak["current"] > MAX_CURRENT_BYTES or cumulative_traffic > MAX_TRAFFIC_BYTES:
        raise RuntimeError("Stage-D confirmation resource ceiling exceeded")
    if stage_c.tree_bytes(output_dir) > MAX_OUTPUT_BYTES \
            or stage_c.tree_bytes(result_dir) > MAX_RESULT_BYTES:
        raise RuntimeError("Stage-D confirmation artifact ceiling exceeded")
    decisions = decide_confirmation(rows_by_seed, controls_by_seed, discovery)
    report = {
        "experiment": "TEACH-0013", "mode": "subspace_confirmation",
        "status": "stage_d_confirmation_complete_pending_audit",
        "mediator": asdict(spec),
        "subspace_discovery_sha256": stage_c.sha_file(
            result_dir / "subspace-discovery.json"),
        "subspace_discovery_audit_sha256": stage_c.sha_file(
            result_dir / "subspace-discovery-audit.json"),
        "subspace_discovery_audit": discovery_audit["audit"],
        "rank_selections": discovery["rank_selections"], "scores": scores,
        "decisions": decisions,
        "artifacts": artifacts, "elapsed_seconds": elapsed,
        "materialized_activation_bytes": traffic["bytes"],
        "cumulative_stage_c_e_seconds": cumulative_time,
        "cumulative_campaign_seconds": cumulative_campaign,
        "cumulative_materialized_bytes": cumulative_traffic,
        "peak_sampled_current_allocated_bytes": peak["current"],
        "peak_sampled_driver_allocated_bytes": peak["driver"],
        "ignored_output_bytes": stage_c.tree_bytes(output_dir),
        "tracked_result_bytes": stage_c.tree_bytes(result_dir),
        "output_directory": str(output_dir.resolve().relative_to(ROOT.resolve())),
        "suite_gzip_sha256": manifest["suite_gzip_sha256"], **provenance,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=ROOT / "results/TEACH-0013")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/TEACH-0013")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    print(json.dumps(confirmation(args.result_dir, args.output_dir, args.device),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
