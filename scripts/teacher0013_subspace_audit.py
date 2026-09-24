#!/usr/bin/env python3
"""Independent no-model audit of TEACH-0013 Stage-D discovery geometry."""

import argparse
from collections import defaultdict
import importlib.util
import json
import math
from pathlib import Path
import random
import subprocess

import torch


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_RANKS = (1, 2, 4, 8, 16, 32, 64)
MINIMUM_SPECTRAL_GAP = .05
THRESHOLDS = {"content": (.80, .60), "binding": (.75, .55), "order": (.75, .55)}
STAGE_C_E_SECONDS = 5400.0
CAMPAIGN_SECONDS = 14_400.0
MAX_CURRENT_BYTES = 24 * 1024**3
MAX_TRAFFIC_BYTES = 300 * 1024**3
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


class AuditError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise AuditError(message)


def file_digest(path: Path) -> str:
    import hashlib
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _load_confirmation_auditor():
    spec = importlib.util.spec_from_file_location(
        "teacher0013_confirmation_audit_dependency",
        ROOT / "scripts/teacher0013_confirmation_audit.py")
    if spec is None or spec.loader is None:
        raise AuditError("Could not load independent exact-logit verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    return module


def load_rows(path: Path):
    import gzip
    with gzip.open(path, "rt") as source:
        return [json.loads(line) for line in source if line.strip()]


def participation_ratio(values):
    values = values.double().clamp_min(0)
    denominator = values.square().sum()
    return 0.0 if denominator == 0 else float(values.sum().square() / denominator)


def blocked_geometry(panel):
    states = panel["states"].double()
    factor, nuisance, blocks = (tuple(panel[name])
                                for name in ("factor", "nuisance", "blocks"))
    need(states.ndim == 2 and states.shape[0] == len(factor) == len(nuisance)
         == len(blocks) and torch.isfinite(states).all().item(),
         "Malformed factor-state panel")
    factors = tuple(sorted(set(factor), key=repr))
    nuisances = tuple(sorted(set(nuisance), key=repr))
    block_levels = tuple(sorted(set(blocks), key=repr))
    contrasts, means = [], []
    for block in block_levels:
        marginalized = []
        for level in factors:
            cells = []
            for nuisance_cell in nuisances:
                selected = [index for index, labels in enumerate(zip(
                    factor, nuisance, blocks, strict=True))
                            if labels == (level, nuisance_cell, block)]
                need(selected, "Incomplete blocked factor-by-nuisance cell")
                cells.append(states[selected].mean(0))
            marginalized.append(torch.stack(cells).mean(0))
        marginalized = torch.stack(marginalized)
        mean = marginalized.mean(0)
        means.append(mean)
        contrasts.extend(marginalized - mean)
    contrasts = torch.stack(contrasts)
    covariance = contrasts.T @ contrasts / contrasts.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = eigenvalues.argsort(descending=True)
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    scale = max(float(eigenvalues[0].abs()), 1.0)
    rank = min(int((eigenvalues > 1e-10 * scale).sum()), 64)
    return {"mean": torch.stack(means).mean(0), "basis": eigenvectors[:, :rank],
            "eigenvalues": eigenvalues, "factor_levels": factors,
            "nuisance_cells": nuisances,
            "participation_ratio": participation_ratio(eigenvalues)}


def orthogonalize(basis, eigenvalues, against):
    basis, against = basis.double(), against.double()
    weighted = basis * eigenvalues.double()[:basis.shape[1]].clamp_min(0).sqrt().unsqueeze(0)
    remainder = weighted - against @ (against.T @ weighted) \
        if against.shape[1] else weighted
    if remainder.shape[1] == 0:
        return remainder, torch.empty(0, dtype=torch.double)
    u, singular, _ = torch.linalg.svd(remainder, full_matrices=False)
    keep = singular > 1e-8 * max(float(singular[0]), 1.0)
    return u[:, keep], singular[keep].square()


def principal_angles(left, right):
    if not left.shape[1] or not right.shape[1]:
        return torch.empty(0, dtype=torch.double)
    return torch.linalg.svdvals(left.double().T @ right.double()).clamp(0, 1).acos()


def _label_position(group, variant, recipient, label):
    labels = group["semantic_layouts"][variant][recipient]["labels"]
    positions = [index for index, candidate in enumerate(labels) if candidate == label]
    need(len(positions) == 1, "Suite endpoint label is not unique")
    return positions[0]


def verify_panel_binding(panel, factor_name, groups, mediator):
    expected_metadata, expected_factor, expected_nuisance, expected_blocks = [], [], [], []
    endpoint_label = (mediator["label"] if mediator["kind"] == "single"
                      else mediator["destination_label"])
    source_label = (mediator["label"] if mediator["kind"] == "single"
                    else mediator["source_label"])
    for group in groups:
        for nuisance_name, base_variant, donor_variant in FACTOR_PAIRS[factor_name]:
            for recipient in range(3):
                base, donor = group[base_variant][recipient], group[donor_variant][recipient]
                item = {
                    "logical_group_id": group["group_id"],
                    "nuisance": (nuisance_name, recipient), "recipient": recipient,
                    "base_variant": base_variant, "donor_variant": donor_variant,
                    "base_logical_id": base["logical_id"],
                    "donor_logical_id": donor["logical_id"],
                    "base_render_id": base["render_id"],
                    "donor_render_id": donor["render_id"],
                    "endpoint_label": endpoint_label,
                    "base_endpoint_position": _label_position(
                        group, base_variant, recipient, endpoint_label),
                    "donor_endpoint_position": _label_position(
                        group, donor_variant if mediator["kind"] == "single" else base_variant,
                        recipient, endpoint_label),
                    "source_label": source_label,
                    "donor_source_position": _label_position(
                        group, donor_variant, recipient, source_label),
                }
                for level, state_source in ((0, "native_base"), (1, "donor_mediated")):
                    expected_metadata.append({**item, "factor": factor_name, "level": level,
                                              "state_source": state_source})
                    expected_factor.append(level)
                    expected_nuisance.append((nuisance_name, recipient))
                    expected_blocks.append(group["group_id"])
    need(tuple(panel["metadata"]) == tuple(expected_metadata)
         and tuple(panel["factor"]) == tuple(expected_factor)
         and tuple(panel["nuisance"]) == tuple(expected_nuisance)
         and tuple(panel["blocks"]) == tuple(expected_blocks),
         f"{factor_name} state panel is not bound to the frozen discovery suite")


def verify_geometry_artifact(metadata: dict, groups, mediator) -> tuple[dict, dict]:
    path = ROOT / metadata["path"]
    need(path.is_file() and path.stat().st_size == metadata["bytes"]
         and file_digest(path) == metadata["sha256"], "Geometry artifact mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    need(payload.get("format") == "TEACH-0013-factor-geometry-v2"
         and set(payload.get("panels", {})) == {"content", "binding", "order"},
         "Unknown factor-geometry format")
    recomputed = {name: blocked_geometry(payload["panels"][name])
                  for name in ("content", "binding", "order")}
    for name in ("content", "binding", "order"):
        verify_panel_binding(payload["panels"][name], name, groups, mediator)
    for name, expected in recomputed.items():
        stored = payload["geometry"][name]
        need(torch.allclose(stored["mean"].double(), expected["mean"], atol=1e-8)
             and torch.allclose(stored["eigenvalues"].double(), expected["eigenvalues"],
                                atol=1e-8, rtol=1e-7)
             and stored["factor_levels"] == expected["factor_levels"]
             and stored["nuisance_cells"] == expected["nuisance_cells"]
             and math.isclose(stored["participation_ratio"],
                              expected["participation_ratio"], abs_tol=1e-10),
             f"{name} geometry summary does not recompute")
        left = stored["basis"].double()
        right = expected["basis"]
        need(left.shape == right.shape and torch.allclose(
            left @ left.T, right @ right.T, atol=1e-7, rtol=1e-7),
            f"{name} eigenspace does not recompute")

    raw = {name: recomputed[name]["basis"] for name in ("content", "binding", "order")}
    spectra = {name: recomputed[name]["eigenvalues"] for name in raw}
    forward, forward_spectra = {}, {}
    assigned = torch.empty(raw["content"].shape[0], 0, dtype=torch.double)
    for name in ("content", "binding", "order"):
        forward[name], forward_spectra[name] = orthogonalize(
            raw[name], spectra[name], assigned)
        assigned = torch.cat((assigned, forward[name]), 1)
    reverse, reverse_spectra = {}, {}
    assigned = torch.empty(raw["content"].shape[0], 0, dtype=torch.double)
    for name in ("order", "binding", "content"):
        reverse[name], reverse_spectra[name] = orthogonalize(
            raw[name], spectra[name], assigned)
        assigned = torch.cat((assigned, reverse[name]), 1)
    for ordering, expected in (("raw", raw), ("forward", forward), ("reverse", reverse)):
        stored = payload["orthogonal"][ordering]
        for name in expected:
            need(torch.allclose(stored[name].double() @ stored[name].double().T,
                                expected[name] @ expected[name].T,
                                atol=1e-7, rtol=1e-7),
                 f"{ordering} {name} orthogonal basis does not recompute")
    for ordering, expected in (("forward", forward_spectra),
                               ("reverse", reverse_spectra)):
        stored = payload["orthogonal"][f"{ordering}_eigenvalues"]
        need(stored.keys() == expected.keys()
             and all(torch.allclose(stored[name].double(), values,
                                    atol=1e-8, rtol=1e-7)
                     for name, values in expected.items()),
             f"{ordering} residualized spectra do not recompute")
    stored_angles = payload["orthogonal"]["raw_principal_angles"]
    expected_angles = {
        f"{left}:{right}": principal_angles(raw[left], raw[right])
        for left, right in (("content", "binding"), ("content", "order"),
                            ("binding", "order"))}
    need(stored_angles.keys() == expected_angles.keys()
         and all(torch.allclose(stored_angles[name].double(), angles,
                                atol=1e-8, rtol=1e-7)
                 for name, angles in expected_angles.items()),
         "Raw principal-angle diagnostics do not recompute")
    return payload, recomputed


def effect(rows):
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["group_id"], row["recipient"])
        need(key not in seen and row["recipient"] in (0, 1, 2),
             "Duplicate or malformed rank row")
        seen.add(key)
        grouped[row["group_id"]].append(row)
    need(rows and all({row["recipient"] for row in group} == {0, 1, 2}
                      for group in grouped.values()), "Incomplete rank recipient groups")
    correct = sum(row["prediction"] == row["target"] for row in rows)
    exact = sum(all(row["prediction"] == row["target"] for row in group)
                for group in grouped.values())
    changed = [row for row in rows if row["recipient"] != 0]
    non_injected = sum(row["prediction"] != row["fixed_donor_answer"] for row in changed)
    gains = [row["target_probability"] - row["base_target_probability"] for row in rows]
    return {"items": len(rows), "correct": correct, "item_accuracy": correct / len(rows),
            "groups": len(grouped), "exact_groups": exact,
            "group_accuracy": exact / len(grouped),
            "changed_recipient_items": len(changed), "non_injected": non_injected,
            "non_injection": non_injected / len(changed),
            "mean_probability_gain": sum(gains) / len(gains)}


def verify_rank_grid(rows, factor_name, groups):
    ranks = {row.get("rank") for row in rows}
    need("full" in ranks and all(rank == "full" or rank in CANDIDATE_RANKS for rank in ranks),
         "Rank grid has unregistered rank labels")
    expected = {}
    for rank in ranks:
        condition = f"{factor_name}_rank_{rank}"
        for direction, base_variant, donor_variant in FACTOR_PAIRS[factor_name]:
            for group in groups:
                base_rows, donor_rows = group[base_variant], group[donor_variant]
                for recipient in range(3):
                    base = base_rows[recipient]
                    donor = (donor_rows[0] if factor_name in ("content", "binding")
                             else donor_rows[recipient])
                    target = (group["recipient_answers"][recipient]
                              if factor_name in ("content", "binding") else base["answer"])
                    fixed = donor_rows[0]["answer"]
                    key = (rank, direction, group["group_id"], recipient)
                    expected[key] = {
                        "condition": condition, "direction": direction,
                        "logical_group_id": group["group_id"],
                        "group_id": f"{group['group_id']}:{direction}",
                        "recipient": recipient, "target": target,
                        "base_answer": base["answer"], "fixed_donor_answer": fixed,
                        "base_render_id": base["render_id"],
                        "donor_render_id": donor["render_id"],
                        "item_id": (f"{factor_name}:{condition}:{direction}:"
                                    f"{group['group_id']}:{recipient}"),
                    }
    observed = {}
    for row in rows:
        key = (row.get("rank"), row.get("direction"),
               row.get("logical_group_id"), row.get("recipient"))
        need(key not in observed, "Duplicate Stage-D rank-grid item")
        observed[key] = row
    need(observed.keys() == expected.keys(), "Stage-D rank grid is incomplete or contaminated")
    for key, registered in expected.items():
        row = observed[key]
        need(row.get("factor") == factor_name
             and all(row.get(name) == value for name, value in registered.items()),
             "Stage-D rank row is not bound to the frozen suite semantics")
    need(len({row["item_id"] for row in rows}) == len(rows),
         "Stage-D rank item IDs are not globally unique within factor")


def rank_summary(rows):
    by_rank = defaultdict(list)
    for row in rows:
        need(row.get("factor") in ("content", "binding", "order")
             and (row.get("rank") == "full" or type(row.get("rank")) is int),
             "Malformed factor/rank labels")
        by_rank[row["rank"]].append(row)
    need("full" in by_rank, "Rank rows lack full reference")
    return {str(rank): effect(selected) for rank, selected in sorted(
        by_rank.items(), key=lambda item: (-1 if item[0] == "full" else item[0]))}


def spectral_boundary_gap(spectrum, rank):
    spectrum = spectrum.double()
    need(0 < rank <= len(spectrum), "Rank lies outside residualized spectrum")
    if rank == len(spectrum):
        return 1.0
    current, following = float(spectrum[rank - 1]), float(spectrum[rank])
    return max(0.0, min(1.0, (current - following) / max(current, 1e-12)))


def joint_selection(summaries, factor, spectra=None):
    full = {seed: rows["full"]["mean_probability_gain"]
            for seed, rows in summaries.items()}
    if any(not math.isfinite(value) or value <= 0 for value in full.values()):
        return {"selection": None, "reason": "nonpositive_full_probability_gain",
                "full_probability_gain_by_seed": full}
    item_floor, group_floor = THRESHOLDS[factor]
    by_seed = {}
    for seed, rows in summaries.items():
        measurements = {}
        for rank in CANDIDATE_RANKS:
            if str(rank) not in rows:
                continue
            cell = rows[str(rank)]
            gap = None if spectra is None else spectral_boundary_gap(spectra[seed], rank)
            qualified = (cell["mean_probability_gain"] >= .95 * full[seed]
                         and cell["item_accuracy"] >= item_floor
                         and cell["group_accuracy"] >= group_floor
                         and (gap is None or gap >= MINIMUM_SPECTRAL_GAP))
            measurements[str(rank)] = {
                "rank": rank, "item_accuracy": cell["item_accuracy"],
                "group_accuracy": cell["group_accuracy"],
                "mean_probability_gain": cell["mean_probability_gain"],
                "qualified": qualified,
                "effect_fraction": cell["mean_probability_gain"] / full[seed]}
            if gap is not None:
                measurements[str(rank)]["spectral_boundary_gap"] = gap
        selected = next((rank for rank in CANDIDATE_RANKS
                         if measurements.get(str(rank), {}).get("qualified")), None)
        by_seed[seed] = {
            "candidate_ranks": list(CANDIDATE_RANKS), "effect_fraction_threshold": .95,
            "minimum_item_accuracy": item_floor,
            "minimum_group_accuracy": group_floor,
            "minimum_spectral_gap": None if spectra is None else MINIMUM_SPECTRAL_GAP,
            "full_probability_gain": full[seed], "measurements": measurements,
            "selection": selected}
    common = [rank for rank in CANDIDATE_RANKS
              if all(result["measurements"].get(str(rank), {}).get("qualified")
                     for result in by_seed.values())]
    return {"candidate_ranks": list(CANDIDATE_RANKS), "by_seed": by_seed,
            "selection": common[0] if common else None,
            "rule": "smallest_rank_qualified_in_every_seed"}


def linear_cka(left, right):
    left, right = left.double() - left.double().mean(0), right.double() - right.double().mean(0)
    cross = torch.linalg.norm(left.T @ right).square()
    denominator = torch.linalg.norm(left.T @ left) * torch.linalg.norm(right.T @ right)
    return math.nan if denominator == 0 else float(cross / denominator)


def procrustes(source, target):
    source, target = source.double(), target.double()
    source_mean, target_mean = source.mean(0), target.mean(0)
    left, right = source - source_mean, target - target_mean
    u, _, vh = torch.linalg.svd(left.T @ right, full_matrices=False)
    rotation = u @ vh
    denominator = torch.linalg.norm(right)
    residual = math.inf if denominator == 0 else float(
        torch.linalg.norm(left @ rotation - right) / denominator)
    return {"source_mean": source_mean, "target_mean": target_mean,
            "rotation": rotation, "normalized_residual": residual}


def verify_procrustes_map(actual, source, target, name):
    source, target = source.double(), target.double()
    source_mean, target_mean = source.mean(0), target.mean(0)
    rotation = actual["rotation"].double()
    need(torch.allclose(rotation.T @ rotation,
                        torch.eye(rotation.shape[0], dtype=torch.double),
                        atol=1e-7, rtol=1e-7)
         and torch.allclose(actual["source_mean"].double(), source_mean, atol=1e-8)
         and torch.allclose(actual["target_mean"].double(), target_mean, atol=1e-8),
         f"{name} affine map is malformed")
    left, right = source - source_mean, target - target_mean
    residual = float(torch.linalg.norm(left @ rotation - right) / torch.linalg.norm(right))
    optimum = procrustes(source, target)["normalized_residual"]
    need(math.isclose(actual["normalized_residual"], residual, abs_tol=1e-10)
         and math.isclose(residual, optimum, abs_tol=1e-10),
         f"{name} does not attain the Procrustes optimum")


def verify_cross_seed(metadata, geometry_payloads):
    path = ROOT / metadata["path"]
    need(path.is_file() and path.stat().st_size == metadata["bytes"]
         and file_digest(path) == metadata["sha256"], "Cross-seed artifact mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    need(payload.get("format") == "TEACH-0013-cross-seed-geometry-v2",
         "Unknown cross-seed format")
    for factor in ("content", "binding", "order"):
        left, right = (geometry_payloads[seed]["panels"][factor]["states"]
                       for seed in ("0", "1"))
        stored = payload["alignment"][factor]
        need(math.isclose(stored["linear_cka"], linear_cka(left, right), abs_tol=1e-10)
             and torch.allclose(stored["principal_angles"], principal_angles(
                 geometry_payloads["0"]["geometry"][factor]["basis"],
                 geometry_payloads["1"]["geometry"][factor]["basis"]), atol=1e-8),
             f"{factor} cross-seed geometry does not recompute")
        for name, source, target in (("seed0_to_seed1", left, right),
                                     ("seed1_to_seed0", right, left)):
            actual = stored[name]
            verify_procrustes_map(actual, source, target, f"{factor} {name}")
        expected_aligned = principal_angles(
            stored["seed0_to_seed1"]["rotation"].double().T
            @ geometry_payloads["0"]["geometry"][factor]["basis"].double(),
            geometry_payloads["1"]["geometry"][factor]["basis"].double())
        need(torch.allclose(stored["aligned_principal_angles"].double(),
                            expected_aligned, atol=1e-8),
             f"{factor} aligned principal angles do not recompute")

    factors = ("content", "binding", "order")

    def shared_map(right_block_map=None):
        left_rows, right_rows, means = [], [], {}
        for factor in factors:
            left_panel = geometry_payloads["0"]["panels"][factor]
            right_panel = geometry_payloads["1"]["panels"][factor]
            right_states = right_panel["states"]
            if right_block_map is not None:
                lookup = {(block, nuisance, level): index for index, (
                    block, nuisance, level) in enumerate(zip(
                        right_panel["blocks"], right_panel["nuisance"],
                        right_panel["factor"], strict=True))}
                indices = [lookup[(right_block_map[block], nuisance, level)]
                           for block, nuisance, level in zip(
                               left_panel["blocks"], left_panel["nuisance"],
                               left_panel["factor"], strict=True)]
                right_states = right_states[indices]
            left_mean, right_mean = left_panel["states"].mean(0), right_states.mean(0)
            scale = len(left_panel["states"]) ** .5
            left_rows.append((left_panel["states"] - left_mean) / scale)
            right_rows.append((right_states - right_mean) / scale)
            means[factor] = {"seed0": left_mean, "seed1": right_mean}
        left, right = torch.cat(left_rows), torch.cat(right_rows)
        mapping = procrustes(left, right)
        return {"rotation": mapping["rotation"],
                "normalized_residual": mapping["normalized_residual"],
                "source": left, "target": right,
                "factor_means": means,
                "aligned_principal_angles": {factor: principal_angles(
                    mapping["rotation"].T
                    @ geometry_payloads["0"]["orthogonal"]["forward"][factor].double(),
                    geometry_payloads["1"]["orthogonal"]["forward"][factor].double())
                    for factor in factors}}

    blocks = sorted(set(geometry_payloads["0"]["panels"]["content"]["blocks"]))
    rng, donors = random.Random(73341), list(blocks)
    for right in range(len(donors) - 1, 0, -1):
        left = rng.randrange(right)
        donors[left], donors[right] = donors[right], donors[left]
    expected_maps = {
        "shared": shared_map(),
        "shared_deranged_control": shared_map(dict(zip(blocks, donors, strict=True))),
    }
    for name, expected in expected_maps.items():
        stored = payload["alignment"][name]
        verify_procrustes_map({"rotation": stored["rotation"],
                               "source_mean": expected["source"].mean(0),
                               "target_mean": expected["target"].mean(0),
                               "normalized_residual": stored["normalized_residual"]},
                              expected["source"], expected["target"], name)
        need(all(torch.allclose(
                 stored["aligned_principal_angles"][factor].double(),
                 principal_angles(
                     stored["rotation"].double().T
                     @ geometry_payloads["0"]["orthogonal"]["forward"][factor].double(),
                     geometry_payloads["1"]["orthogonal"]["forward"][factor].double()),
                 atol=1e-8) for factor in factors),
             f"{name} shared aligned angles do not recompute")
        for factor in factors:
            for seed in ("seed0", "seed1"):
                need(torch.allclose(stored["factor_means"][factor][seed].double(),
                                    expected["factor_means"][factor][seed].double(), atol=1e-8),
                     f"{name} factor means do not recompute")
    control = payload["alignment"]["shared_deranged_control"]
    need(control.get("seed") == 73341 and tuple(control.get("donor_blocks", ())) == tuple(donors)
         and all(left != right for left, right in zip(blocks, donors, strict=True)),
         "Shared deranged-map control is malformed")


def auditor_provenance():
    relative = "scripts/teacher0013_subspace_audit.py"
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", relative], cwd=ROOT,
        check=True, capture_output=True, text=True).stdout.strip()
    need(not status, "Commit Stage-D auditor before audit")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    return {"revision": revision, "sha256": {relative: file_digest(ROOT / relative)}}


def audit(report_path: Path, output_path: Path) -> dict:
    report = json.loads(report_path.read_text())
    need(report.get("experiment") == "TEACH-0013"
         and report.get("status") == "stage_d_discovery_complete",
         "Not a complete Stage-D discovery report")
    confirmation_path = report_path.parent / "confirmation-decision.json"
    confirmation_audit_path = report_path.parent / "confirmation-audit.json"
    manifest_path = report_path.parent / "suite-manifest.json"
    need(confirmation_path.is_file() and confirmation_audit_path.is_file()
         and manifest_path.is_file()
         and file_digest(confirmation_path) == report["confirmation_decision_sha256"]
         and file_digest(confirmation_audit_path) == report["confirmation_audit_sha256"],
         "Stage-D prerequisites no longer match the sealed Stage-C artifacts")
    confirmation_audit = json.loads(confirmation_audit_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    need(confirmation_audit.get("audit") == "pass"
         and confirmation_audit.get("scope") == "stage_c_confirmation"
         and confirmation_audit.get("confirmation_decision_sha256")
         == report["confirmation_decision_sha256"],
         "Stage-D prerequisite audit did not independently pass Stage C")
    need(manifest.get("status") == "suite_frozen"
         and report.get("suite_gzip_sha256") == manifest.get("suite_gzip_sha256")
         and report.get("confirmation_source_git_head") == manifest.get("source_git_head")
         and report.get("confirmation_source_sha256") == manifest.get("source_sha256"),
         "Stage-D report is not bound to the frozen suite source")
    suite_path = ROOT / manifest["suite_path"]
    need(suite_path.is_file() and suite_path.stat().st_size == manifest["suite_gzip_bytes"]
         and file_digest(suite_path) == manifest["suite_gzip_sha256"],
         "Frozen suite artifact mismatch")
    import gzip
    suite = json.loads(gzip.decompress(suite_path.read_bytes()))
    discovery_groups = suite["splits"]["discovery"]
    need([group["group_id"] for group in discovery_groups]
         == manifest["group_ids"]["discovery"], "Frozen discovery group order changed")
    exact = _load_confirmation_auditor()
    geometry_payloads, verified, summaries = {}, {}, {"0": {}, "1": {}}
    for seed in ("0", "1"):
        geometry_payloads[seed], _ = verify_geometry_artifact(
            report["artifacts"][seed]["geometry"], discovery_groups, report["mediator"])
        verified[f"{seed}:geometry"] = report["artifacts"][seed]["geometry"]["sha256"]
        for factor, metadata in report["artifacts"][seed]["rank_rows"].items():
            path = ROOT / metadata["path"]
            need(path.is_file() and path.stat().st_size == metadata["bytes"]
                 and file_digest(path) == metadata["sha256"], "Rank-row artifact mismatch")
            rows = load_rows(path)
            need(len(rows) == metadata["rows"], "Rank-row count mismatch")
            verify_rank_grid(rows, factor, discovery_groups)
            summaries[seed][factor] = rank_summary(rows)
            exact_result = exact.verify_logit_archive(metadata["exact_symbol_logits"], rows)
            verified[f"{seed}:{factor}"] = {
                "rows_sha256": metadata["sha256"], "exact_logits": exact_result}
    need(summaries == report["summaries"], "Rank summaries do not independently recompute")
    selections = {}
    for factor in ("content", "binding", "order"):
        if any(factor not in summaries[seed] for seed in ("0", "1")):
            selections[factor] = {"selection": None, "reason": "empty_orthogonal_basis"}
        else:
            selections[factor] = joint_selection(
                {seed: summaries[seed][factor] for seed in ("0", "1")}, factor,
                {seed: geometry_payloads[seed]["orthogonal"]["forward_eigenvalues"][factor]
                 for seed in ("0", "1")})
    need(selections == report["rank_selections"],
         "Joint factor-rank selections do not independently recompute")
    verify_cross_seed(report["artifacts"]["cross_seed"], geometry_payloads)
    need(report["cumulative_stage_c_e_seconds"] <= STAGE_C_E_SECONDS
         and report["cumulative_campaign_seconds"] <= CAMPAIGN_SECONDS
         and report["peak_sampled_current_allocated_bytes"] <= MAX_CURRENT_BYTES
         and report["cumulative_materialized_bytes"] <= MAX_TRAFFIC_BYTES,
         "Stage-D resource ceiling exceeded")
    result = {"audit": "pass", "experiment": "TEACH-0013",
              "scope": "stage_d_discovery", "status": report["status"],
              "subspace_discovery_sha256": file_digest(report_path),
              "rank_selections": selections, "verified_artifacts": verified,
              "auditor": auditor_provenance()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path,
                        default=ROOT / "results/TEACH-0013/subspace-discovery.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/TEACH-0013/subspace-discovery-audit.json")
    args = parser.parse_args()
    print(json.dumps(audit(args.report, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
