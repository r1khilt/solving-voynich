#!/usr/bin/env python3
"""Independent no-model audit of TEACH-0013 Stage-D discovery geometry."""

import argparse
from collections import defaultdict
import importlib.util
import json
import math
from pathlib import Path
import subprocess

import torch


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_RANKS = (1, 2, 4, 8, 16, 32, 64)
THRESHOLDS = {"content": (.80, .60), "binding": (.75, .55), "order": (.75, .55)}
STAGE_C_E_SECONDS = 5400.0
CAMPAIGN_SECONDS = 14_400.0
MAX_CURRENT_BYTES = 24 * 1024**3
MAX_TRAFFIC_BYTES = 300 * 1024**3


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


def orthogonalize(basis, against):
    basis, against = basis.double(), against.double()
    remainder = basis - against @ (against.T @ basis) if against.shape[1] else basis
    if remainder.shape[1] == 0:
        return remainder
    u, singular, _ = torch.linalg.svd(remainder, full_matrices=False)
    keep = singular > 1e-8 * max(float(singular[0]), 1.0)
    return u[:, keep]


def principal_angles(left, right):
    if not left.shape[1] or not right.shape[1]:
        return torch.empty(0, dtype=torch.double)
    return torch.linalg.svdvals(left.double().T @ right.double()).clamp(0, 1).acos()


def verify_geometry_artifact(metadata: dict) -> tuple[dict, dict]:
    path = ROOT / metadata["path"]
    need(path.is_file() and path.stat().st_size == metadata["bytes"]
         and file_digest(path) == metadata["sha256"], "Geometry artifact mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    need(payload.get("format") == "TEACH-0013-factor-geometry-v1"
         and set(payload.get("panels", {})) == {"content", "binding", "order"},
         "Unknown factor-geometry format")
    recomputed = {name: blocked_geometry(payload["panels"][name])
                  for name in ("content", "binding", "order")}
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
    forward, assigned = {}, torch.empty(raw["content"].shape[0], 0, dtype=torch.double)
    for name in ("content", "binding", "order"):
        forward[name] = orthogonalize(raw[name], assigned)
        assigned = torch.cat((assigned, forward[name]), 1)
    reverse, assigned = {}, torch.empty(raw["content"].shape[0], 0, dtype=torch.double)
    for name in ("order", "binding", "content"):
        reverse[name] = orthogonalize(raw[name], assigned)
        assigned = torch.cat((assigned, reverse[name]), 1)
    for ordering, expected in (("raw", raw), ("forward", forward), ("reverse", reverse)):
        stored = payload["orthogonal"][ordering]
        for name in expected:
            need(torch.allclose(stored[name].double() @ stored[name].double().T,
                                expected[name] @ expected[name].T,
                                atol=1e-7, rtol=1e-7),
                 f"{ordering} {name} orthogonal basis does not recompute")
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


def joint_selection(summaries, factor):
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
            qualified = (cell["mean_probability_gain"] >= .95 * full[seed]
                         and cell["item_accuracy"] >= item_floor
                         and cell["group_accuracy"] >= group_floor)
            measurements[str(rank)] = {
                "rank": rank, "item_accuracy": cell["item_accuracy"],
                "group_accuracy": cell["group_accuracy"],
                "mean_probability_gain": cell["mean_probability_gain"],
                "qualified": qualified,
                "effect_fraction": cell["mean_probability_gain"] / full[seed]}
        selected = next((rank for rank in CANDIDATE_RANKS
                         if measurements.get(str(rank), {}).get("qualified")), None)
        by_seed[seed] = {
            "candidate_ranks": list(CANDIDATE_RANKS), "effect_fraction_threshold": .95,
            "minimum_item_accuracy": item_floor,
            "minimum_group_accuracy": group_floor,
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


def verify_cross_seed(metadata, geometry_payloads):
    path = ROOT / metadata["path"]
    need(path.is_file() and path.stat().st_size == metadata["bytes"]
         and file_digest(path) == metadata["sha256"], "Cross-seed artifact mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    need(payload.get("format") == "TEACH-0013-cross-seed-geometry-v1",
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
            expected = procrustes(source, target)
            actual = stored[name]
            need(torch.allclose(actual["source_mean"], expected["source_mean"], atol=1e-8)
                 and torch.allclose(actual["target_mean"], expected["target_mean"], atol=1e-8)
                 and torch.allclose(actual["rotation"], expected["rotation"], atol=1e-7)
                 and math.isclose(actual["normalized_residual"],
                                  expected["normalized_residual"], abs_tol=1e-10),
                 f"{factor} {name} Procrustes map does not recompute")


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
    exact = _load_confirmation_auditor()
    geometry_payloads, verified, summaries = {}, {}, {"0": {}, "1": {}}
    for seed in ("0", "1"):
        geometry_payloads[seed], _ = verify_geometry_artifact(
            report["artifacts"][seed]["geometry"])
        verified[f"{seed}:geometry"] = report["artifacts"][seed]["geometry"]["sha256"]
        for factor, metadata in report["artifacts"][seed]["rank_rows"].items():
            path = ROOT / metadata["path"]
            need(path.is_file() and path.stat().st_size == metadata["bytes"]
                 and file_digest(path) == metadata["sha256"], "Rank-row artifact mismatch")
            rows = load_rows(path)
            need(len(rows) == metadata["rows"], "Rank-row count mismatch")
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
                {seed: summaries[seed][factor] for seed in ("0", "1")}, factor)
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
